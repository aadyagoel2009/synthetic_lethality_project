"""
Genome-wide top-10 SLI predictions (transfer-mut + expression filter + mut QC).

Refinement of sli_jst_pcc_split_top10_predictions_transfer_mut.py:

  Expression:
    S_exp = C_sclc * (1 + lambda * max(0, zscore(transfer_AGOP)))

  Mutation:
    S_mut = transfer_AGOP * C_pan

  Merge for ranking only:
    within-type percentile ranks

  Expression prevalence filter (unchanged):
    Keep partners with log1p(TPM) >= 1 in >= 25% of SCLC lines.

  Score cutoffs (manual, BELOW the previous elbows):
    Prior elbows were ~0.822 (exp) and ~0.00326 (mut).
    This variant uses lower fixed cutoffs:
      SCORE_CUTOFF_EXP = 0.5
      SCORE_CUTOFF_MUT = 0.001

  Mutation-pair QC (SCLC only), applied before exporting a mut pair:
    1. At least one SCLC line has the partner mutation AND at least one does not
    2. Median Chronos of the KO in partner-MUT SCLC lines is < -0.25

Run:
  python sli_jst_pcc_split_top10_predictions_transfer_mut_refined.py
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch

import sli_jst_pcc_split_pipeline as split
from plot_sclc_ko_by_partner_mutation import (
    load_crispr_effects,
    load_mutation_matrix,
)


# ---------------------------------------------------------------------------
# Fixed method
# ---------------------------------------------------------------------------
GAMMA = 10.0
LAMBDA = 0.05
METHOD = "split_g10_lam0.05"
VARIANT = "transfer_mut_expfilter_refined"

TOP_K_GENES = 10

# Reference elbows from transfer_mut_expfilter ElbowScore sheet.
REFERENCE_ELBOW_EXP = 0.8221797922063231
REFERENCE_ELBOW_MUT = 0.003259949791476882

# Manual cutoffs, lowered relative to the elbows above.
SCORE_CUTOFF_EXP = 0.5
SCORE_CUTOFF_MUT = 0.001

MIN_LOG_TPM = 1.0
MIN_EXPRESSED_FRAC = 0.25
EXPRESSION_CSV = "datasets/2023/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"

# Mutation-pair QC in SCLC.
MIN_MUT_LINES = 1
MIN_WT_LINES = 1
MAX_MEDIAN_MUT_CHRONOS = -0.25

RESULTS_DIR = split.RESULTS_DIR
STRUCTURE_CACHE = os.path.join(RESULTS_DIR, "structure_g10_all_kos.pkl")
FULL_SCORE_CACHE = os.path.join(RESULTS_DIR, f"fi_{METHOD}_{VARIANT}.pkl")

OUTPUT_CSV = os.path.join(
    RESULTS_DIR, f"top10_SLI_predictions_{METHOD}_{VARIANT}.csv"
)
OUTPUT_EXP_CSV = os.path.join(
    RESULTS_DIR, f"top10_SLI_predictions_{METHOD}_{VARIANT}_exp.csv"
)
OUTPUT_MUT_CSV = os.path.join(
    RESULTS_DIR, f"top10_SLI_predictions_{METHOD}_{VARIANT}_mut.csv"
)
OUTPUT_THRESHOLD_CSV = os.path.join(
    RESULTS_DIR, f"score_thresholds_{METHOD}_{VARIANT}.csv"
)
OUTPUT_XLSX = os.path.join(
    RESULTS_DIR, f"top10_SLI_predictions_{METHOD}_{VARIANT}.xlsx"
)

SAVE_FULL_SCORE_MATRIX = True


def partner_gene(feature: str) -> str:
    """Map GENE_exp and GENE mutation features to the gene symbol."""
    return re.sub(r"_exp$", "", str(feature))


def feature_type(feature: str) -> str:
    return "exp" if str(feature).endswith("_exp") else "mut"


def paper_top_scores(channel: pd.DataFrame) -> pd.Series:
    """Paper KO scores on one channel: max(v) - mean(v)."""
    scores = channel.max(axis=0) - channel.mean(axis=0)
    return scores.replace([np.inf, -np.inf], np.nan).dropna()


def load_or_compute_structure(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    p_pan: np.ndarray,
    omega: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    """Load or compute gamma=10 transfer AGOP for every KO."""
    if os.path.exists(STRUCTURE_CACHE):
        print(f"Loading cached transfer structure from {STRUCTURE_CACHE}")
        structure = pd.read_pickle(STRUCTURE_CACHE)
        missing_kos = pan_effects.columns.difference(structure.columns)
        if len(missing_kos) == 0:
            return structure.reindex(
                index=pan_embedding.columns,
                columns=pan_effects.columns,
            ).fillna(0.0)
        print(
            f"Structure cache is missing {len(missing_kos)} KOs; recomputing all."
        )

    print(
        f"Computing gamma={GAMMA:g} transfer structure for "
        f"{pan_effects.shape[1]} KOs. This is the slow step."
    )
    structure = split.compute_jst_structure(
        pan_embedding,
        pan_effects,
        device,
        p_pan,
        omega,
        GAMMA,
    )
    structure.to_pickle(STRUCTURE_CACHE)
    print(f"Saved structure cache: {STRUCTURE_CACHE}")
    return structure


def compute_mut_transfer_x_pan_pcc(
    structure: pd.DataFrame,
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
) -> pd.DataFrame:
    """Mutation channel: S_mut = transfer_AGOP * C_pan."""
    pcc_pan = split.process_pcc_for_sli(
        split.get_pcc(pan_embedding, pan_effects)
    )
    common_idx = structure.index.intersection(pcc_pan.index)
    common_cols = structure.columns.intersection(pcc_pan.columns)
    agop = structure.loc[common_idx, common_cols].astype(float)
    c_pan = pcc_pan.loc[common_idx, common_cols].astype(float).fillna(0.0)
    return (agop * c_pan).fillna(0.0)


def load_log_tpm_expression(path: str = EXPRESSION_CSV) -> pd.DataFrame:
    """Load DepMap log1p(TPM) expression; columns are gene symbols."""
    print(f"Loading log1p(TPM) expression from {path}")
    exp_df = pd.read_csv(path)
    exp_df.columns = [str(c).split()[0] for c in exp_df.columns]
    if "ModelID" not in exp_df.columns:
        raise ValueError(f"'ModelID' not found in {path}")
    exp_df = exp_df.set_index("ModelID")
    exp_df = exp_df.apply(pd.to_numeric, errors="coerce")
    exp_df = exp_df.groupby(level=0).mean()
    return exp_df


def genes_passing_expression_filter(
    partner_genes: List[str],
    log_tpm: pd.DataFrame,
    cell_ids: pd.Index,
    min_log_tpm: float = MIN_LOG_TPM,
    min_frac: float = MIN_EXPRESSED_FRAC,
) -> Set[str]:
    """
    Keep genes with log1p(TPM) >= min_log_tpm in at least min_frac of cell_ids.
    """
    cells = cell_ids.intersection(log_tpm.index)
    if len(cells) == 0:
        raise ValueError("No overlapping cell IDs between filter lines and expression.")

    min_lines = int(np.ceil(min_frac * len(cells)))
    print(
        f"  Expression filter: log1p(TPM) >= {min_log_tpm} in >= {min_frac:.0%} "
        f"of {len(cells)} lines (need >= {min_lines} lines)"
    )

    exp = log_tpm.loc[cells]
    available = set(exp.columns.astype(str))
    kept: Set[str] = set()
    missing = 0
    failed = 0

    for gene in partner_genes:
        gene_str = str(gene)
        if gene_str not in available:
            missing += 1
            continue
        n_pass = int((exp[gene_str] >= min_log_tpm).sum())
        if n_pass >= min_lines:
            kept.add(gene_str)
        else:
            failed += 1

    print(
        f"  Partners kept={len(kept)}, failed_prevalence={failed}, "
        f"missing_from_expression={missing}"
    )
    return kept


def filter_features_by_partner_expression(
    exp_channel: pd.DataFrame,
    mut_channel: pd.DataFrame,
    merged_ranks: pd.DataFrame,
    allowed_genes: Set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Drop features whose partner gene fails the expression prevalence filter."""
    keep_features = [
        f for f in exp_channel.index if partner_gene(str(f)) in allowed_genes
    ]
    keep_features = [f for f in keep_features if f in mut_channel.index]
    keep_features = [f for f in keep_features if f in merged_ranks.index]

    print(
        f"  Features after expression filter: {len(keep_features)} / "
        f"{len(exp_channel.index)}"
    )
    if len(keep_features) == 0:
        raise ValueError("Expression filter removed all features.")

    return (
        exp_channel.loc[keep_features],
        mut_channel.loc[keep_features],
        merged_ranks.loc[keep_features],
    )


def mutation_pair_passes_sclc_qc(
    ko_gene: str,
    partner: str,
    mutations: pd.DataFrame,
    effects: pd.DataFrame,
    sclc_ids: pd.Index,
) -> Tuple[bool, Dict]:
    """
    Keep a mutation pair only if SCLC has both MUT and WT for the partner,
    and median Chronos of the KO in partner-MUT lines is < -0.25.
    """
    info = {
        "n_mut_sclc": 0,
        "n_wt_sclc": 0,
        "median_mut_chronos": float("nan"),
    }

    if partner not in mutations.columns:
        return False, info
    if ko_gene not in effects.columns:
        return False, info

    cells = sclc_ids.intersection(mutations.index).intersection(effects.index)
    if len(cells) == 0:
        return False, info

    mut = mutations.loc[cells, partner].astype(int)
    chronos = effects.loc[cells, ko_gene].astype(float)
    valid = chronos.notna()
    mut = mut.loc[valid]
    chronos = chronos.loc[valid]

    n_mut = int((mut.values == 1).sum())
    n_wt = int((mut.values == 0).sum())
    info["n_mut_sclc"] = n_mut
    info["n_wt_sclc"] = n_wt

    if n_mut < MIN_MUT_LINES or n_wt < MIN_WT_LINES:
        return False, info

    median_mut = float(chronos.loc[mut.values == 1].median())
    info["median_mut_chronos"] = median_mut
    if not np.isfinite(median_mut) or median_mut >= MAX_MEDIAN_MUT_CHRONOS:
        return False, info

    return True, info


def build_top10_predictions(
    exp_channel: pd.DataFrame,
    mut_channel: pd.DataFrame,
    merged_ranks: pd.DataFrame,
    mutations: pd.DataFrame,
    effects: pd.DataFrame,
    sclc_ids: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank by split percentile merge; apply lowered fixed type cutoffs.
    Mutation candidates also pass SCLC MUT/WT + median Chronos QC.
    """
    common_idx = (
        exp_channel.index.intersection(mut_channel.index).intersection(
            merged_ranks.index
        )
    )
    common_cols = (
        exp_channel.columns.intersection(mut_channel.columns).intersection(
            merged_ranks.columns
        )
    )
    exp_idx, mut_idx = split.split_feature_index(common_idx)

    exp_ch = exp_channel.loc[common_idx, common_cols].astype(float)
    mut_ch = mut_channel.loc[common_idx, common_cols].astype(float)
    ranks = merged_ranks.loc[common_idx, common_cols].astype(float)

    top_exp_scores = (
        paper_top_scores(exp_ch.loc[exp_idx]) if exp_idx else pd.Series(dtype=float)
    )
    top_mut_scores = (
        paper_top_scores(mut_ch.loc[mut_idx]) if mut_idx else pd.Series(dtype=float)
    )

    if top_exp_scores.empty and top_mut_scores.empty:
        raise ValueError("No expression or mutation features available for scoring.")

    if len(top_exp_scores) and top_exp_scores.nunique(dropna=True) <= 1:
        raise ValueError("Expression KO scores are identical; check exp channel.")
    if len(top_mut_scores) and top_mut_scores.nunique(dropna=True) <= 1:
        raise ValueError("Mutation KO scores are identical; check mut channel.")

    print(
        f"  Exp raw scale (median KO max): "
        f"{float(exp_ch.loc[exp_idx].max(axis=0).median()):.6g}"
        if exp_idx
        else "  No exp features"
    )
    print(
        f"  Mut raw scale (median KO max): "
        f"{float(mut_ch.loc[mut_idx].max(axis=0).median()):.6g}"
        if mut_idx
        else "  No mut features"
    )
    print(
        f"  Reference elbows: exp={REFERENCE_ELBOW_EXP:.6g}, "
        f"mut={REFERENCE_ELBOW_MUT:.6g}"
    )
    print(
        f"  Using lowered cutoffs: exp={SCORE_CUTOFF_EXP:.6g}, "
        f"mut={SCORE_CUTOFF_MUT:.6g}"
    )
    print(
        f"  Mut QC: >= {MIN_MUT_LINES} MUT and >= {MIN_WT_LINES} WT SCLC lines; "
        f"median MUT Chronos < {MAX_MEDIAN_MUT_CHRONOS}"
    )

    prediction_rows: List[dict] = []
    n_mut_fail_balance = 0
    n_mut_fail_chronos = 0
    n_mut_pass_qc = 0

    for ko_gene in common_cols:
        ranked = (
            ranks[ko_gene]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values(ascending=False)
        )
        if ranked.empty:
            continue

        exp_mean = float(exp_ch.loc[exp_idx, ko_gene].mean()) if exp_idx else 0.0
        mut_mean = float(mut_ch.loc[mut_idx, ko_gene].mean()) if mut_idx else 0.0

        kept = 0
        seen_genes: set[str] = set()

        for feature, percentile_score in ranked.items():
            feature_str = str(feature)
            gene = partner_gene(feature_str)
            if gene in seen_genes:
                continue
            seen_genes.add(gene)

            ftype = feature_type(feature_str)
            if ftype == "exp":
                raw_value = float(exp_ch.at[feature, ko_gene])
                candidate_score = raw_value - exp_mean
                cutoff = SCORE_CUTOFF_EXP
                type_mean = exp_mean
            else:
                raw_value = float(mut_ch.at[feature, ko_gene])
                candidate_score = raw_value - mut_mean
                cutoff = SCORE_CUTOFF_MUT
                type_mean = mut_mean

            if not np.isfinite(candidate_score) or candidate_score < cutoff:
                continue

            qc_info = {
                "n_mut_sclc": np.nan,
                "n_wt_sclc": np.nan,
                "median_mut_chronos": np.nan,
            }
            if ftype == "mut":
                ok, qc_info = mutation_pair_passes_sclc_qc(
                    str(ko_gene),
                    gene,
                    mutations,
                    effects,
                    sclc_ids,
                )
                if not ok:
                    if (
                        qc_info["n_mut_sclc"] < MIN_MUT_LINES
                        or qc_info["n_wt_sclc"] < MIN_WT_LINES
                    ):
                        n_mut_fail_balance += 1
                    else:
                        n_mut_fail_chronos += 1
                    continue
                n_mut_pass_qc += 1

            if kept >= TOP_K_GENES:
                break

            kept += 1
            prediction_rows.append(
                {
                    "ko_gene": str(ko_gene),
                    "rank_in_ko": kept,
                    "partner_gene": gene,
                    "ko_score": candidate_score,
                    "feature": feature_str,
                    "feature_type": ftype,
                    "feature_importance": raw_value,
                    "mean_type_importance": type_mean,
                    "merged_percentile_rank": float(percentile_score),
                    "score_cutoff": cutoff,
                    "n_mut_sclc": qc_info["n_mut_sclc"],
                    "n_wt_sclc": qc_info["n_wt_sclc"],
                    "median_mut_chronos": qc_info["median_mut_chronos"],
                    "method": f"{METHOD}_{VARIANT}",
                }
            )

    print(
        f"  Mut QC tallies: passed={n_mut_pass_qc}, "
        f"fail_mut/wt_balance={n_mut_fail_balance}, "
        f"fail_median_chronos={n_mut_fail_chronos}"
    )

    predictions = pd.DataFrame(prediction_rows)
    if not predictions.empty:
        predictions = predictions.sort_values(
            ["feature_type", "ko_score", "ko_gene", "rank_in_ko"],
            ascending=[True, False, True, True],
        ).reset_index(drop=True)

    threshold_result = pd.DataFrame(
        {
            "reference_elbow_exp": [REFERENCE_ELBOW_EXP],
            "reference_elbow_mut": [REFERENCE_ELBOW_MUT],
            "score_cutoff_exp": [SCORE_CUTOFF_EXP],
            "score_cutoff_mut": [SCORE_CUTOFF_MUT],
            "min_log_tpm": [MIN_LOG_TPM],
            "min_expressed_frac": [MIN_EXPRESSED_FRAC],
            "expression_filter_lines": ["sclc"],
            "min_mut_lines_sclc": [MIN_MUT_LINES],
            "min_wt_lines_sclc": [MIN_WT_LINES],
            "max_median_mut_chronos": [MAX_MEDIAN_MUT_CHRONOS],
        }
    )
    return predictions, threshold_result


def split_predictions_by_type(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split mixed predictions into exp/mut sheets, each sorted by ko_score."""
    if predictions.empty:
        empty = predictions.copy()
        return empty, empty

    exp_preds = (
        predictions.loc[predictions["feature_type"] == "exp"]
        .sort_values(
            ["ko_score", "ko_gene", "rank_in_ko"], ascending=[False, True, True]
        )
        .reset_index(drop=True)
    )
    mut_preds = (
        predictions.loc[predictions["feature_type"] == "mut"]
        .sort_values(
            ["ko_score", "ko_gene", "rank_in_ko"], ascending=[False, True, True]
        )
        .reset_index(drop=True)
    )
    return exp_preds, mut_preds


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Method: {METHOD}_{VARIANT}")
    print("Mut channel: transfer_AGOP * C_pan")
    print("Exp channel: C_sclc * (1 + lambda * max(0, z(transfer_AGOP)))")
    print(
        f"Cutoffs (lowered vs elbows {REFERENCE_ELBOW_EXP:.4g}/"
        f"{REFERENCE_ELBOW_MUT:.4g}): "
        f"exp>={SCORE_CUTOFF_EXP}, mut>={SCORE_CUTOFF_MUT}"
    )

    pan_embedding = split.ensure_row_l2_normalized(
        split.load_hkl_dataframe(split.PAN_EMBED_HKL, "pan")
    )
    pan_effects = split.load_hkl_dataframe(
        split.PAN_GENE_EFFECT_HKL, "pan_effects"
    )
    sclc_embedding = split.ensure_row_l2_normalized(
        split.load_hkl_dataframe(split.SCLC_EMBED_HKL, "sclc")
    )
    sclc_effects = split.load_hkl_dataframe(
        split.SCLC_GENE_EFFECT_HKL, "sclc_effects"
    )

    (
        pan_embedding,
        pan_effects,
        sclc_embedding,
        sclc_effects,
    ) = split.align_embeddings_and_effects(
        pan_embedding,
        pan_effects,
        sclc_embedding,
        sclc_effects,
    )

    split.validate_sclc_in_pan(pan_embedding, sclc_embedding)
    sclc_meta, lung_meta = split.load_depmap_lineage_ids()
    sclc_in_pan, lung_in_pan = split.intersect_lineage_with_pan(
        pan_embedding.index,
        sclc_meta,
        lung_meta,
    )
    split.print_dataset_diagnostics(
        pan_embedding,
        pan_effects,
        sclc_in_pan,
        lung_in_pan,
        device,
    )

    print("\n=== SCLC expression context (C_sclc) ===")
    pcc_sclc = split.process_pcc_for_sli(
        split.get_pcc(sclc_embedding, sclc_effects)
    ).reindex(
        index=pan_embedding.columns,
        columns=pan_effects.columns,
    ).fillna(0.0)

    print("\n=== SCLC alignment and transfer structure ===")
    p_pan = split.load_or_compute_p_pan(
        pan_embedding,
        pan_effects,
        device,
    )
    omega = split.compute_kmm_weights(
        pan_embedding,
        sclc_in_pan,
        lung_in_pan,
    )
    structure = load_or_compute_structure(
        pan_embedding,
        pan_effects,
        p_pan.values,
        omega,
        device,
    )

    print("\n=== Mutation channel: transfer_AGOP * C_pan ===")
    mut_channel = compute_mut_transfer_x_pan_pcc(
        structure,
        pan_embedding,
        pan_effects,
    ).reindex(
        index=pcc_sclc.index,
        columns=pcc_sclc.columns,
    ).fillna(0.0)

    print("\n=== Expression channel (regular split ranking) ===")
    exp_channel = split.compute_exp_jst_pcc_scores(
        pcc_sclc,
        structure,
        LAMBDA,
    )
    merged_ranks = split.merge_type_split_scores(exp_channel, mut_channel)
    print(
        f"Channels before filter: exp={exp_channel.shape}, "
        f"mut={mut_channel.shape}, ranks={merged_ranks.shape}"
    )

    print("\n=== Expression prevalence filter ===")
    log_tpm = load_log_tpm_expression()
    partner_genes = sorted(
        {partner_gene(str(f)) for f in exp_channel.index}
    )
    allowed_genes = genes_passing_expression_filter(
        partner_genes,
        log_tpm,
        sclc_embedding.index,
    )
    exp_channel, mut_channel, merged_ranks = filter_features_by_partner_expression(
        exp_channel,
        mut_channel,
        merged_ranks,
        allowed_genes,
    )

    print("\n=== Mutation / Chronos matrices for SCLC mut QC ===")
    mutations = load_mutation_matrix()
    effects = load_crispr_effects()
    sclc_ids = pd.Index(sclc_embedding.index.astype(str))
    print(
        f"  SCLC lines for mut QC: {len(sclc_ids)}; "
        f"overlap mut={len(sclc_ids.intersection(mutations.index))}, "
        f"overlap crispr={len(sclc_ids.intersection(effects.index))}"
    )

    print("\n=== Lowered cutoffs + mut QC + top-10 ===")
    predictions, threshold_result = build_top10_predictions(
        exp_channel,
        mut_channel,
        merged_ranks,
        mutations,
        effects,
        sclc_ids,
    )
    exp_predictions, mut_predictions = split_predictions_by_type(predictions)

    predictions.to_csv(OUTPUT_CSV, index=False)
    exp_predictions.to_csv(OUTPUT_EXP_CSV, index=False)
    mut_predictions.to_csv(OUTPUT_MUT_CSV, index=False)
    threshold_result.to_csv(OUTPUT_THRESHOLD_CSV, index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        threshold_result.to_excel(writer, sheet_name="Thresholds", index=False)
        exp_predictions.to_excel(writer, sheet_name="ExpSLIPredictions", index=False)
        mut_predictions.to_excel(writer, sheet_name="MutSLIPredictions", index=False)

    if SAVE_FULL_SCORE_MATRIX:
        merged_ranks.to_pickle(FULL_SCORE_CACHE)

    n_kos = int(predictions["ko_gene"].nunique()) if not predictions.empty else 0
    print("\nDone.")
    print(
        f"  Cutoffs exp={SCORE_CUTOFF_EXP}, mut={SCORE_CUTOFF_MUT} "
        f"(ref elbows {REFERENCE_ELBOW_EXP:.6g} / {REFERENCE_ELBOW_MUT:.6g})"
    )
    print(f"  KOs with exported candidates: {n_kos}/{merged_ranks.shape[1]}")
    print(f"  Exported candidate rows: {len(predictions)}")
    print(f"  Exp rows: {len(exp_predictions)} | Mut rows: {len(mut_predictions)}")
    if not predictions.empty:
        print("  feature_type counts:")
        print(predictions["feature_type"].value_counts().to_string())
    print(f"  {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
