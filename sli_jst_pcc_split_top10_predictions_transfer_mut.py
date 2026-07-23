"""
Genome-wide top-10 SLI predictions (transfer-mut + expression filter).

Method variant of split_g10_lam0.05:

  Expression (unchanged split ranking channel):
    S_exp = C_sclc * (1 + lambda * max(0, zscore(transfer_AGOP)))

  Mutation (SCLC-aligned structure x pan correlation):
    S_mut = transfer_AGOP * C_pan

  Merge for ranking only:
    within-type percentile ranks (same as sli_jst_pcc_split_pipeline.py)

Before elbow / cutoff scoring:
  Drop any partner gene whose log1p(TPM) is < 1 in too many lines.
  Keep a partner only if log1p(TPM) >= 1 in at least 25% of SCLC cell lines.
  (Checked via the gene's *_exp column in DepMap expression.)

Then:
  1. Paper score within each type: max(v_type) - mean(v_type)
  2. Elbow_exp / elbow_mut on the filtered features
  3. Up to 10 distinct partners per KO passing the type elbow
  4. Exp and mut predictions written on separate sheets

Run:
  python sli_jst_pcc_split_top10_predictions_transfer_mut.py
"""

from __future__ import annotations

import os
import re
from typing import List, Set

import numpy as np
import pandas as pd
import torch

import sli_jst_pcc_split_pipeline as split


# ---------------------------------------------------------------------------
# Fixed method
# ---------------------------------------------------------------------------
GAMMA = 10.0
LAMBDA = 0.05
METHOD = "split_g10_lam0.05"
VARIANT = "transfer_mut_expfilter"

TOP_K_GENES = 10

MIN_LOG_TPM = 1.0
MIN_EXPRESSED_FRAC = 0.25
EXPRESSION_CSV = "datasets/2023/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"

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
OUTPUT_ELBOW_CSV = os.path.join(
    RESULTS_DIR, f"elbow_score_{METHOD}_{VARIANT}.csv"
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


def calculate_elbow_score(top_ko_scores: pd.Series) -> float:
    """
    Find the elbow by maximum distance from the sorted score curve to the
    straight line joining its endpoints.
    """
    values = (
        top_ko_scores.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_values()
        .to_numpy()
    )
    if values.size == 0:
        raise ValueError("Cannot calculate an elbow from zero valid KO scores.")
    if values.size < 3 or np.allclose(values, values[0]):
        return float(values[0])

    x = np.arange(values.size, dtype=float)
    start = np.array([x[0], values[0]], dtype=float)
    end = np.array([x[-1], values[-1]], dtype=float)
    line_vector = end - start
    line_length = float(np.linalg.norm(line_vector))
    if line_length == 0:
        return float(values[0])

    points = np.column_stack((x, values))
    line_unit = line_vector / line_length
    projection_lengths = (points - start) @ line_unit
    projected_points = start + np.outer(projection_lengths, line_unit)
    distances = np.linalg.norm(points - projected_points, axis=1)
    elbow_index = int(np.argmax(distances))
    return float(values[elbow_index])


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


def build_top10_predictions(
    exp_channel: pd.DataFrame,
    mut_channel: pd.DataFrame,
    merged_ranks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank by split percentile merge; score/cutoff with paper max-mean
    within each feature type. Channels must already be expression-filtered.
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

    elbow_exp = (
        calculate_elbow_score(top_exp_scores) if len(top_exp_scores) else float("nan")
    )
    elbow_mut = (
        calculate_elbow_score(top_mut_scores) if len(top_mut_scores) else float("nan")
    )

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
    print(f"  Elbow exp: {elbow_exp:.12g}")
    print(f"  Elbow mut: {elbow_mut:.12g}")

    prediction_rows: List[dict] = []

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
                elbow = elbow_exp
                type_mean = exp_mean
            else:
                raw_value = float(mut_ch.at[feature, ko_gene])
                candidate_score = raw_value - mut_mean
                elbow = elbow_mut
                type_mean = mut_mean

            if not np.isfinite(elbow) or candidate_score < elbow:
                continue

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
                    "method": f"{METHOD}_{VARIANT}",
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    if not predictions.empty:
        predictions = predictions.sort_values(
            ["feature_type", "ko_score", "ko_gene", "rank_in_ko"],
            ascending=[True, False, True, True],
        ).reset_index(drop=True)

    elbow_result = pd.DataFrame(
        {
            "elbow_score_exp": [elbow_exp],
            "elbow_score_mut": [elbow_mut],
            "min_log_tpm": [MIN_LOG_TPM],
            "min_expressed_frac": [MIN_EXPRESSED_FRAC],
            "expression_filter_lines": ["sclc"],
        }
    )
    return predictions, elbow_result


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

    print("\n=== Expression prevalence filter (before elbow) ===")
    log_tpm = load_log_tpm_expression()
    partner_genes = sorted(
        {partner_gene(str(f)) for f in exp_channel.index}
    )
    # Use SCLC lines for the sparse-expression noise problem discussed.
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

    print("\n=== Elbow + top-10 on filtered features ===")
    predictions, elbow_result = build_top10_predictions(
        exp_channel,
        mut_channel,
        merged_ranks,
    )
    exp_predictions, mut_predictions = split_predictions_by_type(predictions)

    predictions.to_csv(OUTPUT_CSV, index=False)
    exp_predictions.to_csv(OUTPUT_EXP_CSV, index=False)
    mut_predictions.to_csv(OUTPUT_MUT_CSV, index=False)
    elbow_result.to_csv(OUTPUT_ELBOW_CSV, index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        elbow_result.to_excel(writer, sheet_name="ElbowScore", index=False)
        exp_predictions.to_excel(writer, sheet_name="ExpSLIPredictions", index=False)
        mut_predictions.to_excel(writer, sheet_name="MutSLIPredictions", index=False)

    if SAVE_FULL_SCORE_MATRIX:
        merged_ranks.to_pickle(FULL_SCORE_CACHE)

    n_kos = int(predictions["ko_gene"].nunique()) if not predictions.empty else 0
    print("\nDone.")
    print(
        f"  Elbow exp={elbow_result.at[0, 'elbow_score_exp']:.12g}, "
        f"mut={elbow_result.at[0, 'elbow_score_mut']:.12g}"
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
