"""
Genome-wide top-10 SLI predictions using split_g10_lam0.05.

Paper scoring (generate_figures.ipynb), applied within each feature type:
  score_type(k) = max(v_type_k) - mean(v_type_k)

where v_exp is the SCLC JST-PCC channel and v_mut is the pan SL-RFM channel.

Why not one combined raw matrix?
  Exp and mut raw scores live on very different scales, so a single max(v)-mean(v)
  on concatenated raw values lets expression dominate and mutation pairs disappear.
  The paper had one unified FI matrix; the split method keeps two channels, so the
  paper formula is applied within each channel.

Why percentiles are still used (for ranking only):
  Same as sli_jst_pcc_split_pipeline.py: within-type percentiles order candidates
  so exp and mut can compete. Percentiles are NOT used inside max-mean.

Pipeline:
  1. Rank features within each KO by split percentile merge.
  2. Elbow_exp from top expression scores; elbow_mut from top mutation scores.
  3. Export up to 10 distinct partners per KO, in percentile order, keeping
     candidates whose own within-type score (v[f]-mean_type) passes that
     type's elbow. Rank-3 uses the 3rd candidate's score, not top-1.
  4. Write exp and mut predictions on separate sheets so each type is sorted
     by its own ko_score scale.

Run:
  python sli_jst_pcc_split_top10_predictions_typed_elbow.py
"""

from __future__ import annotations

import os
import re
from typing import List

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
VARIANT = "typed_elbow"

TOP_K_GENES = 10

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


def load_pan_fi_for_all_kos(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    feature_index: pd.Index,
    device: torch.device,
) -> pd.DataFrame:
    """Load the paper-style pan SL-RFM matrix used for mutation features."""
    prefix = split.FEATURE_IMPORTANCE_PAN_PREFIX
    suffixes = ("_data.npy", "_index.npy", "_columns.npy")
    if all(os.path.exists(prefix + suffix) for suffix in suffixes):
        print(f"Loading pan feature importance from {prefix}")
        fi_pan = split.load_feature_importance(prefix)
    else:
        print("Precomputed pan feature importance not found; computing it...")
        fi_pan = split.compute_pan_fi(pan_embedding, pan_effects, device)

    return fi_pan.reindex(
        index=feature_index,
        columns=pan_effects.columns,
    ).fillna(0.0)


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


def build_top10_predictions(
    exp_channel: pd.DataFrame,
    mut_channel: pd.DataFrame,
    merged_ranks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank by split percentile merge; score/cutoff with paper max-mean
    within each feature type.
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

    top_exp_scores = paper_top_scores(exp_ch.loc[exp_idx]) if exp_idx else pd.Series(dtype=float)
    top_mut_scores = paper_top_scores(mut_ch.loc[mut_idx]) if mut_idx else pd.Series(dtype=float)

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

            # Percentile order is not monotone in raw score, so do not break.
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
                    "method": METHOD,
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
        .sort_values(["ko_score", "ko_gene", "rank_in_ko"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    mut_preds = (
        predictions.loc[predictions["feature_type"] == "mut"]
        .sort_values(["ko_score", "ko_gene", "rank_in_ko"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return exp_preds, mut_preds


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Method: {METHOD}")

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

    print("\n=== SCLC expression context ===")
    pcc_sclc = split.process_pcc_for_sli(
        split.get_pcc(sclc_embedding, sclc_effects)
    ).reindex(
        index=pan_embedding.columns,
        columns=pan_effects.columns,
    ).fillna(0.0)

    print("\n=== Pan mutation channel ===")
    mut_channel = load_pan_fi_for_all_kos(
        pan_embedding,
        pan_effects,
        pcc_sclc.index,
        device,
    )

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

    print("\n=== split_g10_lam0.05 channels + percentile ranking ===")
    exp_channel = split.compute_exp_jst_pcc_scores(
        pcc_sclc,
        structure,
        LAMBDA,
    )
    merged_ranks = split.merge_type_split_scores(exp_channel, mut_channel)
    print(
        f"Channels: exp={exp_channel.shape}, mut={mut_channel.shape}, "
        f"ranks={merged_ranks.shape}"
    )

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
        if not exp_predictions.empty:
            print(
                f"  exp ko_score range: "
                f"{exp_predictions['ko_score'].min():.6g} .. "
                f"{exp_predictions['ko_score'].max():.6g}"
            )
        if not mut_predictions.empty:
            print(
                f"  mut ko_score range: "
                f"{mut_predictions['ko_score'].min():.6g} .. "
                f"{mut_predictions['ko_score'].max():.6g}"
            )
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_EXP_CSV}")
    print(f"  {OUTPUT_MUT_CSV}")
    print(f"  {OUTPUT_ELBOW_CSV}")
    print(f"  {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
