"""
Genome-wide top-10 SLI predictions using split_g10_lam0.05.

Follows the original SL-RFM paper scoring steps from generate_figures.ipynb:
  score(k) = max(v_k) - mean(v_k)

where v is the feature-importance matrix (NOT percentile ranks).

For the split method, v is built from the raw channel scores:
  - expression features: SCLC JST-PCC (gamma=10, lambda=0.05)
  - mutation features:   pan SL-RFM feature importance

Then:
  1. For the elbow only, compute one score per KO from its top feature:
       top_ko_score(k) = max(v_k) - mean(v_k)
  2. Calculate an elbow cutoff from those top-KO scores.
  3. Export up to 10 distinct partner genes per KO. Each candidate keeps
     its own score:
       candidate_score(f, k) = v[f, k] - mean(v_k)
     so the 3rd-best pair uses the 3rd-best feature's value minus the mean,
     not the top-1 value. Keep candidates with candidate_score >= elbow.

Run:
  python sli_jst_pcc_split_top10_predictions.py
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

TOP_K_GENES = 10

RESULTS_DIR = split.RESULTS_DIR
STRUCTURE_CACHE = os.path.join(RESULTS_DIR, "structure_g10_all_kos.pkl")
FULL_SCORE_CACHE = os.path.join(RESULTS_DIR, f"fi_{METHOD}_full.pkl")

OUTPUT_CSV = os.path.join(RESULTS_DIR, f"top10_SLI_predictions_{METHOD}.csv")
OUTPUT_ELBOW_CSV = os.path.join(RESULTS_DIR, f"elbow_score_{METHOD}.csv")
OUTPUT_XLSX = os.path.join(RESULTS_DIR, f"top10_SLI_predictions_{METHOD}.xlsx")

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


def build_importance_matrix(
    exp_channel: pd.DataFrame,
    mut_channel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the paper-style feature-importance matrix v for the split method.

    Expression features keep their JST-PCC scores; mutation features keep their
    pan SL-RFM scores. This is the matrix that max(v)-mean(v) should be applied
    to -- not the within-type percentile merge used only for panel ranking.
    """
    common_idx = exp_channel.index.intersection(mut_channel.index)
    common_cols = exp_channel.columns.intersection(mut_channel.columns)
    exp_idx, mut_idx = split.split_feature_index(common_idx)

    v = pd.DataFrame(0.0, index=common_idx, columns=common_cols, dtype=float)
    if exp_idx:
        v.loc[exp_idx, common_cols] = (
            exp_channel.loc[exp_idx, common_cols].astype(float).values
        )
    if mut_idx:
        v.loc[mut_idx, common_cols] = (
            mut_channel.loc[mut_idx, common_cols].astype(float).values
        )
    return v.fillna(0.0)


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
    importance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Elbow uses only the top feature per KO (paper):
      top_ko_score(k) = max(v_k) - mean(v_k)

    Each exported candidate uses that candidate's own importance:
      ko_score(f, k) = v[f, k] - mean(v_k)
    e.g. rank-3 row uses the 3rd-best feature minus the KO mean.
    """
    # Paper (generate_figures.ipynb):
    #   top_deviations = feature_importances.max() - feature_importances.mean()
    # Apply this to raw importances only -- never to percentile ranks.
    # Used solely to set the elbow cutoff.
    top_ko_scores = importance.max(axis=0) - importance.mean(axis=0)
    top_ko_scores = top_ko_scores.replace([np.inf, -np.inf], np.nan).dropna()
    if top_ko_scores.nunique(dropna=True) <= 1:
        raise ValueError(
            "All KO scores are identical after max(v)-mean(v). This usually "
            "means v was percentile-ranked instead of raw feature importance."
        )
    elbow_score = calculate_elbow_score(top_ko_scores)

    prediction_rows: List[dict] = []

    for ko_gene in top_ko_scores.index:
        scores = (
            importance[ko_gene]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values(ascending=False)
        )
        if scores.empty:
            continue

        mean_score = float(scores.mean())
        kept = 0
        seen_genes: set[str] = set()

        for feature, value in scores.items():
            feature_str = str(feature)
            gene = partner_gene(feature_str)
            if gene in seen_genes:
                continue
            seen_genes.add(gene)

            # This candidate's own score: v[this feature] - mean(v_k),
            # not max(v_k) - mean(v_k).
            candidate_score = float(value - mean_score)
            if candidate_score < elbow_score:
                # Sorted by v descending, so later candidates cannot pass.
                break

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
                    "feature_type": feature_type(feature_str),
                    "feature_importance": float(value),
                    "mean_ko_importance": mean_score,
                    "method": METHOD,
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    if not predictions.empty:
        predictions = predictions.sort_values(
            ["ko_score", "ko_gene", "rank_in_ko"],
            ascending=[False, True, True],
        ).reset_index(drop=True)

    elbow_result = pd.DataFrame({"elbow_score": [elbow_score]})
    return predictions, elbow_result


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

    print("\n=== split_g10_lam0.05 feature importances ===")
    exp_channel = split.compute_exp_jst_pcc_scores(
        pcc_sclc,
        structure,
        LAMBDA,
    )
    importance = build_importance_matrix(exp_channel, mut_channel)
    print(
        f"Importance matrix v: {importance.shape[0]} features x "
        f"{importance.shape[1]} KOs"
    )

    predictions, elbow_result = build_top10_predictions(importance)

    predictions.to_csv(OUTPUT_CSV, index=False)
    elbow_result.to_csv(OUTPUT_ELBOW_CSV, index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        elbow_result.to_excel(writer, sheet_name="ElbowScore", index=False)
        predictions.to_excel(writer, sheet_name="SLIPredictions", index=False)

    if SAVE_FULL_SCORE_MATRIX:
        importance.to_pickle(FULL_SCORE_CACHE)

    elbow_score = float(elbow_result.at[0, "elbow_score"])
    n_kos = int(predictions["ko_gene"].nunique()) if not predictions.empty else 0
    print("\nDone.")
    print(f"  Calculated elbow score: {elbow_score:.12g}")
    print(f"  KOs with exported candidates: {n_kos}/{importance.shape[1]}")
    print(f"  Exported candidate rows: {len(predictions)}")
    if not predictions.empty:
        print(
            f"  ko_score range: "
            f"{predictions['ko_score'].min():.6g} .. "
            f"{predictions['ko_score'].max():.6g}"
        )
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_ELBOW_CSV}")
    print(f"  {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
