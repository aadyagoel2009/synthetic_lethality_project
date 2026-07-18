"""
Genome-wide top-10 SLI predictions using split_g10_lam0.05.

For every CRISPR knockout gene:
  1. Score expression features with SCLC JST-PCC (gamma=10, lambda=0.05).
  2. Score mutation features with the pan-cancer SL-RFM feature importance.
  3. Merge expression and mutation rankings using within-type percentiles.
  4. Convert features to partner genes while retaining exp/mut evidence.
  5. Export up to 10 distinct partner genes that pass the paper cutoff.

Original-paper cutoff:
  score(k) = max(v_k) - mean(v_k)
  cutoff   = 0.007101906647612656

The paper applied this cutoff to the top feature only. To produce up to 10
predictions per knockout, this script applies the same deviation criterion to
each candidate:
  candidate_score(f, k) = v[f, k] - mean(v_k)
and retains candidate_score >= cutoff.

Important: v is the final split-model percentile score, not the raw SL-RFM
importance used in the paper. The numerical cutoff is reproduced exactly, but
its calibration is not directly comparable because the score scale changed.

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
# Fixed method and paper threshold
# ---------------------------------------------------------------------------
GAMMA = 10.0
LAMBDA = 0.05
METHOD = "split_g10_lam0.05"

TOP_K_GENES = 10
PAPER_SCORE_CUTOFF = 0.007101906647612656

RESULTS_DIR = split.RESULTS_DIR
STRUCTURE_CACHE = os.path.join(RESULTS_DIR, "structure_g10_all_kos.pkl")
FULL_SCORE_CACHE = os.path.join(RESULTS_DIR, f"fi_{METHOD}_full.pkl")

OUTPUT_CSV = os.path.join(RESULTS_DIR, f"top10_SLI_predictions_{METHOD}.csv")
OUTPUT_KO_SUMMARY_CSV = os.path.join(
    RESULTS_DIR, f"KO_threshold_summary_{METHOD}.csv"
)
OUTPUT_XLSX = os.path.join(RESULTS_DIR, f"top10_SLI_predictions_{METHOD}.xlsx")

SAVE_FULL_SCORE_MATRIX = True


def partner_gene(feature: str) -> str:
    """Map GENE_exp and GENE mutation features to the gene symbol."""
    return re.sub(r"_exp$", "", str(feature))


def feature_type(feature: str) -> str:
    return "exp" if str(feature).endswith("_exp") else "mut"


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
    merged_scores: pd.DataFrame,
    exp_channel: pd.DataFrame,
    mut_channel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return up to 10 distinct partner genes per KO plus a KO-level summary.

    Multiple feature types for one partner gene are not emitted twice. The
    higher-ranked feature is retained and its exp/mut type remains explicit.
    """
    prediction_rows: List[dict] = []
    summary_rows: List[dict] = []

    for ko_gene in merged_scores.columns:
        scores = (
            merged_scores[ko_gene]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .sort_values(ascending=False)
        )
        if scores.empty:
            continue

        mean_score = float(scores.mean())
        top_deviation = float(scores.iloc[0] - mean_score)
        ko_passes = top_deviation >= PAPER_SCORE_CUTOFF

        kept = 0
        seen_genes: set[str] = set()
        candidates_passing = 0

        for feature, score in scores.items():
            feature_str = str(feature)
            gene = partner_gene(feature_str)
            if gene in seen_genes:
                continue
            seen_genes.add(gene)

            candidate_deviation = float(score - mean_score)
            passes = candidate_deviation >= PAPER_SCORE_CUTOFF
            if not passes:
                # Scores are descending, so later candidates cannot pass.
                break

            candidates_passing += 1
            if kept >= TOP_K_GENES:
                continue

            ftype = feature_type(feature_str)
            channel = exp_channel if ftype == "exp" else mut_channel
            raw_channel_score = float(channel.at[feature, ko_gene])

            kept += 1
            prediction_rows.append(
                {
                    "ko_gene": str(ko_gene),
                    "rank_in_ko": kept,
                    "partner_gene": gene,
                    "feature": feature_str,
                    "feature_type": ftype,
                    "merged_percentile_score": float(score),
                    "mean_ko_score": mean_score,
                    "candidate_deviation_from_mean": candidate_deviation,
                    "paper_score_cutoff": PAPER_SCORE_CUTOFF,
                    "passes_paper_cutoff": True,
                    "raw_channel_score": raw_channel_score,
                    "method": METHOD,
                }
            )

        summary_rows.append(
            {
                "ko_gene": str(ko_gene),
                "top_feature": str(scores.index[0]),
                "top_partner_gene": partner_gene(str(scores.index[0])),
                "top_feature_type": feature_type(str(scores.index[0])),
                "top_merged_score": float(scores.iloc[0]),
                "mean_ko_score": mean_score,
                "paper_ko_score_top_minus_mean": top_deviation,
                "paper_score_cutoff": PAPER_SCORE_CUTOFF,
                "passes_paper_cutoff": ko_passes,
                "distinct_candidates_passing_cutoff": candidates_passing,
                "candidates_exported": kept,
                "method": METHOD,
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    summary = pd.DataFrame(summary_rows)

    if not predictions.empty:
        predictions = predictions.sort_values(
            ["ko_gene", "rank_in_ko"]
        ).reset_index(drop=True)
    if not summary.empty:
        summary = summary.sort_values(
            "paper_ko_score_top_minus_mean", ascending=False
        ).reset_index(drop=True)

    return predictions, summary


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Method: {METHOD}")
    print(f"Paper cutoff: {PAPER_SCORE_CUTOFF}")

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

    print("\n=== split_g10_lam0.05 score ===")
    exp_channel = split.compute_exp_jst_pcc_scores(
        pcc_sclc,
        structure,
        LAMBDA,
    )
    merged_scores = split.merge_type_split_scores(
        exp_channel,
        mut_channel,
    )
    print(
        f"Merged score matrix: {merged_scores.shape[0]} features x "
        f"{merged_scores.shape[1]} KOs"
    )

    predictions, ko_summary = build_top10_predictions(
        merged_scores,
        exp_channel,
        mut_channel,
    )

    predictions.to_csv(OUTPUT_CSV, index=False)
    ko_summary.to_csv(OUTPUT_KO_SUMMARY_CSV, index=False)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        predictions.to_excel(writer, sheet_name="Top10SLIPredictions", index=False)
        ko_summary.to_excel(writer, sheet_name="KOThresholdSummary", index=False)

    if SAVE_FULL_SCORE_MATRIX:
        merged_scores.to_pickle(FULL_SCORE_CACHE)

    n_kos = int(ko_summary["passes_paper_cutoff"].sum())
    print("\nDone.")
    print(f"  KOs passing cutoff: {n_kos}/{len(ko_summary)}")
    print(f"  Exported candidate rows: {len(predictions)}")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_KO_SUMMARY_CSV}")
    print(f"  {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
