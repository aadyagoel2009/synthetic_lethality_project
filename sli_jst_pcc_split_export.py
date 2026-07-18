"""
Genome-wide SLI pair discovery using the type-split method (fixed hyperparameters).

Method: split_g10_lam0.05
  - Expression features: SCLC JST-PCC  (C_sclc * (1 + 0.05 * max(0, z(transfer_AGOP))))
  - Mutation features: pan SL-RFM     (AGOP_pan * C_pan, from precomputed pan FI)
  - Merge: within-type percentile ranks -> one ranking per KO

Outputs (sli_jst_pcc_split_results/):
  - top_SLI_pairs_split_g10_lam0.05.csv       one row per KO (top partner)
  - top10_candidates_per_ko.csv             top-10 features per KO (long)
  - panel_rank_split_g10_lam0.05.csv          14-panel benchmark check
  - fi_split_g10_lam0.05_full.pkl             full feature x KO score matrix
  - SCLC_SLIs_split_g10_lam0.05.xlsx          summary workbook

First run computes transfer structure for all KOs (slow); cached as structure_g10_all_kos.pkl.

Run: python sli_jst_pcc_split_export.py
"""

from __future__ import annotations

import os
import re
from typing import List

import numpy as np
import pandas as pd
import torch

import sli_jst_pcc_split_pipeline as split

# Fixed method chosen for publication / discovery
SPLIT_GAMMA = 10.0
SPLIT_LAM = 0.05
METHOD_TAG = f"split_g{SPLIT_GAMMA:g}_lam{SPLIT_LAM:g}"

RESULTS_DIR = split.RESULTS_DIR
TOP_K_PER_KO = 10
SAVE_FULL_MATRIX = True
USE_STRUCTURE_CACHE = True
STRUCTURE_CACHE = os.path.join(RESULTS_DIR, "structure_g10_all_kos.pkl")


def build_top_pair_per_ko_full(fi: pd.DataFrame) -> pd.DataFrame:
    """One row per KO with top-1/top-2 partners, gaps, feature types (SLI_identifier style)."""
    rows = []
    for ko in fi.columns:
        s = fi[ko].sort_values(ascending=False)
        top1_feat = str(s.index[0])
        top1 = float(s.iloc[0])
        top2_feat = str(s.index[1]) if len(s) > 1 else ""
        top2 = float(s.iloc[1]) if len(s) > 1 else np.nan
        gap1_2 = top1 - top2 if len(s) > 1 else np.nan
        other_mean = float(s.iloc[1:].mean()) if len(s) > 1 else np.nan
        gap1_other = top1 - other_mean if len(s) > 1 else np.nan
        rows.append(
            {
                "ko_gene": ko,
                "top1_feature": top1_feat,
                "top1_partner_gene": re.sub(r"_exp$", "", top1_feat),
                "top1_feature_type": "exp" if top1_feat.endswith("_exp") else "mut",
                "top1_score": top1,
                "top2_feature": top2_feat,
                "top2_partner_gene": re.sub(r"_exp$", "", top2_feat) if top2_feat else "",
                "top2_feature_type": (
                    "exp" if top2_feat.endswith("_exp") else ("mut" if top2_feat else "")
                ),
                "top2_score": top2,
                "gap_top1_top2": gap1_2,
                "mean_other_score": other_mean,
                "gap_top1_mean_other": gap1_other,
            }
        )
    out = pd.DataFrame(rows)
    out = out.sort_values(["top1_score", "gap_top1_top2"], ascending=[False, False])
    out = out.reset_index(drop=True)
    out.insert(0, "overall_rank", np.arange(1, len(out) + 1))
    return out


def build_top_k_long(fi: pd.DataFrame, k: int, method_tag: str) -> pd.DataFrame:
    rows: List[dict] = []
    for ko in fi.columns:
        s = fi[ko].sort_values(ascending=False).head(k)
        for rank, (feat, score) in enumerate(s.items(), start=1):
            feat_str = str(feat)
            rows.append(
                {
                    "ko_gene": ko,
                    "rank_in_ko": rank,
                    "feature": feat_str,
                    "partner_gene": re.sub(r"_exp$", "", feat_str),
                    "feature_type": "exp" if feat_str.endswith("_exp") else "mut",
                    "score": float(score),
                    "method": method_tag,
                }
            )
    return pd.DataFrame(rows)


def load_or_compute_mut_channel(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    feature_index: pd.Index,
    device,
) -> pd.DataFrame:
    prefix = split.FEATURE_IMPORTANCE_PAN_PREFIX
    if all(os.path.exists(prefix + s) for s in ("_data.npy", "_index.npy", "_columns.npy")):
        print(f"Loading pan FI (mut channel) from {prefix}")
        fi = split.load_feature_importance(prefix)
        return fi.reindex(index=feature_index, columns=pan_effects.columns).fillna(0)

    print("Precomputed pan FI not found; computing pan SL-RFM for all KOs...")
    fi = split.compute_pan_fi(pan_embedding, pan_effects, device)
    return fi.reindex(index=feature_index, columns=pan_effects.columns).fillna(0)


def load_or_compute_structure(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    p_pan_values: np.ndarray,
    omega: np.ndarray,
    device,
) -> pd.DataFrame:
    if USE_STRUCTURE_CACHE and os.path.exists(STRUCTURE_CACHE):
        print(f"Loading cached transfer structure from {STRUCTURE_CACHE}")
        cached = pd.read_pickle(STRUCTURE_CACHE)
        return cached.reindex(
            index=pan_embedding.columns,
            columns=pan_effects.columns,
        ).fillna(0)

    print(
        f"Computing transfer structure (gamma={SPLIT_GAMMA}) for "
        f"{pan_effects.shape[1]} KOs — this may take a long time..."
    )
    structure = split.compute_jst_structure(
        pan_embedding,
        pan_effects,
        device,
        p_pan_values,
        omega,
        SPLIT_GAMMA,
    )
    if USE_STRUCTURE_CACHE:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        structure.to_pickle(STRUCTURE_CACHE)
        print(f"Cached structure -> {STRUCTURE_CACHE}")
    return structure


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Genome-wide SLI export: {METHOD_TAG}")
    print(f"Results -> {RESULTS_DIR}/")

    pan_embedding = split.ensure_row_l2_normalized(
        split.load_hkl_dataframe(split.PAN_EMBED_HKL, "pan")
    )
    pan_effects = split.load_hkl_dataframe(split.PAN_GENE_EFFECT_HKL, "pan_effects")
    sclc_embedding = split.ensure_row_l2_normalized(
        split.load_hkl_dataframe(split.SCLC_EMBED_HKL, "sclc")
    )
    sclc_effects = split.load_hkl_dataframe(split.SCLC_GENE_EFFECT_HKL, "sclc_effects")
    pan_embedding, pan_effects, sclc_embedding, sclc_effects = split.align_embeddings_and_effects(
        pan_embedding, pan_effects, sclc_embedding, sclc_effects
    )

    split.validate_sclc_in_pan(pan_embedding, sclc_embedding)
    sclc_meta, lung_meta = split.load_depmap_lineage_ids()
    sclc_in_pan, lung_in_pan = split.intersect_lineage_with_pan(
        pan_embedding.index, sclc_meta, lung_meta
    )
    split.print_dataset_diagnostics(
        pan_embedding, pan_effects, sclc_in_pan, lung_in_pan, device
    )

    all_kos = list(pan_effects.columns)
    print(f"All KO genes: {len(all_kos)}")

    p_pan = split.load_or_compute_p_pan(pan_embedding, pan_effects, device)
    p_pan_values = p_pan.values

    print("\n=== KMM weights ===")
    omega = split.compute_kmm_weights(pan_embedding, sclc_in_pan, lung_in_pan)

    print("\n=== SCLC PCC (exp channel base) ===")
    pcc_sclc = split.process_pcc_for_sli(split.get_pcc(sclc_embedding, sclc_effects))
    pcc_sclc = pcc_sclc.reindex(columns=all_kos).fillna(0)

    print("\n=== Mutation channel (pan SL-RFM, all KOs) ===")
    mut_channel = load_or_compute_mut_channel(
        pan_embedding, pan_effects, pcc_sclc.index, device
    )

    print("\n=== Transfer structure (exp channel boost) ===")
    structure = load_or_compute_structure(
        pan_embedding, pan_effects, p_pan_values, omega, device
    )

    print(f"\n=== Merge type-split scores ({METHOD_TAG}) ===")
    exp_channel = split.compute_exp_jst_pcc_scores(pcc_sclc, structure, SPLIT_LAM)
    merged = split.merge_type_split_scores(exp_channel, mut_channel)

    print(f"Score matrix: {merged.shape[0]} features x {merged.shape[1]} KOs")

    # --- Panel benchmark sanity check ---
    panel_eval = split.evaluate_panel(merged, METHOD_TAG)
    panel_csv = os.path.join(RESULTS_DIR, f"panel_rank_{METHOD_TAG}.csv")
    panel_eval.to_csv(panel_csv, index=False)
    mean_rank = split.mean_panel_rank(panel_eval, METHOD_TAG)
    print(f"\n14-panel mean rank ({METHOD_TAG}): {mean_rank:.1f}")
    print(panel_eval[["panel", "expected_partner_rank", "matched_feature"]].to_string(index=False))

    # --- Top pairs per KO ---
    top_pairs = build_top_pair_per_ko_full(merged)
    top_csv = os.path.join(RESULTS_DIR, f"top_SLI_pairs_{METHOD_TAG}.csv")
    top_pairs.to_csv(top_csv, index=False)
    print(f"\nTop SLI pair per KO: {len(top_pairs)} rows -> {top_csv}")

    mut_top = (top_pairs["top1_feature_type"] == "mut").sum()
    exp_top = (top_pairs["top1_feature_type"] == "exp").sum()
    print(f"  Top-1 feature type: {mut_top} mut, {exp_top} exp")

    top_k = build_top_k_long(merged, TOP_K_PER_KO, METHOD_TAG)
    top_k_csv = os.path.join(RESULTS_DIR, f"top{TOP_K_PER_KO}_candidates_per_ko_{METHOD_TAG}.csv")
    top_k.to_csv(top_k_csv, index=False)
    print(f"Top-{TOP_K_PER_KO} per KO -> {top_k_csv}")

    if SAVE_FULL_MATRIX:
        pkl_path = os.path.join(RESULTS_DIR, f"fi_{METHOD_TAG}_full.pkl")
        merged.to_pickle(pkl_path)
        print(f"Full matrix -> {pkl_path}")

    xlsx = os.path.join(RESULTS_DIR, f"SCLC_SLIs_{METHOD_TAG}.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        top_pairs.to_excel(writer, sheet_name="TopPairPerKO", index=False)
        panel_eval.to_excel(writer, sheet_name="PanelBenchmark", index=False)
        top_pairs.head(500).to_excel(writer, sheet_name="Top500_KO_hits", index=False)

    print("\nDone.")
    print(f"  {top_csv}")
    print(f"  {xlsx}")


if __name__ == "__main__":
    main()
