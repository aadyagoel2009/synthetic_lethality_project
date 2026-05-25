import os, re
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
FEATURE_IMPORTANCE_PAN_PREFIX  = "datasets/feature_importances_pan"
FEATURE_IMPORTANCE_LUNG_PREFIX = "datasets/feature_importances_lung"
FEATURE_IMPORTANCE_SCLC_PREFIX = "datasets/feature_importances_sclc"

SRC_NAME = "pan"    # "pan", "lung", "sclc"
TGT_NAME = "lung"   # "pan", "lung", "sclc"

RESULTS_DIR = "pancancer_lung_results"  # NEW folder name
BETA_STEP = 0.05

# Format: (sheet_title, ko_gene, expected_partner_gene)
# expected_partner_gene is only used for the summary table printed at the end.
KNOWN_SLI_PANELS = [
    ("ASCL1-ASCL1", "ASCL1", "ASCL1"),
    ("POU2F3-POU2F3", "POU2F3", "POU2F3"),
    ("POU2F3-POU2AF2", "POU2AF2", "POU2F3"),
    ("NEUROD1-NEUROD1", "NEUROD1", "NEUROD1"),
    ("NEUROD1-MYC", "MYC", "NEUROD1"),
    ("EP300-CREBBP", "CREBBP", "EP300"),
    ("CREBBP-EP300", "EP300", "CREBBP"),
    ("POU2F3-IGF1R", "IGF1R", "POU2F3"),
    ("POU2F3-SOX9", "SOX9", "POU2F3"),
    ("POU2F3-ASCL2", "ASCL2", "POU2F3"),
    ("NEUROD1-CDK7", "CDK7", "NEUROD1"),
    ("NEUROD1-PLK1", "PLK1", "NEUROD1"),
    ("ASCL1-PPP2CA", "PPP2CA", "ASCL1"),
    ("ASCL1-DDX3X", "DDX3X", "ASCL1"),
]

EXPORT_ALL_CANDIDATES_SHEET = False  # leave False unless you really want huge files


# =========================
# HELPERS
# =========================
def load_feature_importance(prefix):
    data = np.load(prefix + "_data.npy", allow_pickle=True)
    idx  = np.load(prefix + "_index.npy", allow_pickle=True)
    cols = np.load(prefix + "_columns.npy", allow_pickle=True)
    return pd.DataFrame(data, index=idx, columns=cols)

def safe_sheet_name(name):
    name = re.sub(r'[:\\/?*\[\]]', '_', name)
    return name[:31]

def build_top_pair_per_ko(feature_importances):
    """
    One row per KO gene (A): its top feature (B), score, and score gaps.
    Ranked by top1 score, then by top1-top2 gap.

    gap_top1_mean_other = top1 - mean(all other feature scores)
    """
    rows = []
    for ko in feature_importances.columns:
        s = feature_importances[ko].sort_values(ascending=False)

        top1_feat = str(s.index[0])
        top1 = float(s.iloc[0])

        top2_feat = str(s.index[1]) if len(s) > 1 else ""
        top2 = float(s.iloc[1]) if len(s) > 1 else np.nan
        gap1_2 = top1 - top2 if len(s) > 1 else np.nan

        other_mean = float(s.iloc[1:].mean()) if len(s) > 1 else np.nan
        gap1_other_mean = top1 - other_mean if len(s) > 1 else np.nan

        rows.append({
            "ko_gene": ko,
            "top1_feature": top1_feat,
            "top1_partner_gene": re.sub(r"_exp$", "", top1_feat),
            "top1_feature_type": "exp" if top1_feat.endswith("_exp") else "mut_or_other",
            "top1_score": top1,
            "top2_feature": top2_feat,
            "top2_partner_gene": re.sub(r"_exp$", "", top2_feat) if top2_feat else "",
            "top2_score": top2,
            "gap_top1_top2": gap1_2,
            "mean_other_score": other_mean,
            "gap_top1_mean_other": gap1_other_mean,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["top1_score", "gap_top1_top2"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "overall_rank", np.arange(1, len(out) + 1))
    return out

def rank_all_features_for_one_ko(feature_importance, ko_gene):
    """
    All features ranked for one KO gene.
    """
    importance_values = feature_importance[ko_gene].sort_values(ascending=False)
    df = importance_values.reset_index()
    df.columns = ["feature", "score"]
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["feature_type"] = np.where(df["feature"].astype(str).str.endswith("_exp"), "exp", "mut_or_other")
    df["partner_gene"] = df["feature"].astype(str).str.replace(r"_exp$", "", regex=True)
    return df

def build_all_candidates_long(feature_importance, model_tag):
    """
    Every (feature, KO) row as a long table (can be enormous).
    """
    long_df = (
        feature_importance.stack(dropna=False)
          .rename("score")
          .reset_index()
          .rename(columns={"level_0": "feature", "level_1": "ko_gene"})
    )
    long_df["feature_type"] = np.where(long_df["feature"].astype(str).str.endswith("_exp"), "exp", "mut_or_other")
    long_df["partner_gene"] = long_df["feature"].astype(str).str.replace(r"_exp$", "", regex=True)
    long_df["rank_in_ko"] = long_df.groupby("ko_gene")["score"].rank(ascending=False, method="min").astype(int)
    long_df["model_tag"] = model_tag
    return long_df.sort_values(["ko_gene", "rank_in_ko", "feature"]).reset_index(drop=True)

def get_partner_rank_and_score(feature_importance, ko_gene, expected_partner_gene):
    """
    For the KO sheet, find where expected_partner_gene appears in the ranked list.
    We match using partner_gene parsed from the feature name.
    Returns: (rank, score, matched_feature)
    """
    if ko_gene not in feature_importance.columns:
        return np.nan, np.nan, ""

    df = rank_all_features_for_one_ko(feature_importance, ko_gene)
    hits = df[df["partner_gene"] == expected_partner_gene]
    if len(hits) == 0:
        return np.nan, np.nan, ""

    best = hits.sort_values("rank", ascending=True).iloc[0]
    return int(best["rank"]), float(best["score"]), str(best["feature"])


# =========================
# MAIN
# =========================
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load the three control matrices
    feature_importance_pan  = load_feature_importance(FEATURE_IMPORTANCE_PAN_PREFIX)
    print("loaded pan features")
    feature_importance_lung = load_feature_importance(FEATURE_IMPORTANCE_LUNG_PREFIX)
    print("loaded lung features")
    feature_importance_sclc = load_feature_importance(FEATURE_IMPORTANCE_SCLC_PREFIX)
    print("loaded sclc features")

    by_name = {"pan": feature_importance_pan, "lung": feature_importance_lung, "sclc": feature_importance_sclc}

    # Prepare betas 0..1 step 0.05 (inclusive)
    betas = [round(x, 2) for x in np.arange(0, 1.0001, BETA_STEP)]
    results_rows = []

    for beta in betas:
        print("running beta")
        src = by_name[SRC_NAME]
        tgt = by_name[TGT_NAME]

        # Align
        common_idx = src.index.intersection(tgt.index)
        common_cols = src.columns.intersection(tgt.columns)
        src2 = src.loc[common_idx, common_cols]
        tgt2 = tgt.loc[common_idx, common_cols]

        feature_importance = (1.0 - beta) * src2 + beta * tgt2
        model_tag = f"{SRC_NAME}_to_{TGT_NAME}_beta{beta}"

        # Sheet 1
        sheet1 = build_top_pair_per_ko(feature_importance)

        # Export Excel for this beta
        out_xlsx = os.path.join(RESULTS_DIR, f"beta_{beta}.xlsx")
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            sheet1.to_excel(writer, sheet_name="TopPairPerKO", index=False)

            if EXPORT_ALL_CANDIDATES_SHEET:
                all_long = build_all_candidates_long(feature_importance, model_tag=model_tag)
                all_long.to_excel(writer, sheet_name="AllCandidates", index=False)

            # Sheets 2..: full ranked list for each known KO gene
            for title, ko_gene, expected_partner_gene in KNOWN_SLI_PANELS:
                sheet = safe_sheet_name(title)

                if ko_gene not in feature_importance.columns:
                    pd.DataFrame([{
                        "error": f"KO gene '{ko_gene}' not found in this matrix.",
                        "available_example_kos": ", ".join(map(str, feature_importance.columns[:25]))
                    }]).to_excel(writer, sheet_name=sheet, index=False)
                    continue

                df = rank_all_features_for_one_ko(feature_importance, ko_gene)
                df = df[["rank", "feature", "partner_gene", "feature_type", "score"]]
                df.to_excel(writer, sheet_name=sheet, index=False)

        print(f"Saved: {out_xlsx}")

        # Collect rank/score for summary table (where expected partner appears)
        for title, ko_gene, expected_partner_gene in KNOWN_SLI_PANELS:
            rnk, scr, matched_feat = get_partner_rank_and_score(feature_importance, ko_gene, expected_partner_gene)
            results_rows.append({
                "beta": beta,
                "panel": title,
                "ko_gene": ko_gene,
                "expected_partner_gene": expected_partner_gene,
                "expected_partner_rank": rnk,
                "expected_partner_score": scr,
                "matched_feature": matched_feat
            })

    # Build summary tables
    summary = pd.DataFrame(results_rows)

    # Pivot: panels × betas with ranks
    rank_pivot = summary.pivot_table(
        index=["panel", "ko_gene", "expected_partner_gene"],
        columns="beta",
        values="expected_partner_rank",
        aggfunc="min"
    )

    print("\n=== Expected partner RANK (lower is better) by beta ===")
    print(rank_pivot)

    # Best beta per panel (lowest rank, tie-breaker higher score)
    best_rows = []
    for (panel, ko_gene, expected_partner_gene), grp in summary.groupby(["panel", "ko_gene", "expected_partner_gene"]):
        grp2 = grp.dropna(subset=["expected_partner_rank"]).copy()
        if len(grp2) == 0:
            best_rows.append({
                "panel": panel,
                "ko_gene": ko_gene,
                "expected_partner_gene": expected_partner_gene,
                "best_beta": np.nan,
                "best_rank": np.nan,
                "best_score": np.nan,
                "matched_feature": ""
            })
            continue

        best = grp2.sort_values(["expected_partner_rank", "expected_partner_score"], ascending=[True, False]).iloc[0]
        best_rows.append({
            "panel": panel,
            "ko_gene": ko_gene,
            "expected_partner_gene": expected_partner_gene,
            "best_beta": best["beta"],
            "best_rank": int(best["expected_partner_rank"]),
            "best_score": float(best["expected_partner_score"]),
            "matched_feature": best["matched_feature"]
        })

    best_df = pd.DataFrame(best_rows).sort_values(["best_rank", "panel"], ascending=[True, True])

    print("\n=== Best beta per panel (by rank, then score) ===")
    print(best_df)

    # Save summary CSVs into the same folder
    summary_csv = os.path.join(RESULTS_DIR, "all_beta_ranks.csv")
    best_csv = os.path.join(RESULTS_DIR, "best_beta_summary.csv")
    summary.to_csv(summary_csv, index=False)
    best_df.to_csv(best_csv, index=False)

    print(f"\nSaved summary CSVs:\n- {summary_csv}\n- {best_csv}")


if __name__ == "__main__":
    main()