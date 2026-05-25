import numpy as np
import pandas as pd
import hickle as hkl
from typing import Set

# ---------- paths (lung-specific) ----------

EMBEDDING_PATH = "embeddings/final_X_tcga_lung_processed.hkl"
CRISPR_PATH    = "datasets/2023/CRISPRGeneEffect_lung_processed.hkl"

FI_DATA_PATH   = "datasets/feature_importances_lung_data.npy"
FI_INDEX_PATH  = "datasets/feature_importances_lung_index.npy"
FI_COLS_PATH   = "datasets/feature_importances_lung_columns.npy"

MODEL_PATH     = "datasets/2023/Model.csv"

# if you want to restrict to particular KOs, set e.g. ["ARID1B", "CREBBP"]
KO_WHITELIST   = None


# ---------- loading helpers ----------

def load_embedding_and_crispr():
    embedding = hkl.load(EMBEDDING_PATH)
    gene_effects_df = hkl.load(CRISPR_PATH)

    if not isinstance(embedding, pd.DataFrame):
        embedding = pd.DataFrame(embedding)
    if not isinstance(gene_effects_df, pd.DataFrame):
        gene_effects_df = pd.DataFrame(gene_effects_df)

    inter = embedding.index.intersection(gene_effects_df.index)
    if len(inter) == 0:
        raise ValueError("Embedding and CRISPR indices do not overlap.")

    embedding = embedding.loc[inter]
    gene_effects_df = gene_effects_df.loc[inter]
    return embedding, gene_effects_df


def load_feature_importances():
    vals = np.load(FI_DATA_PATH)
    idx  = np.load(FI_INDEX_PATH,  allow_pickle=True)
    cols = np.load(FI_COLS_PATH,   allow_pickle=True)
    return pd.DataFrame(vals, index=idx, columns=cols)


def get_sclc_model_ids(model_df: pd.DataFrame) -> Set[str]:
    """
    SCLC defined exactly as before:
    OncotreeSubtype == 'small cell lung cancer'
    """
    model_df[["OncotreeSubtype", "OncotreePrimaryDisease"]] = (
        model_df[["OncotreeSubtype", "OncotreePrimaryDisease"]].fillna("")
    )
    is_sclc_subtype = (
        model_df["OncotreeSubtype"].astype(str).str.strip().str.lower()
        == "small cell lung cancer"
    )
    return set(model_df.loc[is_sclc_subtype, "ModelID"].astype(str))


def load_sclc_ids(gene_effects_df: pd.DataFrame):
    model_df = pd.read_csv(MODEL_PATH)
    sclc_ids = gene_effects_df.index.intersection(get_sclc_model_ids(model_df))
    if len(sclc_ids) == 0:
        raise ValueError(
            "No SCLC ModelID values from Model.csv intersect CRISPR index."
        )
    return sclc_ids


# ---------- Cai-style scoring ----------

def compute_slrfm_scores(feature_importance_df: pd.DataFrame,
                         ko_whitelist=None) -> pd.DataFrame:
    """
    For each KO gene B (column of FI), compute Cai-style score:

        score(B) = max(v_B) - mean(v_B)

    where v_B is the PCC-reweighted feature-importance vector for KO B.

    We skip KOs whose importance column is all-NaN or all-zero.
    """
    fi = feature_importance_df.astype(float)

    if ko_whitelist is not None:
        cols = [g for g in fi.columns if g in ko_whitelist]
    else:
        cols = list(fi.columns)

    records = []

    for ko in cols:
        col = fi[ko]

        # drop NaN / inf
        col_valid = col.replace([np.inf, -np.inf], np.nan).dropna()
        if col_valid.empty:
            # no usable values for this KO → skip
            continue

        # if literally all zeros, it's not informative – skip
        if (col_valid.values == 0).all():
            continue

        v = col_valid.values
        max_w = float(v.max())
        mean_w = float(v.mean())
        score = max_w - mean_w

        top_feature = col_valid.idxmax()

        # feature name can be "ARID1A" or "ARID1A_exp"
        tokens = top_feature.split("_")
        if tokens[-1] == "exp":
            top_gene = "_".join(tokens[:-1])
            feature_type = "expression"
        else:
            top_gene = top_feature
            feature_type = "mutation"

        records.append({
            "ko_gene":       ko,
            "top_feature":   top_feature,
            "top_gene":      top_gene,
            "feature_type":  feature_type,
            "score":         score,
            "top_weight":    max_w,
        })

    if not records:
        return pd.DataFrame()

    scores_df = pd.DataFrame(records)
    scores_df = scores_df.sort_values("score", ascending=False)
    return scores_df


# ---------- add SCLC-only annotation (like their lineage view) ----------

def annotate_sclc(scores_df: pd.DataFrame,
                  embedding: pd.DataFrame,
                  gene_effects_df: pd.DataFrame,
                  sclc_ids) -> pd.DataFrame:
    """
    For each (top_gene, ko_gene) pair, compute SCLC-only Pearson correlation
    between feature (top_feature) and KO viability. This is just an annotation
    (not the calling rule), similar in spirit to their per-lineage analysis.
    """
    sclc_ids = embedding.index.intersection(gene_effects_df.index).intersection(sclc_ids)
    if len(sclc_ids) == 0:
        raise ValueError("No overlap between SCLC IDs and lung-trained matrices.")

    sclc_pcc_list = []
    n_sclc_list   = []

    for _, row in scores_df.iterrows():
        ko = row["ko_gene"]
        feat = row["top_feature"]

        if ko not in gene_effects_df.columns or feat not in embedding.columns:
            sclc_pcc_list.append(np.nan)
            n_sclc_list.append(0)
            continue

        x = embedding.loc[sclc_ids, feat]
        y = gene_effects_df.loc[sclc_ids, ko]

        # require some variation
        if x.nunique() <= 1 or y.nunique() <= 1:
            sclc_pcc_list.append(np.nan)
            n_sclc_list.append(len(sclc_ids))
            continue

        r = np.corrcoef(x, y)[0, 1]
        sclc_pcc_list.append(float(r))
        n_sclc_list.append(len(sclc_ids))

    scores_df = scores_df.copy()
    scores_df["sclc_pcc"] = sclc_pcc_list
    scores_df["n_sclc"]   = n_sclc_list
    return scores_df


# ---------- main ----------

def main():
    print("Loading lung-trained embedding + CRISPR...")
    embedding, gene_effects_df = load_embedding_and_crispr()
    print("embedding:", embedding.shape)
    print("CRISPR:   ", gene_effects_df.shape)

    fi = load_feature_importances()
    print("feature_importances (lung):", fi.shape)

    print("Computing Cai-style SLRFM scores...")
    scores = compute_slrfm_scores(fi, ko_whitelist=KO_WHITELIST)
    print("Total KOs scored:", scores.shape[0])

    print("Annotating with SCLC-only correlations...")
    sclc_ids = load_sclc_ids(gene_effects_df)
    scores = annotate_sclc(scores, embedding, gene_effects_df, sclc_ids)

    # ------------------ NEW: mutation-only + SCLC filter ------------------
    # Keep only mutation-context features
    mut_scores = scores[scores["feature_type"] == "mutation"].copy()

    # Optional but very SLI-ish:
    # keep only pairs where SCLC PCC is negative (mutants more sensitive)
    mut_scores = mut_scores[mut_scores["sclc_pcc"] < 0].copy()

    # Sort so that:
    #   - higher Cai-style score first
    #   - more negative SCLC PCC earlier (stronger SLI-like pattern)
    mut_scores = mut_scores.sort_values(
        by=["score", "sclc_pcc"],
        ascending=[False, True]
    )

    out_path = "datasets/sclc_slrfm_candidates_lung_mutation_only.csv"
    mut_scores.to_csv(out_path, index=False)
    print(f"Saved ranked mutation-only SCLC candidate list → {out_path}")

    print("\nTop 20 SCLC mutation-only candidates (Cai-style, lung-trained):")
    print(
        mut_scores[
            ["ko_gene", "top_gene", "feature_type",
             "score", "top_weight", "sclc_pcc", "n_sclc"]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
