import os, re
import numpy as np
import pandas as pd
import torch
import hickle as hkl

from tqdm import tqdm
from numpy.linalg import norm
from sklearn.metrics import pairwise as kernel

SCLC_EMBED_HKL = "embeddings/final_X_sclc_processed.hkl"
SCLC_GENE_EFFECT_HKL = "datasets/2023/CRISPRGeneEffect_sclc_processed.hkl"

FEATURE_IMPORTANCE_PAN_PREFIX  = "datasets/feature_importances_pan"
FEATURE_IMPORTANCE_SCLC_PREFIX = "datasets/feature_importances_lung"

# Output folder + beta sweep naming
RESULTS_DIR = "pan_lung_results_v2"
BETA_STEP = 0.05

# Kernel settings 
BANDWIDTH = 1.0
REG = 1e-5
L_GRAD = 1.0 

# Weight vector construction + stabilization
WEIGHT_AGG = "mean_abs"  
CLIP_LOW = 0.5
CLIP_HIGH = 2.0

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
    One row per KO gene: top feature + gaps.
    Uses gap_top1_mean_other instead of top1 vs top10.
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

def rank_all_features_for_one_ko(feature_importance, ko_gene) :
    s = feature_importance[ko_gene].sort_values(ascending=False)
    df = s.reset_index()
    df.columns = ["feature", "score"]
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["feature_type"] = np.where(df["feature"].astype(str).str.endswith("_exp"), "exp", "mut_or_other")
    df["partner_gene"] = df["feature"].astype(str).str.replace(r"_exp$", "", regex=True)
    return df

def build_all_candidates_long(feature_importance, model_tag):
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
    if ko_gene not in feature_importance.columns:
        return np.nan, np.nan, ""
    df = rank_all_features_for_one_ko(feature_importance, ko_gene)
    hits = df[df["partner_gene"] == expected_partner_gene]
    if len(hits) == 0:
        return np.nan, np.nan, ""
    best = hits.sort_values("rank", ascending=True).iloc[0]
    return int(best["rank"]), float(best["score"]), str(best["feature"])


def train_krr_from_embedding(cell_embedding_df, gene_effects_df, bandwidth, reg, device):
    """
    Closed-form solve:
      sol = (K + reg I)^{-1} Y
    Returns sol with shape (n_cells, n_knockouts)
    """
    X_np = cell_embedding_df.values
    dists_np = kernel.euclidean_distances(X_np, X_np)
    cell_distances = torch.tensor(dists_np, device=device).float()
    cell_distances.fill_diagonal_(0)

    Y = torch.tensor(gene_effects_df.values, device=device).float()

    K = torch.exp(-bandwidth * (cell_distances)**0.5)
    sol = torch.linalg.solve(K + reg * torch.eye(K.shape[0], device=device), Y)
    return sol

def euclidean_distances_torch(samples, centers, M=None, squared=True, diag_only=False):
    if M is None:
        samples_norm = torch.sum(samples**2, dim=1, keepdim=True)
    else:
        if diag_only:
            samples_norm = (samples * M) * samples
        else:
            samples_norm = (samples @ M) * samples
        samples_norm = torch.sum(samples_norm, dim=1, keepdims=True)

    if samples is centers:
        centers_norm = samples_norm
    else:
        if M is None:
            centers_norm = torch.sum(centers**2, dim=1, keepdims=True)
        else:
            if diag_only:
                centers_norm = (centers * M) * centers
            else:
                centers_norm = (centers @ M) * centers
            centers_norm = torch.sum(centers_norm, dim=1, keepdims=True)
    centers_norm = torch.reshape(centers_norm, (1, -1))

    distances = samples.mm(torch.t(centers))
    distances.mul_(-2)
    distances.add_(samples_norm)
    distances.add_(centers_norm)
    if not squared:
        distances.clamp_(min=0)
        distances.sqrt_()
    return distances

def laplace_kernel(samples, centers, bandwidth, M=None, diag_only=False):
    kernel_mat = euclidean_distances_torch(samples, centers, M=M, squared=False, diag_only=diag_only)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def get_grads(X: torch.Tensor, sol_T: torch.Tensor, P: torch.Tensor, L=1.0, diag_only=True):
    """
    X: (n, d)
    sol_T: (num_kos, n)
    returns grads: (d, num_kos)
    """
    K = laplace_kernel(X, X, bandwidth=1, M=P, diag_only=diag_only)

    dist = euclidean_distances_torch(X, X, M=P, squared=False, diag_only=diag_only)
    dist.clamp_(min=0)
    dist[dist < 1e-10] = 0

    K = K / torch.where(dist == 0, torch.ones_like(dist), dist)
    K[torch.isinf(K)] = 0.0

    n, d = X.shape
    num_kos, n2 = sol_T.shape
    assert n == n2

    grads = torch.zeros((d, num_kos), device=X.device)
    for i in tqdm(range(num_kos), desc="grads"):
        weight = sol_T[i, :].reshape((-1, 1))
        step2 = K @ (weight * X)
        step3 = (weight.T @ K).T * X
        G = (step2 - step3) * (-1.0 / L)
        G = torch.sum(G**2, axis=0)
        grads[:, i] = G / n
    return grads

def get_pcc(cell_embedding: pd.DataFrame, gene_effects_df: pd.DataFrame) -> pd.DataFrame:
    exp_cols = [c for c in cell_embedding.columns if c.split("_")[-1] == "exp"]

    std_val = cell_embedding[exp_cols].std(axis=0).replace(0, 1)
    z = (cell_embedding[exp_cols] - cell_embedding[exp_cols].mean(axis=0)) / std_val

    emb = cell_embedding.copy()
    emb[exp_cols] *= (np.abs(z) < 3).fillna(0).astype(int)

    normalized_cell_embedding = emb - emb.mean(axis=0)
    normalized_gene_effects_df = gene_effects_df - gene_effects_df.mean(axis=0)

    cell_norms = (normalized_cell_embedding**2).sum(axis=0).values
    gene_norms = (normalized_gene_effects_df**2).sum(axis=0).values

    pcc = (normalized_cell_embedding.T @ normalized_gene_effects_df) / (
        cell_norms.reshape((-1, 1)) @ gene_norms.reshape((1, -1))
    )**0.5

    return pd.DataFrame(pcc, columns=normalized_gene_effects_df.columns, index=normalized_cell_embedding.columns)

def compute_feature_importance_df(cell_embedding_df,gene_effects_df, device):
    # Train
    sol = train_krr_from_embedding(cell_embedding_df, gene_effects_df, BANDWIDTH, REG, device=device)

    # Grads
    X_t = torch.tensor(cell_embedding_df.values, device=device).float()
    d = X_t.shape[1]
    P = torch.ones(d, device=device).double()
    grads_t = get_grads(X_t, sol.T, P=P, L=L_GRAD, diag_only=True)

    grads_df = pd.DataFrame(grads_t.detach().cpu().numpy(),
                            index=cell_embedding_df.columns,
                            columns=gene_effects_df.columns)

    # PCC
    pcc = get_pcc(cell_embedding_df, gene_effects_df).fillna(0)
    pcc = pcc.loc[grads_df.index]

    mut = [x for x in pcc.index if x.split("_")[-1] != "exp"]
    pcc.loc[mut] = -(pcc.loc[mut].clip(upper=0))

    exp = [x for x in pcc.index if x.split("_")[-1] == "exp"]
    pcc.loc[exp] = abs(pcc.loc[exp])

    return grads_df * pcc


#weight vector building
def feature_weights_from_feature_importance(fi, agg):
    """
    fi: (features x KOs)
    returns per-feature weights (index = features)
    """
    A = fi.abs()
    if agg == "mean_abs":
        w = A.mean(axis=1)
    elif agg == "median_abs":
        w = A.median(axis=1)
    return w

def stabilize_weights(w):
    # normalize mean to 1, then clip
    w = w.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean = w.mean()
    if mean > 0:
        w = w / mean
    w = w.clip(lower=CLIP_LOW, upper=CLIP_HIGH)
    return w

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base SCLC embedding + gene effects 
    sclc_embedding = hkl.load(SCLC_EMBED_HKL)
    sclc_gene_effects = hkl.load(SCLC_GENE_EFFECT_HKL)

    # Keep consistent ordering
    assert sclc_embedding.index.equals(sclc_gene_effects.index)

    # Load feature_importance matrices for deriving source/target weights
    fi_pan = load_feature_importance(FEATURE_IMPORTANCE_PAN_PREFIX)
    fi_sclc = load_feature_importance(FEATURE_IMPORTANCE_SCLC_PREFIX)

    # Derive per-feature weights and align to SCLC embedding columns
    w_pan = feature_weights_from_feature_importance(fi_pan, WEIGHT_AGG)
    w_sclc = feature_weights_from_feature_importance(fi_sclc, WEIGHT_AGG)

    # Align weights to the actual feature columns used in SCLC embedding
    feat_cols = sclc_embedding.columns
    w_pan = w_pan.reindex(feat_cols).fillna(0.0)
    w_sclc = w_sclc.reindex(feat_cols).fillna(0.0)

    w_pan = stabilize_weights(w_pan)
    w_sclc = stabilize_weights(w_sclc)

    betas = [round(x, 2) for x in np.arange(0, 1.0001, BETA_STEP)]
    results_rows = []

    for beta in betas:
        print(f"\n=== beta={beta} ===")

        # Blend weights
        w_beta = (1.0 - beta) * w_pan + beta * w_sclc
        w_beta = stabilize_weights(w_beta)
        print("weights blended")

        # Weighted embedding: scale columns by sqrt(w_beta)
        s = np.sqrt(w_beta.values).reshape(1, -1)
        emb_beta = sclc_embedding.copy()
        emb_beta.loc[:, :] = emb_beta.values * s
        print("weights scaled")

        # L2 normalize rows again (keeps scale comparable)
        emb_beta = emb_beta / norm(emb_beta, axis=1).reshape(-1, 1)
        print("L2 normalize")

        # Retrain + compute feature importance on this weighted embedding
        fi_beta = compute_feature_importance_df(emb_beta, sclc_gene_effects, device=device)
        print("retrain")

        # Export Excel
        out_xlsx = os.path.join(RESULTS_DIR, f"beta_{beta}.xlsx")
        sheet1 = build_top_pair_per_ko(fi_beta)

        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            sheet1.to_excel(writer, sheet_name="TopPairPerKO", index=False)

            for title, ko_gene, expected_partner_gene in KNOWN_SLI_PANELS:
                sheet = safe_sheet_name(title)
                if ko_gene not in fi_beta.columns:
                    pd.DataFrame([{
                        "error": f"KO gene '{ko_gene}' not found in this matrix.",
                        "available_example_kos": ", ".join(map(str, fi_beta.columns[:25]))
                    }]).to_excel(writer, sheet_name=sheet, index=False)
                    continue

                df = rank_all_features_for_one_ko(fi_beta, ko_gene)
                df = df[["rank", "feature", "partner_gene", "feature_type", "score"]]
                df.to_excel(writer, sheet_name=sheet, index=False)

        print(f"Saved: {out_xlsx}")

        # Summary ranks
        for title, ko_gene, expected_partner_gene in KNOWN_SLI_PANELS:
            rnk, scr, matched_feat = get_partner_rank_and_score(fi_beta, ko_gene, expected_partner_gene)
            results_rows.append({
                "beta": beta,
                "panel": title,
                "ko_gene": ko_gene,
                "expected_partner_gene": expected_partner_gene,
                "expected_partner_rank": rnk,
                "expected_partner_score": scr,
                "matched_feature": matched_feat
            })

    # Save summary CSVs
    summary = pd.DataFrame(results_rows)
    summary_csv = os.path.join(RESULTS_DIR, "all_beta_ranks_retrain.csv")
    best_csv = os.path.join(RESULTS_DIR, "best_beta_summary_retrain.csv")

    # Best beta per panel
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

    summary.to_csv(summary_csv, index=False)
    best_df.to_csv(best_csv, index=False)

    print("\nSaved summary CSVs:")
    print("-", summary_csv)
    print("-", best_csv)


if __name__ == "__main__":
    main()