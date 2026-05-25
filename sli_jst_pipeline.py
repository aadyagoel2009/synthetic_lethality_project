"""
Joint Source-Target SL-RFM (JST-SLRFM) for SCLC SLI discovery.

Single unified method (all knockouts / panels):
  1. Source:     joint unweighted KRR on pan -> alpha_0
  2. Alignment:  KMM row weights omega toward SCLC target distribution
  3. Transfer:   joint weighted KRR with P_pan and ridge prior to alpha_0
                 (K_w + (REG+gamma)I) alpha* = Y_w + gamma * alpha_0
  4. Structure:  AGOP(alpha*) only (no PCC in structural channel)
  5. Context:    importance-weighted PCC on all pan lines with omega
  6. Score:      S = structure * context

Baselines (same evaluation): pan_fi, hybrid_pan_fi_x_sclc_pcc, sclc_pcc_only.

Does not use per-KO transfer, tiered 100/10/1 weights, or hybrid-style double PCC.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

import hickle as hkl
import numpy as np
import pandas as pd
import torch
from numpy.linalg import norm
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PAN_EMBED_HKL = "embeddings/final_X_tcga_processed.hkl"
PAN_GENE_EFFECT_HKL = "datasets/2023/CRISPRGeneEffect_processed.hkl"
SCLC_EMBED_HKL = "embeddings/final_X_sclc_processed.hkl"
SCLC_GENE_EFFECT_HKL = "datasets/2023/CRISPRGeneEffect_sclc_processed.hkl"
FEATURE_IMPORTANCE_PAN_PREFIX = "datasets/feature_importances_pan"
MODEL_CSV_PATH = "datasets/2023/Model.csv"

RESULTS_DIR = "sli_jst_results"

BANDWIDTH = 1.0
REG = 1e-5
L_GRAD = 1.0
P_CLIP_LOW = 0.5
P_CLIP_HIGH = 2.0

# KMM: match weighted mean embedding to target mixture, then clip weights
KMM_W_MIN = 0.1
KMM_W_MAX = 10.0
KMM_N_ITER = 25
KMM_TARGET_SCLC_FRAC = 1.0  # 1.0 = SCLC-only target; <1 mixes lung NSCLC into target mean

# Pan-prior strength (one global gamma for all panels; sweep on validation)
JST_GAMMA_SWEEP: List[float] = [0.01, 0.1, 1.0, 10.0]
DEFAULT_JST_GAMMA = 1.0

JST_COMPUTE_ALL_KOS = False

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


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def as_dataframe(obj: object, name: str = "data") -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, pd.Series):
        return obj.to_frame().T
    try:
        return pd.DataFrame(obj)
    except Exception as exc:
        raise TypeError(f"Could not convert {name} to DataFrame") from exc


def load_hkl_dataframe(path: str, name: str) -> pd.DataFrame:
    return as_dataframe(hkl.load(path), name)


def load_feature_importance(prefix: str) -> pd.DataFrame:
    data = np.load(prefix + "_data.npy", allow_pickle=True)
    idx = np.load(prefix + "_index.npy", allow_pickle=True)
    cols = np.load(prefix + "_columns.npy", allow_pickle=True)
    return pd.DataFrame(data, index=idx, columns=cols)


def ensure_row_l2_normalized(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    norms = norm(out.values, axis=1).reshape(-1, 1)
    norms[norms == 0] = 1.0
    out.loc[:, :] = out.values / norms
    return out


def align_embeddings_and_effects(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    sclc_embedding: pd.DataFrame,
    sclc_effects: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pan_rows = pan_embedding.index.intersection(pan_effects.index)
    sclc_rows = sclc_embedding.index.intersection(sclc_effects.index)
    if len(pan_rows) == 0 or len(sclc_rows) == 0:
        raise ValueError("No overlapping cell IDs between embedding and gene effects.")

    common_cols = pan_embedding.columns.intersection(sclc_embedding.columns)
    common_kos = pan_effects.columns.intersection(sclc_effects.columns)
    if len(common_cols) == 0 or len(common_kos) == 0:
        raise ValueError("No overlapping features or KO genes between pan and SCLC.")

    return (
        pan_embedding.loc[pan_rows, common_cols],
        pan_effects.loc[pan_rows, common_kos],
        sclc_embedding.loc[sclc_rows, common_cols],
        sclc_effects.loc[sclc_rows, common_kos],
    )


def validate_sclc_in_pan(
    pan_embedding: pd.DataFrame,
    sclc_embedding: pd.DataFrame,
) -> pd.Index:
    missing = sclc_embedding.index.difference(pan_embedding.index)
    if len(missing) > 0:
        sample = ", ".join(map(str, list(missing[:5])))
        raise ValueError(
            f"{len(missing)} SCLC IDs missing from pan embedding (e.g. {sample})."
        )
    common = sclc_embedding.index.intersection(pan_embedding.index)
    if not np.allclose(
        pan_embedding.loc[common].values,
        sclc_embedding.loc[common].values,
        rtol=1e-5,
        atol=1e-8,
        equal_nan=True,
    ):
        diff = np.nanmax(
            np.abs(pan_embedding.loc[common].values - sclc_embedding.loc[common].values)
        )
        print(f"Warning: SCLC .hkl rows differ from pan (max diff={diff:.6g}); using pan.")
    return common


def load_depmap_lineage_ids(
    model_path: str = MODEL_CSV_PATH,
) -> Tuple[set, set]:
    meta = pd.read_csv(model_path)
    meta[["OncotreeSubtype", "OncotreePrimaryDisease"]] = meta[
        ["OncotreeSubtype", "OncotreePrimaryDisease"]
    ].fillna("")
    is_sclc = (
        meta["OncotreeSubtype"].astype(str).str.strip().str.lower() == "small cell lung cancer"
    )
    is_lung = (
        meta["OncotreePrimaryDisease"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.contains("lung")
    )
    sclc_ids = set(meta.loc[is_sclc, "ModelID"].astype(str))
    lung_ids = set(meta.loc[is_lung, "ModelID"].astype(str))
    return sclc_ids, lung_ids - sclc_ids


def intersect_lineage_with_pan(
    pan_index: pd.Index,
    sclc_ids: set,
    lung_nsclc_ids: set,
) -> Tuple[pd.Index, pd.Index]:
    sclc_in_pan = pd.Index([i for i in pan_index if i in sclc_ids])
    lung_in_pan = pd.Index([i for i in pan_index if i in lung_nsclc_ids])
    return sclc_in_pan, lung_in_pan


def print_dataset_diagnostics(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    sclc_ids_in_pan: pd.Index,
    lung_ids_in_pan: pd.Index,
    device: torch.device,
) -> None:
    n = len(pan_embedding)
    print("\n--- Dataset diagnostics ---")
    print(f"  Device:           {device}")
    print(f"  Pan lines:        {n}")
    print(f"  SCLC in pan:      {len(sclc_ids_in_pan)}")
    print(f"  Lung NSCLC:       {len(lung_ids_in_pan)}")
    print(f"  Features:         {pan_embedding.shape[1]}")
    print(f"  KO genes:         {pan_effects.shape[1]}")
    print(f"  K matrix ~MB:     {(n * n * 4) / 1024**2:.0f}")
    print("----------------------------\n")


# ---------------------------------------------------------------------------
# Kernel / KRR
# ---------------------------------------------------------------------------
def euclidean_distances_torch(
    samples: torch.Tensor,
    centers: torch.Tensor,
    M: Optional[torch.Tensor] = None,
    squared: bool = True,
    diag_only: bool = False,
) -> torch.Tensor:
    if M is None:
        samples_norm = torch.sum(samples**2, dim=1, keepdim=True)
    else:
        samples_norm = (samples * M) * samples if diag_only else (samples @ M) * samples
        samples_norm = torch.sum(samples_norm, dim=1, keepdims=True)

    if samples is centers:
        centers_norm = samples_norm
    else:
        if M is None:
            centers_norm = torch.sum(centers**2, dim=1, keepdims=True)
        else:
            centers_norm = (centers * M) * centers if diag_only else (centers @ M) * centers
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


def build_kernel_matrix(
    X_t: torch.Tensor,
    bandwidth: float,
    device: torch.device,
    sample_weights: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None,
) -> torch.Tensor:
    P_t = torch.tensor(P, device=device).double() if P is not None else None
    dist = euclidean_distances_torch(X_t, X_t, M=P_t, squared=False, diag_only=True)
    dist.fill_diagonal_(0)
    K = torch.exp(-bandwidth * dist)
    if sample_weights is not None:
        scale = torch.tensor(np.sqrt(sample_weights), device=device).float()
        K = scale.unsqueeze(1) * K * scale.unsqueeze(0)
    return K


def train_krr_joint(
    cell_embedding_df: pd.DataFrame,
    gene_effects_df: pd.DataFrame,
    device: torch.device,
    bandwidth: float = BANDWIDTH,
    reg: float = REG,
    sample_weights: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None,
    alpha_prior: Optional[np.ndarray] = None,
    gamma: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Joint KRR for all KO columns.

    With gamma > 0 and alpha_prior: (K_w + (reg+gamma)I) alpha = Y_w + gamma * alpha_prior.
    """
    X_t = torch.tensor(cell_embedding_df.values, device=device).float()
    Y_t = torch.tensor(gene_effects_df.values, device=device).float()
    n = X_t.shape[0]

    K = build_kernel_matrix(X_t, bandwidth, device, sample_weights, P)
    if sample_weights is not None:
        scale = torch.tensor(np.sqrt(sample_weights), device=device).float()
        Y_t = scale.unsqueeze(1) * Y_t

    I = torch.eye(n, device=device)
    if alpha_prior is not None and gamma > 0:
        a_p = torch.tensor(alpha_prior, device=device).float()
        sol = torch.linalg.solve(K + (reg + gamma) * I, Y_t + gamma * a_p)
    else:
        sol = torch.linalg.solve(K + reg * I, Y_t)
    return sol, X_t


def laplace_kernel(
    samples: torch.Tensor,
    centers: torch.Tensor,
    bandwidth: float,
    M: Optional[torch.Tensor] = None,
    diag_only: bool = False,
) -> torch.Tensor:
    kernel_mat = euclidean_distances_torch(
        samples, centers, M=M, squared=False, diag_only=diag_only
    )
    kernel_mat.clamp_(min=0)
    kernel_mat.mul_(-1.0 / bandwidth)
    kernel_mat.exp_()
    return kernel_mat


def get_grads(
    X: torch.Tensor,
    sol_T: torch.Tensor,
    P: torch.Tensor,
    L: float = L_GRAD,
    diag_only: bool = True,
    bandwidth: float = BANDWIDTH,
) -> torch.Tensor:
    K = laplace_kernel(X, X, bandwidth=bandwidth, M=P, diag_only=diag_only)
    dist = euclidean_distances_torch(X, X, M=P, squared=False, diag_only=diag_only)
    dist.clamp_(min=0)
    dist[dist < 1e-10] = 0
    K = K / torch.where(dist == 0, torch.ones_like(dist), dist)
    K[torch.isinf(K)] = 0.0

    n, d = X.shape
    num_kos, _ = sol_T.shape
    grads = torch.zeros((d, num_kos), device=X.device)
    for i in range(num_kos):
        weight = sol_T[i, :].reshape((-1, 1))
        step2 = K @ (weight * X)
        step3 = (weight.T @ K).T * X
        G = (step2 - step3) * (-1.0 / L)
        grads[:, i] = torch.sum(G**2, axis=0) / n
    return grads


# ---------------------------------------------------------------------------
# PCC
# ---------------------------------------------------------------------------
def _embedding_with_exp_outlier_mask(cell_embedding: pd.DataFrame) -> pd.DataFrame:
    exp_cols = [c for c in cell_embedding.columns if c.split("_")[-1] == "exp"]
    std_val = cell_embedding[exp_cols].std(axis=0).replace(0, 1)
    z = (cell_embedding[exp_cols] - cell_embedding[exp_cols].mean(axis=0)) / std_val
    emb = cell_embedding.copy()
    emb[exp_cols] *= (np.abs(z) < 3).fillna(0).astype(int)
    return emb


def get_pcc(
    cell_embedding: pd.DataFrame,
    gene_effects_df: pd.DataFrame,
) -> pd.DataFrame:
    emb = _embedding_with_exp_outlier_mask(cell_embedding)
    norm_emb = emb - emb.mean(axis=0)
    norm_ge = gene_effects_df - gene_effects_df.mean(axis=0)
    cell_norms = (norm_emb**2).sum(axis=0).values
    gene_norms = (norm_ge**2).sum(axis=0).values
    pcc = (norm_emb.T @ norm_ge) / (
        cell_norms.reshape(-1, 1) @ gene_norms.reshape((1, -1))
    ) ** 0.5
    return pd.DataFrame(pcc, index=norm_emb.columns, columns=norm_ge.columns)


def get_pcc_weighted(
    cell_embedding: pd.DataFrame,
    gene_effects_df: pd.DataFrame,
    sample_weights: np.ndarray,
) -> pd.DataFrame:
    """Weighted Pearson: features (rows) x KO columns."""
    emb = _embedding_with_exp_outlier_mask(cell_embedding)
    w = np.asarray(sample_weights, dtype=float)
    w = w / w.sum()
    X = emb.values
    Y = gene_effects_df.values
    n, _ = X.shape
    if len(w) != n:
        raise ValueError("sample_weights length must match number of cell lines.")

    mu_x = (w[:, None] * X).sum(axis=0)
    mu_y = (w[:, None] * Y).sum(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y
    cov = Xc.T @ (w[:, None] * Yc)
    var_x = (w[:, None] * Xc**2).sum(axis=0)
    var_y = (w[:, None] * Yc**2).sum(axis=0)
    denom = np.sqrt(var_x).reshape(-1, 1) * np.sqrt(var_y).reshape(1, -1)
    denom = np.where(denom == 0, np.nan, denom)
    pcc = cov / denom
    return pd.DataFrame(pcc, index=emb.columns, columns=gene_effects_df.columns)


def process_pcc_for_sli(pcc: pd.DataFrame) -> pd.DataFrame:
    out = pcc.copy().fillna(0)
    mut = [x for x in out.index if x.split("_")[-1] != "exp"]
    exp = [x for x in out.index if x.split("_")[-1] == "exp"]
    out.loc[mut] = -(out.loc[mut].clip(upper=0))
    out.loc[exp] = out.loc[exp].abs()
    return out


def stabilize_metric_p(p: pd.Series) -> pd.Series:
    p = p.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if p.mean() > 0:
        p = p / p.mean()
    return p.clip(lower=P_CLIP_LOW, upper=P_CLIP_HIGH)


# ---------------------------------------------------------------------------
# KMM row weights (linear mean matching + clip)
# ---------------------------------------------------------------------------
def compute_target_centroid(
    X: np.ndarray,
    sclc_row_idx: np.ndarray,
    lung_row_idx: np.ndarray,
    sclc_frac: float,
) -> np.ndarray:
    mu_s = X[sclc_row_idx].mean(axis=0)
    if sclc_frac >= 1.0 or len(lung_row_idx) == 0:
        return mu_s
    mu_l = X[lung_row_idx].mean(axis=0)
    return sclc_frac * mu_s + (1.0 - sclc_frac) * mu_l


def compute_kmm_weights(
    pan_embedding: pd.DataFrame,
    sclc_ids_in_pan: pd.Index,
    lung_ids_in_pan: pd.Index,
    sclc_frac: float = KMM_TARGET_SCLC_FRAC,
    w_min: float = KMM_W_MIN,
    w_max: float = KMM_W_MAX,
    n_iter: int = KMM_N_ITER,
) -> np.ndarray:
    """
    Align weighted training mean in embedding space to target centroid (KMM-style).
    Returns omega with sum(omega) = n_lines.
    """
    X = pan_embedding.values.astype(float)
    n = X.shape[0]
    index = pan_embedding.index
    sclc_idx = np.array([i for i, row in enumerate(index) if row in sclc_ids_in_pan])
    lung_idx = np.array([i for i, row in enumerate(index) if row in lung_ids_in_pan])

    if len(sclc_idx) == 0:
        raise ValueError("No SCLC lines in pan embedding for KMM target.")

    mu_target = compute_target_centroid(X, sclc_idx, lung_idx, sclc_frac)
    d = np.linalg.norm(X - mu_target, axis=1)
    sigma = float(np.median(d[sclc_idx])) + 1e-8

    omega = np.exp(-0.5 * (d / sigma) ** 2)
    omega = omega / omega.mean()

    for _ in range(n_iter):
        mu_w = np.average(X, axis=0, weights=omega)
        residual = mu_w - mu_target
        scale = float(np.linalg.norm(residual)) + 1e-8
        adjust = np.exp(-0.5 * (np.linalg.norm(X - mu_target, axis=1) / (sigma + 0.25 * scale)) ** 2)
        omega = omega * adjust
        omega = np.clip(omega, w_min, w_max)
        omega = omega * n / omega.sum()

    n_s = len(sclc_idx)
    n_l = len(lung_idx)
    print(
        f"  KMM weights (target SCLC frac={sclc_frac:.2f}): "
        f"sum={omega.sum():.1f}, mean={omega.mean():.3f}, "
        f"SCLC mean w={omega[sclc_idx].mean():.3f} ({n_s} lines), "
        f"lung mean w={omega[lung_idx].mean():.3f} ({n_l} lines) "
        f"if lung present"
    )
    return omega


# ---------------------------------------------------------------------------
# P_pan and pan FI (baseline)
# ---------------------------------------------------------------------------
def load_or_compute_p_pan(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    device: torch.device,
    results_dir: str = RESULTS_DIR,
) -> pd.Series:
    for subdir in (results_dir, "sli_transfer_results"):
        csv_path = os.path.join(subdir, "P_pan.csv")
        if os.path.exists(csv_path):
            print(f"Loading cached P_pan from {csv_path}")
            p_pan = pd.read_csv(csv_path, index_col=0).squeeze(axis=1)
            return p_pan.reindex(pan_embedding.columns).fillna(1.0)

    print("Computing P_pan from pan AGOP...")
    alpha0, X_t = train_krr_joint(
        pan_embedding, pan_effects, device, P=None, gamma=0.0
    )
    P_ones = torch.ones(X_t.shape[1], device=device).double()
    grads = get_grads(X_t, alpha0.T, P=P_ones)
    p_pan = stabilize_metric_p(
        pd.Series(grads.detach().cpu().numpy().mean(axis=1), index=pan_embedding.columns)
    )
    os.makedirs(results_dir, exist_ok=True)
    p_pan.to_csv(os.path.join(results_dir, "P_pan.csv"))
    np.save(os.path.join(results_dir, "P_pan.npy"), p_pan.values)
    return p_pan


def compute_pan_fi(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    prefix = FEATURE_IMPORTANCE_PAN_PREFIX
    if all(os.path.exists(prefix + s) for s in ("_data.npy", "_index.npy", "_columns.npy")):
        print(f"Loading precomputed pan FI from {prefix}")
        fi = load_feature_importance(prefix)
        return fi.reindex(index=pan_embedding.columns, columns=pan_effects.columns).fillna(0)

    print("Computing pan FI (joint KRR + AGOP x PCC on pan lines)...")
    alpha0, X_t = train_krr_joint(pan_embedding, pan_effects, device, P=None, gamma=0.0)
    P_ones = torch.ones(X_t.shape[1], device=device).double()
    grads = get_grads(X_t, alpha0.T, P=P_ones)
    grads_df = pd.DataFrame(
        grads.cpu().numpy(), index=pan_embedding.columns, columns=pan_effects.columns
    )
    pcc = process_pcc_for_sli(get_pcc(pan_embedding, pan_effects))
    pcc = pcc.reindex(index=grads_df.index, columns=grads_df.columns).fillna(0)
    return grads_df * pcc


# ---------------------------------------------------------------------------
# JST-SLRFM (primary)
# ---------------------------------------------------------------------------
def compute_jst_slrfm(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    device: torch.device,
    p_pan_values: np.ndarray,
    omega: np.ndarray,
    gamma: float,
    ko_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Joint source-target SL-RFM: AGOP(alpha*) * PCC_omega (separate channels).
    """
    effects = pan_effects if ko_columns is None else pan_effects.loc[:, ko_columns]

    print(f"  JST source KRR (joint, unweighted)...")
    alpha0, X_t = train_krr_joint(
        pan_embedding, effects, device, P=None, gamma=0.0
    )
    alpha0_np = alpha0.detach().cpu().numpy()

    print(f"  JST transfer KRR (joint, KMM weights, gamma={gamma})...")
    alpha_star, _ = train_krr_joint(
        pan_embedding,
        effects,
        device,
        sample_weights=omega,
        P=p_pan_values,
        alpha_prior=alpha0_np,
        gamma=gamma,
    )

    P_t = torch.tensor(p_pan_values, device=device).double()
    structure = get_grads(X_t, alpha_star.T, P=P_t)
    structure_df = pd.DataFrame(
        structure.cpu().numpy(),
        index=pan_embedding.columns,
        columns=effects.columns,
    )

    context = process_pcc_for_sli(
        get_pcc_weighted(pan_embedding, effects, omega)
    )
    context = context.reindex(index=structure_df.index, columns=structure_df.columns).fillna(0)

    return structure_df * context


def jst_method_tag(gamma: float) -> str:
    return f"jst_kmm_g{gamma:g}"


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def hybrid_score_for_ko(
    fi_pan: pd.DataFrame, pcc_sclc: pd.DataFrame, ko_gene: str
) -> pd.Series:
    if ko_gene not in fi_pan.columns or ko_gene not in pcc_sclc.columns:
        return pd.Series(dtype=float)
    pan_col = fi_pan[ko_gene].abs()
    sclc_col = pcc_sclc[ko_gene]
    idx = pan_col.index.intersection(sclc_col.index)
    return pan_col.loc[idx] * sclc_col.loc[idx].abs()


def build_hybrid_fi(
    fi_pan: pd.DataFrame, pcc_sclc: pd.DataFrame, ko_columns: List[str]
) -> pd.DataFrame:
    cols = {}
    for ko in ko_columns:
        cols[ko] = hybrid_score_for_ko(fi_pan, pcc_sclc, ko)
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def rank_all_features_for_one_ko(scores: pd.Series, ko_gene: str) -> pd.DataFrame:
    s = scores.sort_values(ascending=False)
    df = s.reset_index()
    df.columns = ["feature", "score"]
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["ko_gene"] = ko_gene
    df["partner_gene"] = df["feature"].astype(str).str.replace(r"_exp$", "", regex=True)
    return df


def get_partner_rank_and_score(
    scores: pd.Series, ko_gene: str, expected_partner: str
) -> Tuple[float, float, str]:
    df = rank_all_features_for_one_ko(scores, ko_gene)
    hits = df[df["partner_gene"] == expected_partner]
    if len(hits) == 0:
        return np.nan, np.nan, ""
    best = hits.sort_values("rank").iloc[0]
    return int(best["rank"]), float(best["score"]), str(best["feature"])


def build_top_pair_per_ko(fi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ko in fi.columns:
        s = fi[ko].sort_values(ascending=False)
        top1 = float(s.iloc[0])
        other_mean = float(s.iloc[1:].mean()) if len(s) > 1 else np.nan
        rows.append(
            {
                "ko_gene": ko,
                "top1_feature": str(s.index[0]),
                "top1_partner_gene": re.sub(r"_exp$", "", str(s.index[0])),
                "top1_score": top1,
                "gap_top1_mean_other": top1 - other_mean if len(s) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("top1_score", ascending=False)


def evaluate_panel(fi: pd.DataFrame, method_name: str) -> pd.DataFrame:
    rows = []
    for title, ko, partner in KNOWN_SLI_PANELS:
        col = fi[ko] if ko in fi.columns else pd.Series(dtype=float)
        rank, score, feat = get_partner_rank_and_score(col, ko, partner)
        rows.append(
            {
                "method": method_name,
                "panel": title,
                "ko_gene": ko,
                "expected_partner_gene": partner,
                "expected_partner_rank": rank,
                "expected_partner_score": score,
                "matched_feature": feat,
            }
        )
    return pd.DataFrame(rows)


def mean_panel_rank(eval_df: pd.DataFrame, method: str) -> float:
    sub = eval_df.loc[eval_df["method"] == method, "expected_partner_rank"].dropna()
    return float(sub.mean()) if len(sub) else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pan_embedding = ensure_row_l2_normalized(load_hkl_dataframe(PAN_EMBED_HKL, "pan"))
    pan_effects = load_hkl_dataframe(PAN_GENE_EFFECT_HKL, "pan_effects")
    sclc_embedding = ensure_row_l2_normalized(load_hkl_dataframe(SCLC_EMBED_HKL, "sclc"))
    sclc_effects = load_hkl_dataframe(SCLC_GENE_EFFECT_HKL, "sclc_effects")
    pan_embedding, pan_effects, sclc_embedding, sclc_effects = align_embeddings_and_effects(
        pan_embedding, pan_effects, sclc_embedding, sclc_effects
    )

    validate_sclc_in_pan(pan_embedding, sclc_embedding)
    sclc_meta, lung_meta = load_depmap_lineage_ids()
    sclc_in_pan, lung_in_pan = intersect_lineage_with_pan(
        pan_embedding.index, sclc_meta, lung_meta
    )
    print_dataset_diagnostics(pan_embedding, pan_effects, sclc_in_pan, lung_in_pan, device)

    p_pan = load_or_compute_p_pan(pan_embedding, pan_effects, device)
    p_pan_values = p_pan.values

    print("\n=== KMM target alignment weights ===")
    omega = compute_kmm_weights(
        pan_embedding, sclc_in_pan, lung_in_pan, sclc_frac=KMM_TARGET_SCLC_FRAC
    )
    np.save(os.path.join(RESULTS_DIR, "kmm_omega.npy"), omega)
    pd.Series(omega, index=pan_embedding.index).to_csv(
        os.path.join(RESULTS_DIR, "kmm_omega.csv")
    )

    pcc_sclc = process_pcc_for_sli(get_pcc(sclc_embedding, sclc_effects))

    panel_kos = sorted({ko for _, ko, _ in KNOWN_SLI_PANELS})
    eval_kos = list(pan_effects.columns) if JST_COMPUTE_ALL_KOS else panel_kos

    eval_rows: List[pd.DataFrame] = []

    print("\n=== Baseline: pan_fi ===")
    fi_pan = compute_pan_fi(pan_embedding, pan_effects.loc[:, eval_kos], device)
    eval_rows.append(evaluate_panel(fi_pan, "pan_fi"))

    print("\n=== Baseline: hybrid (pan FI x |SCLC PCC|) ===")
    fi_hybrid = build_hybrid_fi(fi_pan, pcc_sclc, eval_kos)
    eval_rows.append(evaluate_panel(fi_hybrid, "hybrid_pan_fi_x_sclc_pcc"))

    print("\n=== Baseline: sclc_pcc_only ===")
    eval_rows.append(evaluate_panel(pcc_sclc.loc[:, eval_kos], "sclc_pcc_only"))

    fi_by_gamma: Dict[str, pd.DataFrame] = {}
    for gamma in JST_GAMMA_SWEEP:
        tag = jst_method_tag(gamma)
        print(f"\n=== Primary: JST-SLRFM ({tag}) ===")
        fi_jst = compute_jst_slrfm(
            pan_embedding,
            pan_effects,
            device,
            p_pan_values,
            omega,
            gamma=gamma,
            ko_columns=eval_kos,
        )
        fi_by_gamma[tag] = fi_jst
        eval_rows.append(evaluate_panel(fi_jst, tag))

    eval_df = pd.concat(eval_rows, ignore_index=True)
    eval_csv = os.path.join(RESULTS_DIR, "panel_rank_comparison.csv")
    eval_df.to_csv(eval_csv, index=False)

    rank_summary = []
    for method in eval_df["method"].unique():
        rank_summary.append(
            {"method": method, "mean_panel_rank": mean_panel_rank(eval_df, method)}
        )
    rank_summary_df = pd.DataFrame(rank_summary).sort_values("mean_panel_rank")
    rank_summary_df.to_csv(os.path.join(RESULTS_DIR, "mean_panel_rank.csv"), index=False)

    best_gamma_tag = rank_summary_df.iloc[0]["method"]
    if best_gamma_tag.startswith("jst_"):
        print(f"\nBest JST gamma by mean panel rank: {best_gamma_tag}")
    else:
        jst_only = rank_summary_df[rank_summary_df["method"].str.startswith("jst_")]
        if len(jst_only):
            print(f"\nBest JST gamma by mean panel rank: {jst_only.iloc[0]['method']}")
        else:
            print("\nNo JST config in rank summary.")

    best_per_panel = (
        eval_df.dropna(subset=["expected_partner_rank"])
        .sort_values(["panel", "expected_partner_rank", "expected_partner_score"])
        .groupby("panel", as_index=False)
        .first()
    )
    best_csv = os.path.join(RESULTS_DIR, "best_method_per_panel.csv")
    best_per_panel.to_csv(best_csv, index=False)

    default_tag = jst_method_tag(DEFAULT_JST_GAMMA)
    fi_default = fi_by_gamma[default_tag] if default_tag in fi_by_gamma else list(fi_by_gamma.values())[-1]
    build_top_pair_per_ko(fi_default).to_csv(
        os.path.join(RESULTS_DIR, "top_pairs_jst_default.csv"), index=False
    )
    fi_default.to_pickle(os.path.join(RESULTS_DIR, f"fi_{default_tag}.pkl"))

    xlsx = os.path.join(RESULTS_DIR, "sli_jst_summary.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        eval_df.to_excel(writer, sheet_name="PanelRankComparison", index=False)
        best_per_panel.to_excel(writer, sheet_name="BestMethodPerPanel", index=False)
        rank_summary_df.to_excel(writer, sheet_name="MeanPanelRank", index=False)
        build_top_pair_per_ko(fi_default).head(200).to_excel(
            writer, sheet_name="TopPairsJST", index=False
        )

    print("\nDone.")
    print(f"  {eval_csv}")
    print(f"  {best_csv}")
    print(f"  {xlsx}")
    print(f"\nPrimary method: JST-SLRFM (joint KRR + KMM + pan-prior)")
    print(f"  Default gamma={DEFAULT_JST_GAMMA}; sweep={JST_GAMMA_SWEEP}")
    print("  Set JST_COMPUTE_ALL_KOS=True for genome-wide FI.")


if __name__ == "__main__":
    main()
