"""
SCLC SLI discovery via pan-cancer transfer SL-RFM.

Primary method (per knockout B, all DepMap lines once):
  1. Pan source:  unweighted KRR with P=ones  -> alpha_pan
  2. Transfer:    tiered row weights (SCLC >> lung NSCLC >> other),
                  kernel K with P_pan, pan-prior ridge
                  (K_w + (REG+gamma)I) alpha = y_w + gamma * alpha_pan
  3. Score:       AGOP(transfer alpha) x PCC on SCLC + lung NSCLC rows

Baselines: precomputed pan FI, hybrid (pan FI x SCLC PCC), SCLC PCC only.

Does NOT stack duplicate SCLC rows; does NOT use joint multi-KO weighted KRR
or global mean_abs embedding warps.
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

RESULTS_DIR = "sli_transfer_results"

# SL-RFM / kernel settings (match original repo)
BANDWIDTH = 1.0
REG = 1e-5
L_GRAD = 1.0
P_CLIP_LOW = 0.5
P_CLIP_HIGH = 2.0

# Tiered transfer weights (other DepMap lines = 1.0)
WEIGHT_OTHER = 1.0
WEIGHT_LUNG_NSCLC = 10.0
WEIGHT_SCLC = 100.0

# Pan-prior strength; sweep on panel, then pick one config for discovery
TRANSFER_HPARAM_SWEEP: List[Dict[str, float]] = [
    {"w_sclc": 50.0, "w_lung": 10.0, "gamma": 0.1},
    {"w_sclc": 100.0, "w_lung": 10.0, "gamma": 1.0},
    {"w_sclc": 100.0, "w_lung": 10.0, "gamma": 10.0},
]
DEFAULT_TRANSFER_CONFIG = TRANSFER_HPARAM_SWEEP[1]

# False = panel KOs only (fast eval); True = full genome-wide FI matrix
TRANSFER_COMPUTE_ALL_KOS = False

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


def safe_sheet_name(name: str) -> str:
    return re.sub(r"[:\\/?*\[\]]", "_", name)[:31]


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
        diff = np.nanmax(np.abs(pan_embedding.loc[common].values - sclc_embedding.loc[common].values))
        print(f"Warning: SCLC .hkl rows differ from pan (max diff={diff:.6g}); using pan for training.")
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
) -> Tuple[pd.Index, pd.Index, pd.Index]:
    sclc_in_pan = pd.Index([i for i in pan_index if i in sclc_ids])
    lung_in_pan = pd.Index([i for i in pan_index if i in lung_nsclc_ids])
    target_pcc_ids = sclc_in_pan.union(lung_in_pan)
    return sclc_in_pan, lung_in_pan, target_pcc_ids


def build_tiered_sample_weights(
    pan_index: pd.Index,
    sclc_ids_in_pan: pd.Index,
    lung_nsclc_ids_in_pan: pd.Index,
    w_sclc: float,
    w_lung: float,
    w_other: float = WEIGHT_OTHER,
) -> np.ndarray:
    weights = np.full(len(pan_index), w_other, dtype=float)
    weights[pan_index.isin(lung_nsclc_ids_in_pan)] = w_lung
    weights[pan_index.isin(sclc_ids_in_pan)] = w_sclc
    n_s = int(pan_index.isin(sclc_ids_in_pan).sum())
    n_l = int(pan_index.isin(lung_nsclc_ids_in_pan).sum())
    print(
        f"  Tiered weights ({len(pan_index)} lines): "
        f"SCLC={n_s}@{w_sclc}, lung_NSCLC={n_l}@{w_lung}, "
        f"other={len(pan_index) - n_s - n_l}@{w_other}"
    )
    return weights


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
# Kernel / SL-RFM core
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


def train_krr_column(
    X_t: torch.Tensor,
    y: np.ndarray,
    bandwidth: float,
    reg: float,
    device: torch.device,
    sample_weights: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None,
    K_prebuilt: Optional[torch.Tensor] = None,
    alpha_pan_prior: Optional[np.ndarray] = None,
    gamma: float = 0.0,
) -> torch.Tensor:
    """
    Dual coefficients alpha for one KO column.

    Unweighted pan source: gamma=0, sample_weights=None, P=None.
    Transfer with pan prior: minimize ||sqrt(W)(y-Ka)||^2 + gamma||a - a_pan||^2
    solved as (K_w + (REG+gamma)I) a = y_w + gamma * a_pan.
    """
    y_t = torch.tensor(y.reshape(-1, 1), device=device).float()
    K = K_prebuilt if K_prebuilt is not None else build_kernel_matrix(
        X_t, bandwidth, device, sample_weights, P
    )
    if sample_weights is not None:
        scale = torch.tensor(np.sqrt(sample_weights), device=device).float()
        y_t = scale.unsqueeze(1) * y_t

    n = K.shape[0]
    I = torch.eye(n, device=device)
    if alpha_pan_prior is not None and gamma > 0:
        a_p = torch.tensor(alpha_pan_prior.reshape(-1, 1), device=device).float()
        sol = torch.linalg.solve(K + (reg + gamma) * I, y_t + gamma * a_p)
    else:
        sol = torch.linalg.solve(K + reg * I, y_t)
    return sol.squeeze(1)


def laplace_kernel(
    samples: torch.Tensor,
    centers: torch.Tensor,
    bandwidth: float,
    M: Optional[torch.Tensor] = None,
    diag_only: bool = False,
) -> torch.Tensor:
    kernel_mat = euclidean_distances_torch(samples, centers, M=M, squared=False, diag_only=diag_only)
    kernel_mat.clamp_(min=0)
    kernel_mat.mul_(-1.0 / bandwidth)
    kernel_mat.exp_()
    return kernel_mat


def get_grads(
    X: torch.Tensor,
    sol_T: torch.Tensor,
    P: torch.Tensor,
    L: float = 1.0,
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


def get_pcc(cell_embedding: pd.DataFrame, gene_effects_df: pd.DataFrame) -> pd.DataFrame:
    exp_cols = [c for c in cell_embedding.columns if c.split("_")[-1] == "exp"]
    std_val = cell_embedding[exp_cols].std(axis=0).replace(0, 1)
    z = (cell_embedding[exp_cols] - cell_embedding[exp_cols].mean(axis=0)) / std_val
    emb = cell_embedding.copy()
    emb[exp_cols] *= (np.abs(z) < 3).fillna(0).astype(int)
    norm_emb = emb - emb.mean(axis=0)
    norm_ge = gene_effects_df - gene_effects_df.mean(axis=0)
    cell_norms = (norm_emb**2).sum(axis=0).values
    gene_norms = (norm_ge**2).sum(axis=0).values
    pcc = (norm_emb.T @ norm_ge) / (cell_norms.reshape(-1, 1) @ gene_norms.reshape((1, -1))) ** 0.5
    return pd.DataFrame(pcc, index=norm_emb.columns, columns=norm_ge.columns)


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


def load_or_compute_p_pan(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    device: torch.device,
    results_dir: str = RESULTS_DIR,
) -> pd.Series:
    csv_path = os.path.join(results_dir, "P_pan.csv")
    if os.path.exists(csv_path):
        print(f"Loading cached P_pan from {csv_path}")
        p_pan = pd.read_csv(csv_path, index_col=0).squeeze(axis=1)
        return p_pan.reindex(pan_embedding.columns).fillna(1.0)

    print("Computing P_pan from pan AGOP...")
    X_t = torch.tensor(pan_embedding.values, device=device).float()
    K = build_kernel_matrix(X_t, BANDWIDTH, device, None, None)
    Y = torch.tensor(pan_effects.values, device=device).float()
    sol = torch.linalg.solve(K + REG * torch.eye(K.shape[0], device=device), Y)
    P_ones = torch.ones(X_t.shape[1], device=device).double()
    grads = get_grads(X_t, sol.T, P=P_ones)
    p_pan = stabilize_metric_p(
        pd.Series(grads.detach().cpu().numpy().mean(axis=1), index=pan_embedding.columns)
    )
    os.makedirs(results_dir, exist_ok=True)
    p_pan.to_csv(csv_path)
    np.save(os.path.join(results_dir, "P_pan.npy"), p_pan.values)
    return p_pan


def load_or_compute_pan_fi(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    prefix = FEATURE_IMPORTANCE_PAN_PREFIX
    if all(os.path.exists(prefix + s) for s in ("_data.npy", "_index.npy", "_columns.npy")):
        print(f"Loading precomputed pan FI from {prefix}")
        fi = load_feature_importance(prefix)
        return fi.reindex(index=pan_embedding.columns, columns=pan_effects.columns).fillna(0)
    print("Training pan FI (all KOs at once)...")
    return _compute_fi_batch(
        pan_embedding, pan_effects, device, None, None, pan_embedding, pan_effects
    )


# ---------------------------------------------------------------------------
# Transfer SL-RFM (primary)
# ---------------------------------------------------------------------------
def transfer_slrfm_column(
    X_t: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
    p_pan_values: np.ndarray,
    sample_weights: np.ndarray,
    feature_index: pd.Index,
    pcc_embedding: pd.DataFrame,
    pcc_effects_ko: pd.DataFrame,
    gamma: float,
    K_pan: torch.Tensor,
    K_transfer: torch.Tensor,
) -> pd.Series:
    alpha_pan = train_krr_column(
        X_t, y, BANDWIDTH, REG, device,
        sample_weights=None, P=None, K_prebuilt=K_pan,
        alpha_pan_prior=None, gamma=0.0,
    )
    sol = train_krr_column(
        X_t, y, BANDWIDTH, REG, device,
        sample_weights=sample_weights, P=p_pan_values, K_prebuilt=K_transfer,
        alpha_pan_prior=alpha_pan.detach().cpu().numpy(), gamma=gamma,
    )
    P_t = torch.tensor(p_pan_values, device=device).double()
    grads = get_grads(X_t, sol.unsqueeze(0), P=P_t).detach().cpu().numpy().ravel()
    pcc = process_pcc_for_sli(get_pcc(pcc_embedding, pcc_effects_ko))
    pcc_vec = pcc.iloc[:, 0].reindex(feature_index).fillna(0.0).values
    return pd.Series(grads * pcc_vec, index=feature_index)


def compute_transfer_slrfm(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    device: torch.device,
    p_pan_values: np.ndarray,
    sample_weights: np.ndarray,
    pcc_embedding: pd.DataFrame,
    pcc_effects: pd.DataFrame,
    gamma: float,
    ko_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    if ko_columns is None:
        ko_columns = list(pan_effects.columns)

    X_t = torch.tensor(pan_embedding.values, device=device).float()
    K_pan = build_kernel_matrix(X_t, BANDWIDTH, device, None, None)
    K_transfer = build_kernel_matrix(X_t, BANDWIDTH, device, sample_weights, p_pan_values)

    out = {}
    for ko in tqdm(ko_columns, desc="Transfer SL-RFM (per-KO)"):
        if ko not in pan_effects.columns:
            continue
        out[ko] = transfer_slrfm_column(
            X_t,
            pan_effects[ko].values,
            device,
            p_pan_values,
            sample_weights,
            pan_embedding.columns,
            pcc_embedding,
            pcc_effects.loc[:, [ko]],
            gamma,
            K_pan,
            K_transfer,
        )
    return pd.DataFrame(out)


def _compute_fi_batch(
    embedding: pd.DataFrame,
    effects: pd.DataFrame,
    device: torch.device,
    sample_weights: Optional[np.ndarray],
    P: Optional[np.ndarray],
    pcc_embedding: pd.DataFrame,
    pcc_effects: pd.DataFrame,
) -> pd.DataFrame:
    X_t = torch.tensor(embedding.values, device=device).float()
    K = build_kernel_matrix(X_t, BANDWIDTH, device, sample_weights, P)
    Y = torch.tensor(effects.values, device=device).float()
    if sample_weights is not None:
        scale = torch.tensor(np.sqrt(sample_weights), device=device).float()
        Y = scale.unsqueeze(1) * Y
    sol = torch.linalg.solve(K + REG * torch.eye(K.shape[0], device=device), Y)
    P_t = torch.ones(X_t.shape[1], device=device).double() if P is None else torch.tensor(P, device=device).double()
    grads = get_grads(X_t, sol.T, P=P_t)
    grads_df = pd.DataFrame(grads.cpu().numpy(), index=embedding.columns, columns=effects.columns)
    pcc = process_pcc_for_sli(get_pcc(pcc_embedding, pcc_effects))
    pcc = pcc.reindex(index=grads_df.index, columns=grads_df.columns).fillna(0)
    return grads_df * pcc


def transfer_config_tag(cfg: Dict[str, float]) -> str:
    return f"transfer_wS{cfg['w_sclc']}_wL{cfg['w_lung']}_g{cfg['gamma']}"


# ---------------------------------------------------------------------------
# Baselines & evaluation
# ---------------------------------------------------------------------------
def hybrid_score_for_ko(fi_pan: pd.DataFrame, pcc_sclc: pd.DataFrame, ko_gene: str) -> pd.Series:
    if ko_gene not in fi_pan.columns or ko_gene not in pcc_sclc.columns:
        return pd.Series(dtype=float)
    pan_col = fi_pan[ko_gene].abs()
    sclc_col = pcc_sclc[ko_gene]
    idx = pan_col.index.intersection(sclc_col.index)
    return pan_col.loc[idx] * sclc_col.loc[idx].abs()


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


def evaluate_panel(scores: pd.DataFrame, method_name: str) -> pd.DataFrame:
    rows = []
    for title, ko, partner in KNOWN_SLI_PANELS:
        col = scores[ko] if ko in scores.columns else pd.Series(dtype=float)
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
    sclc_meta, lung_nsclc_meta = load_depmap_lineage_ids()
    sclc_in_pan, lung_in_pan, target_pcc_ids = intersect_lineage_with_pan(
        pan_embedding.index, sclc_meta, lung_nsclc_meta
    )
    print_dataset_diagnostics(pan_embedding, pan_effects, sclc_in_pan, lung_in_pan, device)

    fi_pan = load_or_compute_pan_fi(pan_embedding, pan_effects, device)
    p_pan_values = load_or_compute_p_pan(pan_embedding, pan_effects, device).values

    pcc_emb_target = pan_embedding.loc[target_pcc_ids]
    pcc_eff_target = pan_effects.loc[target_pcc_ids]
    pcc_sclc = process_pcc_for_sli(get_pcc(sclc_embedding, sclc_effects))

    panel_kos = sorted({ko for _, ko, _ in KNOWN_SLI_PANELS})
    eval_kos = list(pan_effects.columns) if TRANSFER_COMPUTE_ALL_KOS else panel_kos

    eval_rows = []
    eval_rows.append(evaluate_panel(fi_pan, "pan_fi"))
    eval_rows.append(
        evaluate_panel(
            pd.DataFrame(
                {ko: hybrid_score_for_ko(fi_pan, pcc_sclc, ko) for ko in panel_kos}
            ),
            "hybrid_pan_fi_x_sclc_pcc",
        )
    )
    eval_rows.append(evaluate_panel(pcc_sclc, "sclc_pcc_only"))

    fi_by_config: Dict[str, pd.DataFrame] = {}
    for cfg in TRANSFER_HPARAM_SWEEP:
        tag = transfer_config_tag(cfg)
        print(f"\n=== Primary: {tag} ===")
        weights = build_tiered_sample_weights(
            pan_embedding.index, sclc_in_pan, lung_in_pan,
            w_sclc=cfg["w_sclc"], w_lung=cfg["w_lung"],
        )
        fi = compute_transfer_slrfm(
            pan_embedding,
            pan_effects,
            device,
            p_pan_values,
            weights,
            pcc_emb_target,
            pcc_eff_target,
            gamma=cfg["gamma"],
            ko_columns=eval_kos,
        )
        fi_by_config[tag] = fi
        eval_rows.append(evaluate_panel(fi, tag))
        if cfg == DEFAULT_TRANSFER_CONFIG:
            build_top_pair_per_ko(fi).to_csv(
                os.path.join(RESULTS_DIR, "top_pairs_transfer.csv"), index=False
            )

    eval_df = pd.concat(eval_rows, ignore_index=True)
    eval_csv = os.path.join(RESULTS_DIR, "panel_rank_comparison.csv")
    eval_df.to_csv(eval_csv, index=False)

    best_per_panel = (
        eval_df.dropna(subset=["expected_partner_rank"])
        .sort_values(["panel", "expected_partner_rank", "expected_partner_score"])
        .groupby("panel", as_index=False)
        .first()
    )
    best_csv = os.path.join(RESULTS_DIR, "best_method_per_panel.csv")
    best_per_panel.to_csv(best_csv, index=False)

    default_tag = transfer_config_tag(DEFAULT_TRANSFER_CONFIG)
    fi_default = fi_by_config[default_tag]
    xlsx = os.path.join(RESULTS_DIR, "sli_transfer_summary.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        eval_df.to_excel(writer, sheet_name="PanelRankComparison", index=False)
        best_per_panel.to_excel(writer, sheet_name="BestMethodPerPanel", index=False)
        build_top_pair_per_ko(fi_default).head(200).to_excel(writer, sheet_name="TopPairsTransfer", index=False)

    fi_default.to_pickle(os.path.join(RESULTS_DIR, "fi_transfer_default.pkl"))
    fi_pan.to_pickle(os.path.join(RESULTS_DIR, "fi_pan.pkl"))

    print("\nDone.")
    print(f"  {eval_csv}")
    print(f"  {best_csv}")
    print(f"  {xlsx}")
    print("\nPrimary method: transfer_* (per-KO, tiered weights, P_pan, pan-prior)")
    print("Set TRANSFER_COMPUTE_ALL_KOS=True for genome-wide top_pairs_transfer.csv")


if __name__ == "__main__":
    main()
