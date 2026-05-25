"""
SLI discovery pipeline for SCLC with proper pan-cancer transfer.

Implements:
  Phase A — Pan SL-RFM (load precomputed FI or train on pan cells)
  Phase B — Weighted KRR on all DepMap lines (SCLC rows upweighted, P_pan metric)
  Phase C — Hybrid ranking: pan per-KO FI × SCLC per-KO PCC
  Phase D — Panel re-score with same P_pan as Phase B (single-KO retrain for speed)

SCLC lines are already included in the pan embedding; Phase B upweights those rows
rather than stacking duplicate copies of SCLC cell lines.

Removed (broken transfer patterns):
  - global mean_abs feature weights
  - beta sweep on blended global weights
  - column scaling + row L2 re-normalization warp
  - SCLC-only KRR as the primary discovery model
  - P=ones in AGOP while embedding was manually warped
  - concat pan + SCLC subsets (duplicate kernel rows)
"""

import os
import re
from typing import Optional, Tuple

import pandas as pd

import hickle as hkl
import numpy as np
import torch
from numpy.linalg import norm
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (adjust if your layout differs)
# ---------------------------------------------------------------------------
PAN_EMBED_HKL = "embeddings/final_X_tcga_processed.hkl"
PAN_GENE_EFFECT_HKL = "datasets/2023/CRISPRGeneEffect_processed.hkl"
SCLC_EMBED_HKL = "embeddings/final_X_sclc_processed.hkl"
SCLC_GENE_EFFECT_HKL = "datasets/2023/CRISPRGeneEffect_sclc_processed.hkl"
FEATURE_IMPORTANCE_PAN_PREFIX = "datasets/feature_importances_pan"

RESULTS_DIR = "sli_pooled_results"

# Kernel / RFM settings (match original SL-RFM repo)
BANDWIDTH = 1.0
REG = 1e-5
L_GRAD = 1.0

# Pooled training: upweight SCLC rows relative to pan rows
SCLC_SAMPLE_WEIGHT = 10.0
SCLC_WEIGHT_SWEEP = [5.0, 10.0, 20.0]

# Stabilize pan AGOP metric P
P_CLIP_LOW = 0.5
P_CLIP_HIGH = 2.0

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
# I/O helpers
# ---------------------------------------------------------------------------
def as_dataframe(obj: object, name: str = "data") -> pd.DataFrame:
    """Coerce hickle / array loads into a DataFrame."""
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
    name = re.sub(r"[:\\/?*\[\]]", "_", name)
    return name[:31]


def align_embeddings_and_effects(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    sclc_embedding: pd.DataFrame,
    sclc_effects: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Align row indices and shared feature/KO columns across pan and SCLC."""
    pan_rows = pan_embedding.index.intersection(pan_effects.index)
    sclc_rows = sclc_embedding.index.intersection(sclc_effects.index)
    if len(pan_rows) == 0:
        raise ValueError("No overlapping cell IDs between pan embedding and pan gene effects.")
    if len(sclc_rows) == 0:
        raise ValueError("No overlapping cell IDs between SCLC embedding and SCLC gene effects.")

    if len(pan_rows) != len(pan_embedding) or len(pan_rows) != len(pan_effects):
        print(
            f"Warning: aligning pan rows "
            f"({len(pan_embedding)} emb, {len(pan_effects)} eff) -> {len(pan_rows)}"
        )
    if len(sclc_rows) != len(sclc_embedding) or len(sclc_rows) != len(sclc_effects):
        print(
            f"Warning: aligning SCLC rows "
            f"({len(sclc_embedding)} emb, {len(sclc_effects)} eff) -> {len(sclc_rows)}"
        )

    common_cols = pan_embedding.columns.intersection(sclc_embedding.columns)
    common_kos = pan_effects.columns.intersection(sclc_effects.columns)
    if len(common_cols) == 0 or len(common_kos) == 0:
        raise ValueError("No overlapping features or KO genes between pan and SCLC matrices.")

    pan_embedding = pan_embedding.loc[pan_rows, common_cols]
    pan_effects = pan_effects.loc[pan_rows, common_kos]
    sclc_embedding = sclc_embedding.loc[sclc_rows, common_cols]
    sclc_effects = sclc_effects.loc[sclc_rows, common_kos]

    return pan_embedding, pan_effects, sclc_embedding, sclc_effects


def validate_sclc_in_pan(
    pan_embedding: pd.DataFrame,
    sclc_embedding: pd.DataFrame,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> pd.Index:
    """
    Ensure SCLC cell IDs exist in the pan matrix and optionally check row equality.
    Returns SCLC IDs present in pan (canonical index for upweighting / PCC).
    """
    sclc_ids = sclc_embedding.index
    missing = sclc_ids.difference(pan_embedding.index)
    if len(missing) > 0:
        sample = ", ".join(map(str, missing[:5]))
        raise ValueError(
            f"{len(missing)} SCLC cell ID(s) not found in pan embedding "
            f"(e.g. {sample}). SCLC must be a subset of pan DepMap lines."
        )

    common = sclc_ids.intersection(pan_embedding.index)
    pan_sub = pan_embedding.loc[common]
    sclc_sub = sclc_embedding.loc[common]
    if not np.allclose(pan_sub.values, sclc_sub.values, rtol=rtol, atol=atol, equal_nan=True):
        max_diff = float(np.nanmax(np.abs(pan_sub.values - sclc_sub.values)))
        print(
            f"Warning: SCLC embedding differs from pan for {len(common)} shared IDs "
            f"(max abs diff={max_diff:.6g}). Using pan rows for Phase B training."
        )
    return common


def print_dataset_diagnostics(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    sclc_ids_in_pan: pd.Index,
    device: torch.device,
) -> None:
    n_pan = len(pan_embedding)
    n_sclc = len(sclc_ids_in_pan)
    n_features = pan_embedding.shape[1]
    n_kos = pan_effects.shape[1]
    n_non_sclc = n_pan - n_sclc
    kernel_mb = (n_pan**2) * 4 / (1024**2)

    print("\n--- Dataset diagnostics ---")
    print(f"  Device:              {device}")
    print(f"  Pan cell lines:      {n_pan}")
    print(f"  SCLC in pan:         {n_sclc}")
    print(f"  Non-SCLC lines:      {n_non_sclc}")
    print(f"  Features:            {n_features}")
    print(f"  KO genes:            {n_kos}")
    print(f"  Approx K matrix RAM: {kernel_mb:.0f} MB (float32, n×n)")
    if n_sclc == 0:
        print("  WARNING: no SCLC lines matched — Phase B upweighting will have no effect.")
    print("----------------------------\n")


def ensure_row_l2_normalized(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    norms = norm(out.values, axis=1).reshape(-1, 1)
    norms[norms == 0] = 1.0
    out.loc[:, :] = out.values / norms
    return out


# ---------------------------------------------------------------------------
# Core SL-RFM math (P-aware KRR + AGOP + PCC)
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
    gamma = 1.0 / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat


def train_krr(
    cell_embedding_df: pd.DataFrame,
    gene_effects_df: pd.DataFrame,
    bandwidth: float,
    reg: float,
    device: torch.device,
    sample_weights: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted kernel ridge regression with optional diagonal metric P.

    Sample weights enter as sqrt(w_i) scaling of K rows/cols and Y rows
    (equivalent to weighted least squares).
    """
    X_t = torch.tensor(cell_embedding_df.values, device=device).float()
    Y_t = torch.tensor(gene_effects_df.values, device=device).float()
    n = X_t.shape[0]

    P_t = None
    if P is not None:
        P_t = torch.tensor(P, device=device).double()

    dist = euclidean_distances_torch(X_t, X_t, M=P_t, squared=False, diag_only=True)
    dist.fill_diagonal_(0)
    K = torch.exp(-bandwidth * dist)

    if sample_weights is not None:
        sw = torch.tensor(sample_weights, device=device).float()
        scale = torch.sqrt(sw)
        K = scale.unsqueeze(1) * K * scale.unsqueeze(0)
        Y_t = scale.unsqueeze(1) * Y_t

    sol = torch.linalg.solve(K + reg * torch.eye(n, device=device), Y_t)
    return sol, X_t


def get_grads(
    X: torch.Tensor,
    sol_T: torch.Tensor,
    P: torch.Tensor,
    L: float = 1.0,
    diag_only: bool = True,
    bandwidth: float = BANDWIDTH,
) -> torch.Tensor:
    """AGOP diagonal: average squared gradient magnitudes per feature."""
    K = laplace_kernel(X, X, bandwidth=bandwidth, M=P, diag_only=diag_only)

    dist = euclidean_distances_torch(X, X, M=P, squared=False, diag_only=diag_only)
    dist.clamp_(min=0)
    dist[dist < 1e-10] = 0

    K = K / torch.where(dist == 0, torch.ones_like(dist), dist)
    K[torch.isinf(K)] = 0.0

    n, d = X.shape
    num_kos, n2 = sol_T.shape
    assert n == n2

    grads = torch.zeros((d, num_kos), device=X.device)
    for i in tqdm(range(num_kos), desc="AGOP grads"):
        weight = sol_T[i, :].reshape((-1, 1))
        step2 = K @ (weight * X)
        step3 = (weight.T @ K).T * X
        G = (step2 - step3) * (-1.0 / L)
        G = torch.sum(G**2, axis=0)
        grads[:, i] = G / n
    return grads


def get_pcc(
    cell_embedding: pd.DataFrame,
    gene_effects_df: pd.DataFrame,
) -> pd.DataFrame:
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
    ) ** 0.5

    return pd.DataFrame(
        pcc,
        columns=normalized_gene_effects_df.columns,
        index=normalized_cell_embedding.columns,
    )


def process_pcc_for_sli(pcc: pd.DataFrame) -> pd.DataFrame:
    """Direction filter: mutations negative-only; expression uses |PCC|."""
    out = pcc.copy().fillna(0)
    mut = [x for x in out.index if x.split("_")[-1] != "exp"]
    exp = [x for x in out.index if x.split("_")[-1] == "exp"]
    out.loc[mut] = -(out.loc[mut].clip(upper=0))
    out.loc[exp] = out.loc[exp].abs()
    return out


def stabilize_metric_p(p: pd.Series) -> pd.Series:
    p = p.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mean = p.mean()
    if mean > 0:
        p = p / mean
    return p.clip(lower=P_CLIP_LOW, upper=P_CLIP_HIGH)


def compute_feature_importance_df(
    cell_embedding_df: pd.DataFrame,
    gene_effects_df: pd.DataFrame,
    device: torch.device,
    sample_weights: Optional[np.ndarray] = None,
    P: Optional[np.ndarray] = None,
    pcc_embedding_df: Optional[pd.DataFrame] = None,
    pcc_gene_effects_df: Optional[pd.DataFrame] = None,
    ko_columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Full SL-RFM feature importance: weighted KRR + AGOP (with P) × PCC.

    pcc_* defaults to the training matrices; pass SCLC subset for weighted training.
    """
    if ko_columns is not None:
        gene_effects_df = gene_effects_df.loc[:, ko_columns]

    sol, X_t = train_krr(
        cell_embedding_df,
        gene_effects_df,
        BANDWIDTH,
        REG,
        device,
        sample_weights=sample_weights,
        P=P,
    )

    d = X_t.shape[1]
    if P is not None:
        P_t = torch.tensor(P, device=device).double()
    else:
        P_t = torch.ones(d, device=device).double()

    grads_t = get_grads(X_t, sol.T, P=P_t, L=L_GRAD, diag_only=True)
    grads_df = pd.DataFrame(
        grads_t.detach().cpu().numpy(),
        index=cell_embedding_df.columns,
        columns=gene_effects_df.columns,
    )

    emb_pcc = pcc_embedding_df if pcc_embedding_df is not None else cell_embedding_df
    ge_pcc = pcc_gene_effects_df if pcc_gene_effects_df is not None else gene_effects_df
    pcc = process_pcc_for_sli(get_pcc(emb_pcc, ge_pcc))
    pcc = pcc.loc[grads_df.index, grads_df.columns]

    return grads_df * pcc


def load_or_compute_p_pan(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    device: torch.device,
    results_dir: str = RESULTS_DIR,
) -> pd.Series:
    """Load cached P_pan or compute from pan KRR + AGOP."""
    csv_path = os.path.join(results_dir, "P_pan.csv")
    if os.path.exists(csv_path):
        print(f"Loading cached P_pan from {csv_path}")
        p_pan = pd.read_csv(csv_path, index_col=0).squeeze(axis=1)
        p_pan.index = p_pan.index.astype(str)
        missing = pan_embedding.columns.difference(p_pan.index)
        if len(missing) > 0:
            print(f"  Warning: P_pan missing {len(missing)} features; filling with 1.0")
            p_pan = p_pan.reindex(pan_embedding.columns).fillna(1.0)
        else:
            p_pan = p_pan.reindex(pan_embedding.columns)
        return p_pan

    p_pan = compute_p_pan_from_pan(pan_embedding, pan_effects, device)
    os.makedirs(results_dir, exist_ok=True)
    p_pan.to_csv(csv_path)
    np.save(os.path.join(results_dir, "P_pan.npy"), p_pan.values)
    return p_pan


def compute_p_pan_from_pan(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    device: torch.device,
) -> pd.Series:
    """
    Phase A metric transfer: pan KRR + AGOP with P=ones → global diagonal P.
    Per-KO AGOP diagonals are averaged (NOT mean_abs across FI scores).
    """
    print("Computing P_pan from pan-cancer AGOP...")
    sol, X_t = train_krr(pan_embedding, pan_effects, BANDWIDTH, REG, device)
    P_ones = torch.ones(X_t.shape[1], device=device).double()
    grads_t = get_grads(X_t, sol.T, P=P_ones, L=L_GRAD, diag_only=True)
    p_pan = pd.Series(
        grads_t.detach().cpu().numpy().mean(axis=1),
        index=pan_embedding.columns,
    )
    return stabilize_metric_p(p_pan)


# ---------------------------------------------------------------------------
# Phase B: weighted training data (no duplicate stacking)
# ---------------------------------------------------------------------------
def build_weighted_dataset(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    sclc_ids_in_pan: pd.Index,
    sclc_weight: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Use all DepMap lines once. Upweight rows whose ModelID is SCLC.

    Returns (embedding, effects, sample_weights) with one row per cell line.
    """
    weights = np.ones(len(pan_embedding), dtype=float)
    is_sclc = pan_embedding.index.isin(sclc_ids_in_pan)
    weights[is_sclc] = sclc_weight
    n_sclc = int(is_sclc.sum())
    print(
        f"  Weighted dataset: {len(pan_embedding)} lines "
        f"({n_sclc} SCLC @ weight={sclc_weight}, "
        f"{len(pan_embedding) - n_sclc} other @ weight=1.0)"
    )
    return pan_embedding, pan_effects, weights


def sclc_subset_from_pan(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    sclc_ids_in_pan: pd.Index,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """SCLC rows from the pan matrices (for SCLC-only PCC in Phase B)."""
    return pan_embedding.loc[sclc_ids_in_pan], pan_effects.loc[sclc_ids_in_pan]


# ---------------------------------------------------------------------------
# Phase C: hybrid per-KO scoring (no global mean_abs)
# ---------------------------------------------------------------------------
def hybrid_score_for_ko(
    fi_pan: pd.DataFrame,
    pcc_sclc: pd.DataFrame,
    ko_gene: str,
) -> pd.Series:
    """
    Per-KO pan FI column × SCLC PCC column.
    Uses fi_pan[:, ko] directly — KO-specific, not averaged across all KOs.
    """
    if ko_gene not in fi_pan.columns or ko_gene not in pcc_sclc.columns:
        return pd.Series(dtype=float)
    pan_col = fi_pan[ko_gene].abs()
    sclc_col = pcc_sclc[ko_gene]
    common = pan_col.index.intersection(sclc_col.index)
    return pan_col.loc[common] * sclc_col.loc[common].abs()


# ---------------------------------------------------------------------------
# Ranking / evaluation helpers
# ---------------------------------------------------------------------------
def rank_all_features_for_one_ko(scores: pd.Series, ko_gene: str) -> pd.DataFrame:
    s = scores.sort_values(ascending=False)
    df = s.reset_index()
    df.columns = ["feature", "score"]
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["ko_gene"] = ko_gene
    df["feature_type"] = np.where(
        df["feature"].astype(str).str.endswith("_exp"), "exp", "mut_or_other"
    )
    df["partner_gene"] = (
        df["feature"].astype(str).str.replace(r"_exp$", "", regex=True)
    )
    return df


def get_partner_rank_and_score(
    scores: pd.Series,
    ko_gene: str,
    expected_partner_gene: str,
) -> Tuple[float, float, str]:
    df = rank_all_features_for_one_ko(scores, ko_gene)
    hits = df[df["partner_gene"] == expected_partner_gene]
    if len(hits) == 0:
        return np.nan, np.nan, ""
    best = hits.sort_values("rank", ascending=True).iloc[0]
    return int(best["rank"]), float(best["score"]), str(best["feature"])


def build_top_pair_per_ko(fi: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ko in fi.columns:
        s = fi[ko].sort_values(ascending=False)
        top1_feat = str(s.index[0])
        top1 = float(s.iloc[0])
        top2 = float(s.iloc[1]) if len(s) > 1 else np.nan
        other_mean = float(s.iloc[1:].mean()) if len(s) > 1 else np.nan
        rows.append(
            {
                "ko_gene": ko,
                "top1_feature": top1_feat,
                "top1_partner_gene": re.sub(r"_exp$", "", top1_feat),
                "top1_score": top1,
                "gap_top1_mean_other": top1 - other_mean if len(s) > 1 else np.nan,
                "gap_top1_top2": top1 - top2 if len(s) > 1 else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("top1_score", ascending=False).reset_index(drop=True)


def evaluate_panel(
    score_fn,
    method_name: str,
    extra_kwargs: Optional[dict] = None,
) -> pd.DataFrame:
    extra_kwargs = extra_kwargs or {}
    rows = []
    for title, ko_gene, expected_partner in KNOWN_SLI_PANELS:
        if callable(score_fn):
            scores = score_fn(ko_gene=ko_gene, **extra_kwargs)
        else:
            scores = score_fn[ko_gene] if ko_gene in score_fn.columns else pd.Series(dtype=float)
        rank, score, feat = get_partner_rank_and_score(scores, ko_gene, expected_partner)
        rows.append(
            {
                "method": method_name,
                "panel": title,
                "ko_gene": ko_gene,
                "expected_partner_gene": expected_partner,
                "expected_partner_rank": rank,
                "expected_partner_score": score,
                "matched_feature": feat,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase A: load or compute pan FI
# ---------------------------------------------------------------------------
def load_or_compute_pan_fi(
    pan_embedding: pd.DataFrame,
    pan_effects: pd.DataFrame,
    device: torch.device,
) -> pd.DataFrame:
    prefix = FEATURE_IMPORTANCE_PAN_PREFIX
    if (
        os.path.exists(prefix + "_data.npy")
        and os.path.exists(prefix + "_index.npy")
        and os.path.exists(prefix + "_columns.npy")
    ):
        print(f"Loading precomputed pan FI from {prefix}")
        fi = load_feature_importance(prefix)
        n_before = fi.size
        fi = fi.reindex(index=pan_embedding.columns, columns=pan_effects.columns)
        n_missing = int(fi.isna().sum().sum())
        if n_missing > 0:
            pct = 100.0 * n_missing / n_before if n_before else 0.0
            print(
                f"  Warning: pan FI reindex introduced {n_missing} missing entries "
                f"({pct:.1f}% of loaded matrix); filling with 0."
            )
        return fi.fillna(0)

    print("Precomputed pan FI not found — training Phase A on pan cells...")
    return compute_feature_importance_df(pan_embedding, pan_effects, device)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load data ---
    print("Loading embeddings and gene effects...")
    pan_embedding = ensure_row_l2_normalized(load_hkl_dataframe(PAN_EMBED_HKL, "pan_embedding"))
    pan_effects = load_hkl_dataframe(PAN_GENE_EFFECT_HKL, "pan_effects")
    sclc_embedding = ensure_row_l2_normalized(load_hkl_dataframe(SCLC_EMBED_HKL, "sclc_embedding"))
    sclc_effects = load_hkl_dataframe(SCLC_GENE_EFFECT_HKL, "sclc_effects")

    pan_embedding, pan_effects, sclc_embedding, sclc_effects = align_embeddings_and_effects(
        pan_embedding, pan_effects, sclc_embedding, sclc_effects
    )
    sclc_ids_in_pan = validate_sclc_in_pan(pan_embedding, sclc_embedding)
    print_dataset_diagnostics(pan_embedding, pan_effects, sclc_ids_in_pan, device)

    # --- Phase A: pan FI + P_pan metric ---
    fi_pan = load_or_compute_pan_fi(pan_embedding, pan_effects, device)
    p_pan = load_or_compute_p_pan(pan_embedding, pan_effects, device)
    p_pan_values = p_pan.values

    # SCLC-only PCC (Phase C + PCC filter for Phase B; uses SCLC annotation file)
    pcc_sclc_raw = get_pcc(sclc_embedding, sclc_effects)
    pcc_sclc = process_pcc_for_sli(pcc_sclc_raw)

    all_eval_rows = []

    # --- Baseline: pan FI per-KO (no transfer retrain) ---
    print("\n=== Baseline: pan FI (per-KO columns) ===")
    pan_eval = evaluate_panel(fi_pan, "pan_fi")
    all_eval_rows.append(pan_eval)

    # --- Phase C: hybrid pan FI × SCLC PCC ---
    print("\n=== Phase C: hybrid (pan FI × |SCLC PCC|) per-KO ===")
    hybrid_eval = evaluate_panel(
        lambda ko_gene: hybrid_score_for_ko(fi_pan, pcc_sclc, ko_gene),
        "hybrid_pan_fi_x_sclc_pcc",
    )
    all_eval_rows.append(hybrid_eval)

    # --- SCLC PCC-only baseline ---
    print("\n=== Baseline: SCLC PCC only ===")
    pcc_eval = evaluate_panel(pcc_sclc, "sclc_pcc_only")
    all_eval_rows.append(pcc_eval)

    # --- Phase B: weighted KRR with P_pan, sweep SCLC row weight ---
    fi_weighted_by_weight = {}
    sclc_emb_pcc, sclc_eff_pcc = sclc_subset_from_pan(
        pan_embedding, pan_effects, sclc_ids_in_pan
    )
    for sclc_w in SCLC_WEIGHT_SWEEP:
        print(f"\n=== Phase B: weighted KRR (SCLC weight={sclc_w}, P=P_pan) ===")
        train_emb, train_eff, sample_w = build_weighted_dataset(
            pan_embedding, pan_effects, sclc_ids_in_pan, sclc_w
        )

        fi_weighted = compute_feature_importance_df(
            train_emb,
            train_eff,
            device,
            sample_weights=sample_w,
            P=p_pan_values,
            pcc_embedding_df=sclc_emb_pcc,
            pcc_gene_effects_df=sclc_eff_pcc,
        )
        fi_weighted_by_weight[sclc_w] = fi_weighted

        method = f"weighted_w{sclc_w}_Ppan_pcc_sclc"
        weighted_eval = evaluate_panel(fi_weighted, method)
        all_eval_rows.append(weighted_eval)

        top_pairs = build_top_pair_per_ko(fi_weighted)
        top_pairs.to_csv(
            os.path.join(RESULTS_DIR, f"top_pairs_weighted_w{sclc_w}.csv"),
            index=False,
        )

    # --- Phase D: panel validation — same settings as Phase B (P_pan, weights, SCLC PCC) ---
    # Panel ranks from Phase B weighted_eval are the primary validation; this re-trains
    # one KO at a time on the panel only (faster than full multi-KO pass per check).
    print("\n=== Phase D: panel validation (same P_pan + weights as Phase B, single-KO) ===")
    best_w = SCLC_WEIGHT_SWEEP[len(SCLC_WEIGHT_SWEEP) // 2]
    train_emb, train_eff, sample_w = build_weighted_dataset(
        pan_embedding, pan_effects, sclc_ids_in_pan, best_w
    )

    panel_kos = sorted({ko for _, ko, _ in KNOWN_SLI_PANELS})
    fi_panel = pd.DataFrame(index=train_emb.columns)

    for ko in tqdm(panel_kos, desc="Panel single-KO (P_pan)"):
        if ko not in train_eff.columns:
            continue
        fi_ko = compute_feature_importance_df(
            train_emb,
            train_eff,
            device,
            sample_weights=sample_w,
            P=p_pan_values,
            pcc_embedding_df=sclc_emb_pcc,
            pcc_gene_effects_df=sclc_eff_pcc,
            ko_columns=[ko],
        )
        fi_panel[ko] = fi_ko[ko]

    phase_d_eval = evaluate_panel(fi_panel, f"panel_singleKO_Ppan_weighted_w{best_w}")
    all_eval_rows.append(phase_d_eval)

    # --- Combined evaluation table ---
    eval_df = pd.concat(all_eval_rows, ignore_index=True)
    eval_csv = os.path.join(RESULTS_DIR, "panel_rank_comparison.csv")
    eval_df.to_csv(eval_csv, index=False)

    # Best method per panel entry
    best_per_panel = (
        eval_df.dropna(subset=["expected_partner_rank"])
        .sort_values(["panel", "expected_partner_rank", "expected_partner_score"], ascending=[True, True, False])
        .groupby("panel", as_index=False)
        .first()
    )
    best_csv = os.path.join(RESULTS_DIR, "best_method_per_panel.csv")
    best_per_panel.to_csv(best_csv, index=False)

    # --- Excel workbook: hybrid + default weighted + panel detail sheets ---
    if 10.0 in fi_weighted_by_weight:
        default_weighted = fi_weighted_by_weight[10.0]
    else:
        default_weighted = fi_weighted_by_weight[SCLC_WEIGHT_SWEEP[0]]

    xlsx_path = os.path.join(RESULTS_DIR, "sli_pooled_pipeline_summary.xlsx")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        eval_df.to_excel(writer, sheet_name="PanelRankComparison", index=False)
        best_per_panel.to_excel(writer, sheet_name="BestMethodPerPanel", index=False)
        build_top_pair_per_ko(default_weighted).head(200).to_excel(
            writer, sheet_name="TopPairsWeighted_w10", index=False
        )

        for title, ko_gene, _ in KNOWN_SLI_PANELS:
            sheet = safe_sheet_name(title)
            rows = []
            for method_label, score_series in [
                ("pan_fi", fi_pan[ko_gene] if ko_gene in fi_pan.columns else pd.Series()),
                ("hybrid", hybrid_score_for_ko(fi_pan, pcc_sclc, ko_gene)),
                ("weighted_w10", default_weighted[ko_gene] if ko_gene in default_weighted.columns else pd.Series()),
                ("panel_singleKO", fi_panel[ko_gene] if ko_gene in fi_panel.columns else pd.Series()),
            ]:
                if len(score_series) == 0:
                    continue
                df = rank_all_features_for_one_ko(score_series, ko_gene)
                df.insert(0, "method", method_label)
                rows.append(df)
            if rows:
                pd.concat(rows, ignore_index=True).to_excel(writer, sheet_name=sheet, index=False)

    # Save core artifacts (P_pan may already exist from load_or_compute_p_pan)
    np.save(os.path.join(RESULTS_DIR, "P_pan.npy"), p_pan_values)
    p_pan.to_csv(os.path.join(RESULTS_DIR, "P_pan.csv"))
    fi_pan.to_pickle(os.path.join(RESULTS_DIR, "fi_pan.pkl"))
    default_weighted.to_pickle(os.path.join(RESULTS_DIR, "fi_weighted_w10.pkl"))

    print("\nDone.")
    print(f"  Panel comparison: {eval_csv}")
    print(f"  Best per panel:   {best_csv}")
    print(f"  Excel summary:    {xlsx_path}")
    print("\nMethods compared:")
    print("  pan_fi                     — Phase A, per-KO pan columns (baseline)")
    print("  hybrid_pan_fi_x_sclc_pcc    — Phase C, pan discovery + SCLC direction")
    print("  sclc_pcc_only              — SCLC correlation only (small-n baseline)")
    print("  weighted_w*                — Phase B, all DepMap lines, SCLC rows upweighted")
    print("  panel_singleKO_Ppan        — Phase D, same P/weights/PCC as B, panel KOs only")


if __name__ == "__main__":
    main()
