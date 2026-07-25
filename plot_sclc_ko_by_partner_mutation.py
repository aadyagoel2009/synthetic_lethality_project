"""
Validate selected (KO, partner) pairs in SCLC:

  Groups: partner-gene MUT vs partner-gene WT
  Y-axis: Chronos / CRISPR gene-effect for knocking out KO

Mutation = damaging OR hotspot call > 0.
Restricted to SCLC cell lines present in both CRISPR and mutation matrices.

Pairs:
  KRAS/KRAS, CTNNB1/APC, PIK3CA/PIK3CA, E2F3/RB1, WRN/ARID1A,
  ARID1B/ARID1A, SKP2/RB1, WRN/KMT2D, CREBBP/EP300, SMARCA2/SMARCA4

Run:
  python plot_sclc_ko_by_partner_mutation.py
"""

from __future__ import annotations

import os
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PAIRS: List[Tuple[str, str]] = [
    ("KRAS", "KRAS"),
    ("CTNNB1", "APC"),
    ("PIK3CA", "PIK3CA"),
    ("E2F3", "RB1"),
    ("WRN", "ARID1A"),
    ("ARID1B", "ARID1A"),
    ("SKP2", "RB1"),
    ("WRN", "KMT2D"),
    ("CREBBP", "EP300"),
    ("SMARCA2", "SMARCA4"),
]

DAMAGING_CSV = "datasets/2023/OmicsSomaticMutationsMatrixDamaging.csv"
HOTSPOT_CSV = "datasets/2023/OmicsSomaticMutationsMatrixHotspot.csv"
CRISPR_CSV = "datasets/2023/CRISPRGeneEffect.csv"
MODEL_CSV = "datasets/2023/Model.csv"

RESULTS_DIR = os.path.join("sli_jst_pcc_split_results", "partner_mut_validation")
OUTPUT_PNG = os.path.join(RESULTS_DIR, "sclc_ko_effects_by_partner_mutation.png")
OUTPUT_PDF = os.path.join(RESULTS_DIR, "sclc_ko_effects_by_partner_mutation.pdf")
OUTPUT_STATS_CSV = os.path.join(RESULTS_DIR, "sclc_ko_effects_by_partner_mutation_stats.csv")
OUTPUT_POINTS_CSV = os.path.join(RESULTS_DIR, "sclc_ko_effects_by_partner_mutation_points.csv")


def load_mutation_matrix(
    damaging_path: str = DAMAGING_CSV,
    hotspot_path: str = HOTSPOT_CSV,
) -> pd.DataFrame:
    """Binary mutation matrix: 1 if damaging or hotspot > 0."""

    def _load(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [str(c).split()[0] for c in df.columns]
        if "ModelID" not in df.columns:
            raise ValueError(f"'ModelID' missing in {path}")
        df = df.set_index("ModelID")
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
        df = df.groupby(level=0).max()
        df.index = df.index.astype(str)
        return df

    damaging = _load(damaging_path)
    hotspot = _load(hotspot_path)
    cells = damaging.index.union(hotspot.index)
    genes = damaging.columns.union(hotspot.columns)
    damaging = damaging.reindex(index=cells, columns=genes, fill_value=0)
    hotspot = hotspot.reindex(index=cells, columns=genes, fill_value=0)
    combined = np.maximum(damaging.values, hotspot.values)
    out = pd.DataFrame(combined, index=cells, columns=genes)
    return (out > 0).astype(int)


def load_crispr_effects(path: str = CRISPR_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).split()[0] for c in df.columns]
    id_col = df.columns[0]
    df = df.set_index(id_col)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.groupby(level=0).mean()
    df.index = df.index.astype(str)
    return df


def get_sclc_model_ids(model_path: str = MODEL_CSV) -> Set[str]:
    meta = pd.read_csv(model_path)
    meta[["OncotreeSubtype", "OncotreePrimaryDisease"]] = meta[
        ["OncotreeSubtype", "OncotreePrimaryDisease"]
    ].fillna("")
    is_sclc = (
        meta["OncotreeSubtype"].astype(str).str.strip().str.lower()
        == "small cell lung cancer"
    )
    return set(meta.loc[is_sclc, "ModelID"].astype(str))


def collect_pair_data(
    effects: pd.DataFrame,
    mutations: pd.DataFrame,
    sclc_ids: pd.Index,
    ko_gene: str,
    partner_gene: str,
) -> pd.DataFrame:
    """One row per SCLC line with KO Chronos score and partner mut/WT label."""
    if ko_gene not in effects.columns:
        raise KeyError(f"KO gene {ko_gene} missing from CRISPR effects.")
    if partner_gene not in mutations.columns:
        raise KeyError(f"Partner gene {partner_gene} missing from mutation matrix.")

    cells = sclc_ids.intersection(effects.index).intersection(mutations.index)
    if len(cells) == 0:
        raise ValueError(f"No overlapping SCLC cells for {ko_gene}/{partner_gene}.")

    chronos = effects.loc[cells, ko_gene].astype(float)
    mut = mutations.loc[cells, partner_gene].astype(int)
    valid = chronos.notna()
    chronos = chronos.loc[valid]
    mut = mut.loc[valid]

    return pd.DataFrame(
        {
            "model_id": chronos.index.astype(str),
            "ko_gene": ko_gene,
            "partner_gene": partner_gene,
            "pair": f"{ko_gene} KO | {partner_gene} mut status",
            "group": np.where(mut.values == 1, "Partner MUT", "Partner WT"),
            "partner_mutated": mut.values.astype(int),
            "chronos_ko_effect": chronos.values,
        }
    )


def mannwhitney_onetailed_mut_more_dependent(
    mut_vals: np.ndarray,
    wt_vals: np.ndarray,
) -> float:
    """
    One-sided Mann-Whitney: are MUT Chronos scores lower (more dependent)
    than WT? Lower Chronos = stronger KO effect.
    """
    if len(mut_vals) == 0 or len(wt_vals) == 0:
        return float("nan")
    res = stats.mannwhitneyu(mut_vals, wt_vals, alternative="less")
    return float(res.pvalue)


def plot_pair_boxplot(ax: plt.Axes, pair_df: pd.DataFrame, title: str) -> Dict:
    mut_vals = pair_df.loc[
        pair_df["group"] == "Partner MUT", "chronos_ko_effect"
    ].to_numpy()
    wt_vals = pair_df.loc[
        pair_df["group"] == "Partner WT", "chronos_ko_effect"
    ].to_numpy()

    data = [wt_vals, mut_vals]
    labels = [f"WT\n(n={len(wt_vals)})", f"MUT\n(n={len(mut_vals)})"]

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    colors = ["#9ecae1", "#fc9272"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        x = i + rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x, vals, s=18, color="black", alpha=0.55, zorder=3)

    pval = mannwhitney_onetailed_mut_more_dependent(mut_vals, wt_vals)
    mut_mean = float(np.mean(mut_vals)) if len(mut_vals) else float("nan")
    wt_mean = float(np.mean(wt_vals)) if len(wt_vals) else float("nan")

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Chronos gene effect (KO)")
    ax.set_xlabel("")
    if np.isfinite(pval):
        ax.text(
            0.5,
            0.98,
            f"MWU p={pval:.3g}\n(MUT more dependent)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    return {
        "ko_gene": str(pair_df["ko_gene"].iloc[0]),
        "partner_gene": str(pair_df["partner_gene"].iloc[0]),
        "n_wt": int(len(wt_vals)),
        "n_mut": int(len(mut_vals)),
        "mean_wt": wt_mean,
        "mean_mut": mut_mean,
        "delta_mut_minus_wt": mut_mean - wt_mean,
        "mannwhitney_p_mut_less": pval,
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading CRISPR gene effects...")
    effects = load_crispr_effects()
    sclc_ids = pd.Index(
        [i for i in effects.index if i in get_sclc_model_ids()]
    )
    print(f"  SCLC lines in CRISPR: {len(sclc_ids)}")

    print("Loading damaging/hotspot mutations...")
    mutations = load_mutation_matrix()

    point_frames: List[pd.DataFrame] = []
    stats_rows: List[Dict] = []

    n_pairs = len(PAIRS)
    ncols = 5
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 3.6 * nrows),
        squeeze=False,
    )

    for i, (ko_gene, partner_gene) in enumerate(PAIRS):
        ax = axes[i // ncols][i % ncols]
        title = f"{ko_gene} KO\nby {partner_gene} mutation"
        try:
            pair_df = collect_pair_data(
                effects, mutations, sclc_ids, ko_gene, partner_gene
            )
            point_frames.append(pair_df)
            stats_rows.append(plot_pair_boxplot(ax, pair_df, title))
            print(
                f"  {ko_gene}/{partner_gene}: "
                f"WT={stats_rows[-1]['n_wt']}, MUT={stats_rows[-1]['n_mut']}, "
                f"p={stats_rows[-1]['mannwhitney_p_mut_less']:.3g}"
            )
        except Exception as exc:
            ax.set_title(title, fontsize=10)
            ax.text(
                0.5,
                0.5,
                f"Skipped\n{exc}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            print(f"  {ko_gene}/{partner_gene}: SKIPPED ({exc})")

    for j in range(n_pairs, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        "SCLC: KO Chronos effect in partner-MUT vs partner-WT lines",
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        OUTPUT_PNG,
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Ensure RGB PNG (some viewers choke on RGBA) and write a JPG fallback.
    try:
        from PIL import Image

        rgb = Image.open(OUTPUT_PNG).convert("RGB")
        rgb.save(OUTPUT_PNG, format="PNG", optimize=True)
        jpg_path = OUTPUT_PNG.replace(".png", ".jpg")
        rgb.save(jpg_path, format="JPEG", quality=95)
        print(f"  Also wrote {jpg_path}")
    except Exception as exc:
        print(f"  Warning: could not convert PNG to RGB/JPG ({exc})")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(OUTPUT_STATS_CSV, index=False)

    if point_frames:
        points_df = pd.concat(point_frames, ignore_index=True)
        points_df.to_csv(OUTPUT_POINTS_CSV, index=False)
    else:
        points_df = pd.DataFrame()

    print("\nDone.")
    print(f"  {OUTPUT_PNG}")
    print(f"  {OUTPUT_PDF}")
    print(f"  {OUTPUT_STATS_CSV}")
    if not points_df.empty:
        print(f"  {OUTPUT_POINTS_CSV}")


if __name__ == "__main__":
    main()
