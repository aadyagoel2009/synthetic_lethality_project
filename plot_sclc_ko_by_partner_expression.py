"""
Validate expression-based (KO, partner) pairs in SCLC.

For each pair:
  1. Decide whether the dependency is on LOW or HIGH partner expression, using
     the sign of the correlation between partner log1p(TPM) and the KO Chronos
     effect across SCLC lines.
       corr > 0  -> more expression means a weaker KO effect  -> needs LOW
       corr < 0  -> more expression means a stronger KO effect -> needs HIGH
  2. Split SCLC lines at the median partner expression and box-plot the KO
     Chronos effect for both halves. The half predicted to be more dependent
     is highlighted and named in the panel title.

Whiskers are matplotlib default Tukey whiskers (1.5 x IQR), not 2 SD / 2 SEM.
Restricted to SCLC cell lines only.

Run:
  python plot_sclc_ko_by_partner_expression.py
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
    ("DNAJC19", "DNAJC15"),
    ("FAM50A", "FAM50B"),
    ("TIMM17A", "TIMM17B"),
    ("CYB5B", "CYB5A"),
    ("COPG1", "COPG2"),
    ("GSPT1", "GSPT2"),
    ("DDX3X", "DDX3Y"),
    ("MAGOH", "MAGOHB"),
    ("EIF1AX", "EIF1AY"),
    ("CSTF2", "CSTF2T"),
    ("CCND1", "CCND1"),
    ("CCNE1", "CCNE1"),
    ("MYC", "MYC"),
    ("FGFR2", "FGFR2"),
    ("CCND3", "CCND3"),
    ("SOX9", "SOX9"),
    ("NFIB", "KCNA1"),
    ("NFIB", "NFIB"),
    ("FOXA2", "FOXA2"),
    ("NKX2-1", "NKX2-1"),
    ("SOX2", "SOX2"),
]

EXPRESSION_CSV = "datasets/2023/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
CRISPR_CSV = "datasets/2023/CRISPRGeneEffect.csv"
MODEL_CSV = "datasets/2023/Model.csv"

RESULTS_DIR = os.path.join("sli_jst_pcc_split_results", "partner_exp_validation")
OUTPUT_PNG = os.path.join(RESULTS_DIR, "sclc_ko_effects_by_partner_expression.png")
OUTPUT_PDF = os.path.join(RESULTS_DIR, "sclc_ko_effects_by_partner_expression.pdf")
OUTPUT_STATS_CSV = os.path.join(
    RESULTS_DIR, "sclc_ko_effects_by_partner_expression_stats.csv"
)
OUTPUT_POINTS_CSV = os.path.join(
    RESULTS_DIR, "sclc_ko_effects_by_partner_expression_points.csv"
)


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


def load_selected_columns(path: str, genes: List[str], id_col_first: bool) -> pd.DataFrame:
    """
    Read only the requested gene columns from a wide DepMap CSV.

    DepMap headers look like "TP53 (7157)", so match on the leading symbol.
    """
    header = pd.read_csv(path, nrows=0)
    raw_cols = list(header.columns)
    id_col = "ModelID" if "ModelID" in raw_cols else raw_cols[0]

    wanted = {str(g) for g in genes}
    symbol_to_raw: Dict[str, str] = {}
    for col in raw_cols:
        symbol = str(col).split()[0]
        if symbol in wanted and symbol not in symbol_to_raw:
            symbol_to_raw[symbol] = col

    usecols = [id_col] + list(symbol_to_raw.values())
    df = pd.read_csv(path, usecols=usecols)
    df = df.rename(columns={v: k for k, v in symbol_to_raw.items()})
    df = df.set_index(id_col)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.groupby(level=0).mean()
    df.index = df.index.astype(str)

    missing = sorted(wanted - set(df.columns))
    if missing:
        print(f"  Not found in {os.path.basename(path)}: {', '.join(missing)}")
    return df


def decide_direction(expression: np.ndarray, chronos: np.ndarray) -> Tuple[str, float, float]:
    """
    Return ("low"|"high", pearson_r, p) for the partner expression requirement.

    Chronos is more negative when the knockout is more lethal, so a positive
    correlation means low expression goes with the stronger dependency.
    """
    if len(expression) < 3 or np.std(expression) == 0 or np.std(chronos) == 0:
        return "low", float("nan"), float("nan")
    r, p = stats.pearsonr(expression, chronos)
    return ("low" if r >= 0 else "high"), float(r), float(p)


def collect_pair_data(
    effects: pd.DataFrame,
    expression: pd.DataFrame,
    sclc_ids: pd.Index,
    ko_gene: str,
    partner_gene: str,
) -> pd.DataFrame:
    """One row per SCLC line: KO Chronos effect plus partner expression half."""
    if ko_gene not in effects.columns:
        raise KeyError(f"KO gene {ko_gene} missing from CRISPR effects.")
    if partner_gene not in expression.columns:
        raise KeyError(f"Partner gene {partner_gene} missing from expression matrix.")

    cells = sclc_ids.intersection(effects.index).intersection(expression.index)
    if len(cells) == 0:
        raise ValueError(f"No overlapping SCLC cells for {ko_gene}/{partner_gene}.")

    chronos = effects.loc[cells, ko_gene].astype(float)
    expr = expression.loc[cells, partner_gene].astype(float)
    valid = chronos.notna() & expr.notna()
    chronos = chronos.loc[valid]
    expr = expr.loc[valid]
    if len(chronos) < 4:
        raise ValueError(f"Only {len(chronos)} usable SCLC lines.")

    direction, corr_r, corr_p = decide_direction(expr.to_numpy(), chronos.to_numpy())
    median_expr = float(expr.median())
    is_low = expr <= median_expr

    return pd.DataFrame(
        {
            "model_id": chronos.index.astype(str),
            "ko_gene": ko_gene,
            "partner_gene": partner_gene,
            "required_direction": direction,
            "pearson_r": corr_r,
            "pearson_p": corr_p,
            "median_log_tpm": median_expr,
            "partner_log_tpm": expr.values,
            "expression_half": np.where(is_low.values, "low", "high"),
            "chronos_ko_effect": chronos.values,
        }
    )


def plot_pair_boxplot(ax: plt.Axes, pair_df: pd.DataFrame) -> Dict:
    ko_gene = str(pair_df["ko_gene"].iloc[0])
    partner_gene = str(pair_df["partner_gene"].iloc[0])
    direction = str(pair_df["required_direction"].iloc[0])
    corr_r = float(pair_df["pearson_r"].iloc[0])

    low_vals = pair_df.loc[
        pair_df["expression_half"] == "low", "chronos_ko_effect"
    ].to_numpy()
    high_vals = pair_df.loc[
        pair_df["expression_half"] == "high", "chronos_ko_effect"
    ].to_numpy()

    # Reference half first, hypothesised-dependent half second.
    if direction == "low":
        data = [high_vals, low_vals]
        labels = [
            f"{partner_gene} high\n(n={len(high_vals)})",
            f"{partner_gene} LOW\n(n={len(low_vals)})",
        ]
        group_vals, ref_vals = low_vals, high_vals
    else:
        data = [low_vals, high_vals]
        labels = [
            f"{partner_gene} low\n(n={len(low_vals)})",
            f"{partner_gene} HIGH\n(n={len(high_vals)})",
        ]
        group_vals, ref_vals = high_vals, low_vals

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    for patch, color in zip(bp["boxes"], ["#9ecae1", "#fc9272"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        ax.scatter(
            i + rng.uniform(-0.12, 0.12, size=len(vals)),
            vals,
            s=18,
            color="black",
            alpha=0.55,
            zorder=3,
        )

    pval = float("nan")
    if len(group_vals) and len(ref_vals):
        pval = float(
            stats.mannwhitneyu(group_vals, ref_vals, alternative="less").pvalue
        )

    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_title(
        f"{ko_gene} KO by {partner_gene} expression\n"
        f"needs {direction.upper()} {partner_gene} (r={corr_r:.2f})",
        fontsize=9,
    )
    ax.set_ylabel("Chronos gene effect (KO)")
    if np.isfinite(pval):
        ax.text(
            0.5,
            0.98,
            f"MWU p={pval:.3g}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    return {
        "ko_gene": ko_gene,
        "partner_gene": partner_gene,
        "required_direction": direction,
        "pearson_r": corr_r,
        "pearson_p": float(pair_df["pearson_p"].iloc[0]),
        "median_log_tpm": float(pair_df["median_log_tpm"].iloc[0]),
        "n_reference": int(len(ref_vals)),
        "n_group": int(len(group_vals)),
        "mean_reference": float(np.mean(ref_vals)) if len(ref_vals) else float("nan"),
        "mean_group": float(np.mean(group_vals)) if len(group_vals) else float("nan"),
        "delta_group_minus_reference": (
            float(np.mean(group_vals) - np.mean(ref_vals))
            if len(group_vals) and len(ref_vals)
            else float("nan")
        ),
        "mannwhitney_p_group_less": pval,
    }


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ko_genes = sorted({ko for ko, _ in PAIRS})
    partner_genes = sorted({partner for _, partner in PAIRS})

    print("Loading CRISPR gene effects (selected KOs)...")
    effects = load_selected_columns(CRISPR_CSV, ko_genes, id_col_first=True)
    sclc_ids = pd.Index([i for i in effects.index if i in get_sclc_model_ids()])
    print(f"  SCLC lines in CRISPR: {len(sclc_ids)}")

    print("Loading log1p(TPM) expression (selected partners)...")
    expression = load_selected_columns(EXPRESSION_CSV, partner_genes, id_col_first=False)

    point_frames: List[pd.DataFrame] = []
    stats_rows: List[Dict] = []

    n_pairs = len(PAIRS)
    ncols = 5
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.4 * ncols, 3.8 * nrows),
        squeeze=False,
        facecolor="white",
    )

    for i, (ko_gene, partner_gene) in enumerate(PAIRS):
        ax = axes[i // ncols][i % ncols]
        try:
            pair_df = collect_pair_data(
                effects, expression, sclc_ids, ko_gene, partner_gene
            )
            point_frames.append(pair_df)
            row = plot_pair_boxplot(ax, pair_df)
            stats_rows.append(row)
            print(
                f"  {ko_gene}/{partner_gene}: needs {row['required_direction']}, "
                f"r={row['pearson_r']:.2f}, p={row['mannwhitney_p_group_less']:.3g}"
            )
        except Exception as exc:
            ax.set_title(f"{ko_gene} KO by {partner_gene}", fontsize=9)
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
        "SCLC: KO Chronos effect split by partner expression "
        "(red = half expected to be more dependent)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    try:
        from PIL import Image

        Image.open(OUTPUT_PNG).convert("RGB").save(
            OUTPUT_PNG, format="PNG", optimize=True
        )
    except Exception as exc:
        print(f"  Warning: could not convert PNG to RGB ({exc})")

    pd.DataFrame(stats_rows).to_csv(OUTPUT_STATS_CSV, index=False)
    if point_frames:
        pd.concat(point_frames, ignore_index=True).to_csv(OUTPUT_POINTS_CSV, index=False)

    print("\nDone.")
    print(f"  {OUTPUT_PNG}")
    print(f"  {OUTPUT_PDF}")
    print(f"  {OUTPUT_STATS_CSV}")
    print(f"  {OUTPUT_POINTS_CSV}")


if __name__ == "__main__":
    main()
