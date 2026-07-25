"""
Same SCLC mutation box plots as plot_sclc_ko_by_partner_mutation.py, for the
TP53-partnered pairs:

  USP28/TP53, PLK1/TP53, ATM/TP53, CHEK2/TP53

Restricted to SCLC cell lines only.

Note: every SCLC line in DepMap is TP53-mutant, so the WT box will be empty
and the Mann-Whitney p-value is undefined. The figure still shows the SCLC
MUT Chronos distribution for each KO.

Whiskers are matplotlib default Tukey whiskers (1.5 x IQR), not 2 SD / 2 SEM.

Run:
  python plot_sclc_ko_by_tp53_mutation.py
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_sclc_ko_by_partner_mutation import (
    collect_pair_data,
    get_sclc_model_ids,
    load_crispr_effects,
    load_mutation_matrix,
    plot_pair_boxplot,
)


PAIRS: List[Tuple[str, str]] = [
    ("USP28", "TP53"),
    ("PLK1", "TP53"),
    ("ATM", "TP53"),
    ("CHEK2", "TP53"),
]

RESULTS_DIR = os.path.join("sli_jst_pcc_split_results", "partner_mut_validation")
OUTPUT_PNG = os.path.join(RESULTS_DIR, "sclc_ko_effects_by_tp53_mutation.png")
OUTPUT_PDF = os.path.join(RESULTS_DIR, "sclc_ko_effects_by_tp53_mutation.pdf")
OUTPUT_STATS_CSV = os.path.join(
    RESULTS_DIR, "sclc_ko_effects_by_tp53_mutation_stats.csv"
)
OUTPUT_POINTS_CSV = os.path.join(
    RESULTS_DIR, "sclc_ko_effects_by_tp53_mutation_points.csv"
)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading CRISPR gene effects...")
    effects = load_crispr_effects()
    sclc_ids = pd.Index([i for i in effects.index if i in get_sclc_model_ids()])
    print(f"  SCLC lines in CRISPR: {len(sclc_ids)}")

    print("Loading damaging/hotspot mutations...")
    mutations = load_mutation_matrix()

    point_frames: List[pd.DataFrame] = []
    stats_rows: List[Dict] = []

    ncols = len(PAIRS)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(3.4 * ncols, 4.0),
        squeeze=False,
        facecolor="white",
    )

    for i, (ko_gene, partner_gene) in enumerate(PAIRS):
        ax = axes[0][i]
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
                f"p={stats_rows[-1]['mannwhitney_p_mut_less']}"
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

    fig.suptitle(
        "SCLC only: KO Chronos effect in TP53-MUT vs TP53-WT lines\n"
        "(all SCLC lines are TP53-mutant in DepMap; WT n=0)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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
