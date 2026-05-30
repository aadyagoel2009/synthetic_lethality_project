"""
List SCLC cell lines with EP300 or CREBBP mutation features,
including the kind/type of mutation when available.

Run from your project root:

    python list_sclc_ep300_crebbp_mutations.py
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

import hickle as hkl
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths from your existing pipeline
# ---------------------------------------------------------------------------
SCLC_EMBED_HKL = "embeddings/final_X_sclc_processed.hkl"
MODEL_CSV_PATH = "datasets/2023/Model.csv"
RESULTS_DIR = "results"

GENES_TO_CHECK = ["EP300", "CREBBP"]


# ---------------------------------------------------------------------------
# Helpers
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
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Run this script from your project root."
        )

    return as_dataframe(hkl.load(path), name)


def load_model_metadata(model_path: str) -> pd.DataFrame:
    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. Using all rows in SCLC embedding.")
        return pd.DataFrame()

    return pd.read_csv(model_path)


def get_sclc_model_ids(meta: pd.DataFrame) -> set:
    """
    Return ModelIDs where OncotreeSubtype is small cell lung cancer.
    """
    if meta.empty:
        return set()

    required_cols = {"ModelID", "OncotreeSubtype"}
    if not required_cols.issubset(meta.columns):
        print("Warning: Model.csv missing ModelID or OncotreeSubtype. Using all embedding rows.")
        return set()

    subtype = meta["OncotreeSubtype"].fillna("").astype(str).str.strip().str.lower()
    is_sclc = subtype.eq("small cell lung cancer")

    return set(meta.loc[is_sclc, "ModelID"].astype(str))


def pick_name_column(meta: pd.DataFrame) -> str | None:
    """
    Pick a readable cell-line-name column if one exists.
    """
    candidates = [
        "StrippedCellLineName",
        "CellLineName",
        "CCLEName",
        "ModelName",
        "DepmapModelName",
    ]

    for col in candidates:
        if col in meta.columns:
            return col

    return None


def build_model_name_lookup(meta: pd.DataFrame) -> Dict[str, str]:
    """
    Map ModelID -> readable cell line name if possible.
    """
    if meta.empty or "ModelID" not in meta.columns:
        return {}

    name_col = pick_name_column(meta)

    if name_col is None:
        return {}

    tmp = meta[["ModelID", name_col]].dropna()

    return dict(zip(tmp["ModelID"].astype(str), tmp[name_col].astype(str)))


def find_mutation_columns(columns: pd.Index, gene: str) -> List[str]:
    """
    Finds mutation-style feature columns for a gene.

    Expression columns ending in _exp are excluded.

    Examples included:
      EP300
      EP300_mut
      EP300_mutation
      EP300_missense
      EP300_nonsense
      EP300_frame_shift_del
      CREBBP
      CREBBP_mut

    Examples excluded:
      EP300_exp
      CREBBP_exp
    """
    gene_upper = gene.upper()
    matches = []

    for col in columns:
        col_str = str(col)
        col_upper = col_str.upper()

        # Exclude expression features.
        if col_upper.endswith("_EXP"):
            continue

        # Match exact gene name or gene followed by a delimiter.
        if col_upper == gene_upper or re.match(rf"^{re.escape(gene_upper)}[_\-.]", col_upper):
            matches.append(col_str)

    return matches


def is_mutated_value(x: object) -> bool:
    """
    Decide whether a feature value indicates mutation presence.

    Handles:
      - numeric mutation columns: nonzero means mutated
      - boolean columns: True means mutated
      - string columns: non-empty/non-WT means mutated
    """
    if pd.isna(x):
        return False

    if isinstance(x, (int, float, np.integer, np.floating, bool)):
        return float(x) != 0.0

    text = str(x).strip().lower()

    not_mutated_values = {
        "",
        "0",
        "0.0",
        "false",
        "none",
        "nan",
        "wt",
        "wildtype",
        "wild_type",
        "not_mutated",
        "no_mutation",
    }

    if text in not_mutated_values:
        return False

    return True


def infer_mutation_kind_from_feature(feature_col: str, gene: str) -> str:
    """
    Try to infer mutation kind from the feature column name.

    For example:
      EP300_missense         -> missense
      EP300_nonsense         -> nonsense
      EP300_frame_shift_del  -> frame_shift_del
      EP300_splice_site      -> splice_site
      EP300_mut              -> mutation
      EP300                  -> mutation_present
    """
    col = str(feature_col)
    gene_upper = gene.upper()
    col_upper = col.upper()

    if col_upper == gene_upper:
        return "mutation_present"

    # Remove the gene prefix and delimiters.
    remainder = re.sub(
        rf"^{re.escape(gene)}[_\-.]*",
        "",
        col,
        flags=re.IGNORECASE,
    ).strip("_-. ")

    if remainder == "":
        return "mutation_present"

    normalized = remainder.lower()

    # Clean common naming variants.
    normalized = normalized.replace("mutation", "mutation")
    normalized = normalized.replace("mut", "mutation") if normalized == "mut" else normalized

    return normalized


def infer_mutation_kind_from_value(value: object) -> str:
    """
    Try to infer mutation kind from the feature value itself.

    This helps if the column is something like EP300_mutation
    and the value is Missense_Mutation, Nonsense_Mutation, etc.
    """
    if pd.isna(value):
        return ""

    if isinstance(value, (int, float, np.integer, np.floating, bool)):
        return ""

    text = str(value).strip()

    if text == "":
        return ""

    text_lower = text.lower()

    not_kind_values = {
        "1",
        "1.0",
        "true",
        "mut",
        "mutation",
        "mutated",
        "yes",
    }

    if text_lower in not_kind_values:
        return ""

    return text


def infer_mutation_kind(feature_col: str, gene: str, value: object) -> str:
    """
    Prefer the mutation kind from the value if it has a descriptive string.
    Otherwise infer it from the feature column name.
    """
    value_kind = infer_mutation_kind_from_value(value)

    if value_kind:
        return value_kind

    return infer_mutation_kind_from_feature(feature_col, gene)


def get_mutated_rows_for_gene(
    embedding: pd.DataFrame,
    gene: str,
    feature_cols: List[str],
    model_name_lookup: Dict[str, str],
) -> pd.DataFrame:
    rows = []

    if not feature_cols:
        return pd.DataFrame(
            columns=[
                "gene",
                "model_id",
                "cell_line_name",
                "mutation_feature",
                "mutation_kind",
                "feature_value",
            ]
        )

    for model_id, vals in embedding[feature_cols].iterrows():
        model_id_str = str(model_id)

        for feature_col in feature_cols:
            val = vals[feature_col]

            if is_mutated_value(val):
                mutation_kind = infer_mutation_kind(
                    feature_col=feature_col,
                    gene=gene,
                    value=val,
                )

                rows.append(
                    {
                        "gene": gene,
                        "model_id": model_id_str,
                        "cell_line_name": model_name_lookup.get(model_id_str, ""),
                        "mutation_feature": feature_col,
                        "mutation_kind": mutation_kind,
                        "feature_value": val,
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading SCLC embedding...")
    sclc_embedding = load_hkl_dataframe(SCLC_EMBED_HKL, "sclc_embedding")
    sclc_embedding.index = sclc_embedding.index.astype(str)

    print("Loading Model.csv metadata...")
    meta = load_model_metadata(MODEL_CSV_PATH)

    sclc_ids = get_sclc_model_ids(meta)
    model_name_lookup = build_model_name_lookup(meta)

    # The SCLC embedding should already be SCLC-only.
    # This extra filter double-checks against Model.csv when possible.
    if sclc_ids:
        keep_ids = [idx for idx in sclc_embedding.index if idx in sclc_ids]

        if keep_ids:
            sclc_embedding = sclc_embedding.loc[keep_ids]
        else:
            print(
                "Warning: no overlap between SCLC embedding IDs and Model.csv SCLC IDs. "
                "Using all rows in SCLC embedding."
            )

    print(f"\nSCLC rows being checked: {len(sclc_embedding)}")
    print(f"Total embedding features: {sclc_embedding.shape[1]}")

    all_results = []

    for gene in GENES_TO_CHECK:
        feature_cols = find_mutation_columns(sclc_embedding.columns, gene)

        print("\n" + "=" * 80)
        print(f"{gene}: mutation-style feature columns found:")
        print(feature_cols if feature_cols else "NONE")

        result_df = get_mutated_rows_for_gene(
            embedding=sclc_embedding,
            gene=gene,
            feature_cols=feature_cols,
            model_name_lookup=model_name_lookup,
        )

        if result_df.empty:
            print(f"\nNo SCLC cell lines found with a nonzero/non-empty {gene} mutation feature.")
            continue

        collapsed = (
            result_df.groupby(["gene", "model_id", "cell_line_name"], dropna=False)
            .agg(
                mutation_features=("mutation_feature", lambda x: "; ".join(map(str, x))),
                mutation_kinds=("mutation_kind", lambda x: "; ".join(sorted(set(map(str, x))))),
                feature_values=("feature_value", lambda x: "; ".join(map(str, x))),
            )
            .reset_index()
            .sort_values(["gene", "cell_line_name", "model_id"])
        )

        print(f"\nMutated SCLC cell lines for {gene}: {len(collapsed)}")
        print(collapsed.to_string(index=False))

        all_results.append(collapsed)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
    else:
        final_df = pd.DataFrame(
            columns=[
                "gene",
                "model_id",
                "cell_line_name",
                "mutation_features",
                "mutation_kinds",
                "feature_values",
            ]
        )

    out_csv = os.path.join(
        RESULTS_DIR,
        "sclc_ep300_crebbp_mutation_cell_lines.csv",
    )

    final_df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print(f"Saved summary CSV: {out_csv}")


if __name__ == "__main__":
    main()