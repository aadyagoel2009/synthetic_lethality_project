import numpy as np
import pandas as pd
import hickle as hkl
from numpy.linalg import solve, svd, norm
import matplotlib.pyplot as plt
import torch

def make_cell_embedding():
    prefix = "datasets/2023/"

    damaging_file   = prefix + "OmicsSomaticMutationsMatrixDamaging.csv"
    hotspot_file    = prefix + "OmicsSomaticMutationsMatrixHotspot.csv"
    expression_file = prefix + "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    crispr_file     = prefix + "CRISPRGeneEffect.csv"

    # ----------------- CRISPR gene effect matrix -----------------
    # First column = correct cell ID (as you said)
    gene_effects_df = pd.read_csv(crispr_file)
    # Clean gene column names: keep token before first space
    gene_effects_df.columns = [c.split()[0] for c in gene_effects_df.columns]

    id_col = gene_effects_df.columns[0]  # the correct ID column in CRISPR
    gene_effects_df = gene_effects_df.set_index(id_col)

    # Convert to numeric and merge duplicate IDs by averaging gene effects
    gene_effects_df = gene_effects_df.apply(pd.to_numeric, errors="coerce")
    gene_effects_df = gene_effects_df.groupby(level=0).mean()

    # ----------------- mutation features (damaging + hotspot) -----------------
    def load_mut(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = [c.split()[0] for c in df.columns]

        if "ModelID" not in df.columns:
            raise ValueError(f"'ModelID' not found in {path} columns: {df.columns[:10]}")

        # Use ModelID as index (correct ID for these files)
        df = df.set_index("ModelID")

        # Values are 0/1/2; convert to int (NaN -> 0)
        df = df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

        # If multiple rows per cell, keep the max per gene (0/1/2 semantics)
        df = df.groupby(level=0).max()

        return df

    mut_damaging = load_mut(damaging_file)
    mut_hotspot  = load_mut(hotspot_file)

    # Align mutation matrices to same set of cells & genes
    all_cells = mut_damaging.index.union(mut_hotspot.index)
    all_genes = mut_damaging.columns.union(mut_hotspot.columns)

    mut_damaging = mut_damaging.reindex(index=all_cells, columns=all_genes, fill_value=0)
    mut_hotspot  = mut_hotspot.reindex(index=all_cells, columns=all_genes, fill_value=0)

    # Combine damaging + hotspot by taking max per cell/gene (preserve 0/1/2)
    mut_combined_values = np.maximum(mut_damaging.values, mut_hotspot.values)
    embedding = pd.DataFrame(mut_combined_values, index=all_cells, columns=all_genes)

    # ----------------- expression features -----------------
    exp_df = pd.read_csv(expression_file)
    exp_df.columns = [c.split()[0] for c in exp_df.columns]

    if "ModelID" not in exp_df.columns:
        raise ValueError(f"'ModelID' not found in expression columns: {exp_df.columns[:10]}")

    exp_df = exp_df.set_index("ModelID")

    # numeric expression values
    exp_df = exp_df.apply(pd.to_numeric, errors="coerce")

    # Merge duplicate rows per cell by mean expression
    exp_df = exp_df.groupby(level=0).mean()

    # add "_exp" suffix to expression genes
    exp_df.columns = [c + "_exp" for c in exp_df.columns]

    # ----------------- align cell lines across CRISPR / mut / expr -----------------
    common_cells = (
        gene_effects_df.index
        .intersection(embedding.index)
        .intersection(exp_df.index)
    )

    if len(common_cells) == 0:
        raise ValueError(
            "No overlapping cell IDs between CRISPR, mutations, and expression. "
            "Given your description, the CRISPR first column and ModelID columns "
            "should share IDs; if they don't, we need to inspect the headers."
        )

    embedding       = embedding.loc[common_cells]
    gene_effects_df = gene_effects_df.loc[common_cells]
    exp_df          = exp_df.loc[common_cells]

    # merge mutation + expression features for those common cells
    embedding = embedding.join(exp_df, how="inner")
    embedding = embedding.dropna(axis=1, how="all").fillna(0)

    # ----------------- TCGA-based column filtering -----------------
    directory = "tcga/"
    disease   = "BRCA"

    tcga_mut_cols = pd.read_csv(
        f"{directory}{disease}/mc3_gene_level_{disease}_mc3_gene_level.txt",
        sep="\t",
        index_col=0,
    ).T.columns.tolist()

    tcga_exp_cols = pd.read_csv(
        f"{directory}{disease}/TCGA.{disease}.sampleMap_HiSeqV2",
        sep="\t",
        index_col=0,
    ).T.columns.tolist()
    # expression genes will show up in embedding with "_exp" suffix
    tcga_exp_cols = [c + "_exp" for c in tcga_exp_cols]

    tcga_cols   = tcga_mut_cols + tcga_exp_cols
    common_cols = list(set(tcga_cols) & set(embedding.columns))
    embedding   = embedding[common_cols]

    # which of these are expression features?
    exp_cols = [c for c in common_cols if c.endswith("_exp")]

    embedding = embedding.fillna(0)

    # ----------------- z-score expression columns + L2 normalize -----------------
    std_val = embedding[exp_cols].std(axis=0).replace(0, 1)
    embedding[exp_cols] = (embedding[exp_cols] - embedding[exp_cols].mean(axis=0)) / std_val

    # L2-normalize rows
    embedding /= norm(embedding, axis=1).reshape(-1, 1)

    return embedding, gene_effects_df