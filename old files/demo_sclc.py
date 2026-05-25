import pandas as pd
import numpy as np
import hickle as hkl
import matplotlib.pyplot as plt
from scipy.stats import zscore
from typing import Set

EMBEDDING_PATH = "embeddings/final_X_tcga_lung_processed.hkl"
CRISPR_PATH    = "datasets/2023/CRISPRGeneEffect_lung_processed.hkl"

FI_DATA_PATH   = "datasets/feature_importances_lung_data.npy"
FI_INDEX_PATH  = "datasets/feature_importances_lung_index.npy"
FI_COLS_PATH   = "datasets/feature_importances_lung_columns.npy"

MODEL_PATH     = "datasets/2023/Model.csv"

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
    SCLC defined exactly as your previous helper:
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
    sclc_model_ids = get_sclc_model_ids(model_df)
    sclc_ids = gene_effects_df.index.intersection(sclc_model_ids)
    if len(sclc_ids) == 0:
        raise ValueError(
            "No SCLC ModelID values from Model.csv intersect CRISPR index."
        )
    return sclc_ids


def get_top_indicators(M, k=10, feature_type="genes", importances=False):
    """
    Identical logic to original demo.py.
    """
    M = M.sort_values(ascending=False)
    if feature_type == "genes":
        features_sorted = [x.split("_")[0] for x in M.index]
        seen = set()
        genes_ordered = []
        for g in features_sorted:
            if g not in seen:
                genes_ordered.append(g)
                seen.add(g)
        return genes_ordered[:k]
    else:
        if importances:
            return M.loc[M.index[:k]].to_string(index=True, header=False)
        else:
            return [x for x in M.index[:k] if x.split("_")[-1] != "exp"]


def generate_inset_plots_SCLC(M, knockout, embedding, gene_effects_df, sclc_ids):
    """
    Same plot style as demo.py, but inset scatter is SCLC-only.
    """
    M = M.sort_values(ascending=False)
    feature_importances = M.to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    freq, bins, patches = ax.hist(feature_importances)
    ax.set_title("{}".format(knockout))
    ax.set_xlabel("Feature Weights")
    ax.set_ylabel("Frequency (log scale)")
    plt.yscale('log')

    top_feature = M.index[0]

    bin_centers = np.diff(bins) * 0.5 + bins[:-1]
    plt.annotate(
        "{}".format(top_feature),
        xy=(bin_centers[-1], int(freq[-1]) + 0.1),
        xytext=(0, 0.2),
        textcoords="offset points",
        ha='center', va='bottom',
        fontsize=8,
        weight='bold'
    )

    # inset
    left, bottom, width, height = [0.4, 0.35, 0.48, 0.48]
    ax2 = fig.add_axes([left, bottom, width, height])

    size = 8
    font = {'size': size}

    exp_feature = top_feature.split("_")[-1] == "exp"

    # ---- SCLC-only cells ----
    emb_sclc = embedding.loc[sclc_ids]
    ge_sclc  = gene_effects_df.loc[sclc_ids]

    if exp_feature:
        vals = emb_sclc[top_feature].values
        ind = np.abs(zscore(vals)) < 3
        x = emb_sclc[top_feature][ind]
        y = ge_sclc[knockout][ind]
    else:
        x = emb_sclc[top_feature]
        y = ge_sclc[knockout]

    r = np.corrcoef([x, y])[0, 1]

    exp_color = "seagreen" if r > 0 else "darkorange"
    ax2.scatter(x, y, c=exp_color if exp_feature else "darkviolet")

    if exp_feature:
        ax2.set_xlabel("Gene expression TPM of {}".format(top_feature.split("_")[0]), **font)
    else:
        ax2.set_xlabel("Mutation status of {}".format(top_feature), **font)

    if exp_feature:
        line_color = "limegreen" if r > 0 else "saddlebrown"
    else:
        line_color = "violet"

    xs = np.array(x)
    ys = np.array(y)
    if len(np.unique(xs)) > 1:
        ax2.plot(
            np.unique(xs),
            np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs)),
            zorder=10,
            color=line_color
        )

    ax2.set_ylabel("Gene effects (viability) of {}".format(knockout), **font)
    ax2.tick_params(axis='x', labelsize=size)
    ax2.tick_params(axis='y', labelsize=size)

    plt.title("PCC (SCLC): {}".format(str(round(r, 3))), **font)
    plt.show()


def prompt_input_SCLC(M, knockout, k, feature_type="features", importances=False):
    """
    Same as demo.py's prompt_input, but uses SCLC-only inset plotting.
    Uses global embedding/gene_effects_df/sclc_ids defined below.
    """
    if feature_type == "genes" and importances:
        print("ERROR: can only get importances for features not genes")
        return
    generate_inset_plots_SCLC(M, knockout, embedding, gene_effects_df, sclc_ids)
    data = get_top_indicators(M, k=k, feature_type=feature_type, importances=importances)
    if not importances:
        data = ", ".join(data)
    print("Top {} most important {}: \n".format(k, feature_type) + str(data))


embedding, gene_effects_df = load_embedding_and_crispr()
feature_importance_df     = load_feature_importances()
sclc_ids                  = load_sclc_ids(gene_effects_df)

print("embedding:", embedding.shape)
print("CRISPR:",   gene_effects_df.shape)
print("FI:",       feature_importance_df.shape)
print("SCLC lines:", len(sclc_ids))

# ===== choose KO and run, exactly like original demo.py =====
knockout     = "ATP1A1"          # <--- change this to e.g. "CREBBP", "SMARCA4", etc.
feature_type = "features"        # "features" or "genes"
importances  = True              # print actual weights?
k            = 10                # how many top features/genes

prompt_input_SCLC(feature_importance_df[knockout], knockout, k, feature_type, importances)
