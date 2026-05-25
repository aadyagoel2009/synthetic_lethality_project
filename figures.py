import numpy as np
import pandas as pd
import hickle as hkl
import matplotlib.pyplot as plt
from typing import Set

CRISPR_PATH = "datasets/2023/CRISPRGeneEffect_lung_processed.hkl"
MODEL_PATH  = "datasets/2023/Model.csv"

# ---- helper to get SCLC IDs (same as before) ----
def get_sclc_model_ids(model_df: pd.DataFrame) -> Set[str]:
    model_df[["OncotreeSubtype", "OncotreePrimaryDisease"]] = (
        model_df[["OncotreeSubtype", "OncotreePrimaryDisease"]].fillna("")
    )
    is_sclc_subtype = (
        model_df["OncotreeSubtype"].astype(str).str.strip().str.lower()
        == "small cell lung cancer"
    )
    return set(model_df.loc[is_sclc_subtype, "ModelID"].astype(str))


# ---- load CRISPR + restrict to SCLC lines ----
gene_effects_df = hkl.load(CRISPR_PATH)
if not isinstance(gene_effects_df, pd.DataFrame):
    gene_effects_df = pd.DataFrame(gene_effects_df)

model_df = pd.read_csv(MODEL_PATH)
sclc_ids = gene_effects_df.index.intersection(get_sclc_model_ids(model_df))

print("SCLC-only CRISPR matrix shape:", gene_effects_df.loc[sclc_ids].shape)

# ---- KO panel: your genes + strongly lethal / survival genes in SCLC ----
KO_GENES = [
    # your list
    "KRAS", "PTEN", "MYC", "MYCL", "MYCN", "MAX", "RICTOR",
    "SOX2", "KMT2D", "KMT2C", "ARID1A",
    # anti-apoptotic / survival dependencies in SCLC
    "BCL2", "MCL1", "BCL2L1",
    # mitotic / checkpoint kinases often very lethal
    "AURKA", "PLK1", "CHEK1", "WEE1",
    # DNA repair / epigenetic target
    "PARP1", "EZH2",
]

# keep only genes actually present in CRISPR matrix
ko_present = [g for g in KO_GENES if g in gene_effects_df.columns]
print("KO genes present in CRISPR:", ko_present)

sclc_ge = gene_effects_df.loc[sclc_ids, ko_present]

# ---- plot heatmap ----
plt.figure(figsize=(len(ko_present) * 0.4 + 4, len(sclc_ids) * 0.25 + 4))

im = plt.imshow(
    sclc_ge.values,
    aspect="auto",
    cmap="coolwarm",        # blue = positive, red = negative
    vmin=-2.5, vmax=1.5     # tweak if you want more/less red
)

plt.colorbar(im, label="CRISPR gene-effect score")

plt.yticks(ticks=np.arange(len(sclc_ids)), labels=sclc_ids, fontsize=7)
plt.xticks(ticks=np.arange(len(ko_present)), labels=ko_present, fontsize=8, rotation=90)

plt.xlabel("Knockout gene")
plt.ylabel("SCLC cell line")
plt.title("SCLC viability across selected KO genes")

plt.tight_layout()
plt.show()
