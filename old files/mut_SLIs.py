import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load SLRFM scores
# --------------------------------------------------
scores = pd.read_csv("datasets/sclc_slrfm_candidates_lung.csv")

# Clean the score column (remove NaN/Inf)
score_vals = (
    scores["score"]
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    .values
)

# --------------------------------------------------
# 2. Find elbow on the score distribution
#    (classic "max distance from line between endpoints" method)
# --------------------------------------------------
s_sorted = np.sort(score_vals)
n = len(s_sorted)
x = np.arange(n)

# line from (0, s0) to (n-1, s_last)
start = np.array([0.0, s_sorted[0]])
end   = np.array([n - 1.0, s_sorted[-1]])
line_vec = end - start
line_unit = line_vec / np.linalg.norm(line_vec)

# vector from start to each point
points = np.vstack([x, s_sorted]).T
points_from_start = points - start

# projection of each point onto the line
proj_len = points_from_start.dot(line_unit)
proj_points = np.outer(proj_len, line_unit) + start

# distance from each point to the straight line
distances = np.linalg.norm(points - proj_points, axis=1)

# elbow index & cutoff score
elbow_idx = np.argmax(distances)
score_cutoff = float(s_sorted[elbow_idx])

print(f"Elbow-based score cutoff ≈ {score_cutoff:.6f}")

num_above = (scores["score"] >= score_cutoff).sum()
print(f"{num_above} KOs have score ≥ cutoff")

# --------------------------------------------------
# 3. Make SI-style histogram figure
# --------------------------------------------------
plt.figure(figsize=(6,4))

plt.hist(score_vals, bins=80, color="tab:blue", edgecolor="none")
plt.yscale("log")
plt.axvline(score_cutoff, color="orange", linestyle="--", linewidth=2)

plt.xlabel("Score")
plt.ylabel("Frequency (log)")
plt.title("Distribution of knockout scores")

plt.tight_layout()
# Optional: save figure to file
# plt.savefig("sclc_slrfm_score_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# --------------------------------------------------
# 4. Filter SL pairs by cutoff & write Excel
# --------------------------------------------------
hits = scores[scores["score"] >= score_cutoff].copy()
hits = hits.sort_values("score", ascending=False)

# All hits (mixed mutation + expression)
all_hits = hits

# Expression-only hits
expr_hits = hits[hits["feature_type"] == "expression"]

# Mutation-only hits
mut_hits = hits[hits["feature_type"] == "mutation"]

print(f"All hits above cutoff:        {len(all_hits)}")
print(f"Expression-only hits above:  {len(expr_hits)}")
print(f"Mutation-only hits above:    {len(mut_hits)}")

out_xlsx = "datasets/sclc_slrfm_hits_elbow_lung.xlsx"

# If you still get ModuleNotFoundError for openpyxl, run:
#   pip install openpyxl
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    all_hits.to_excel(writer, sheet_name="all_hits", index=False)
    expr_hits.to_excel(writer, sheet_name="expression_hits", index=False)
    mut_hits.to_excel(writer, sheet_name="mutation_hits", index=False)

print(f"Saved Excel file with 3 sheets → {out_xlsx}")
