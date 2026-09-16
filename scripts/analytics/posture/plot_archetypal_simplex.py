import os

# Pin BLAS to one thread before numpy/archetypes load: multi-threaded BLAS
# reduction order isn't deterministic run-to-run, which can flip a seeded
# AA.fit() to a different local optimum (see posture_features.py).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "gold" / "posture"))
from posture_features import CLUSTER_FEATURES, fit_aa  # noqa: E402
import layers as L  # noqa: E402

# 1. Load universe and define proper 4-digit SIC mapping
fu = (L.scan("silver.firm_universe").filter(pl.col("ticker").is_not_null())
      .select("ticker", "company_name", "sic", "industry_group").collect().to_pandas())
fu["sic2"] = fu["sic"].astype(str).str.zfill(4).str[:2]

def map_sector(sic2):
    try: s = int(sic2)
    except: return "Other / Diversified"
    if s in [73, 35, 36]: return "Technology"
    elif s in [48]: return "Communications"
    elif s in [28, 38, 80]: return "Healthcare & Pharma"
    elif 60 <= s <= 67: return "Financial Services"
    elif 20 <= s <= 39: return "Industrials & Mfg"
    elif 40 <= s <= 47: return "Transportation"
    elif s == 49: return "Utilities"
    elif 10 <= s <= 14 or s == 29: return "Energy & Mining"
    elif 50 <= s <= 59: return "Retail & Wholesale"
    elif 70 <= s <= 89: return "Business Services"
    else: return "Other / Diversified"

fu["sector"] = fu["sic2"].apply(map_sector)
fu_sector_map = fu.set_index("ticker")["sector"].to_dict()
fu_name_map = fu.set_index("ticker")["company_name"].to_dict()

# 2. Load posture data and fit AA(k=3)
df = L.read_gold("firm", ("covariates", "posture_archetype_static"))
# fitted features come from the one place that defines them: intensity is a
# covariate, not a posture, so it does not shape the vertices
FEATS = list(CLUSTER_FEATURES)
fit_pop = df[df["archetype"] != "No AI"].copy()
X = fit_pop[FEATS].values
X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
aa, W = fit_aa(X_std, 3)

A = aa.archetypes_
# 2. Fix archetypes by semantic definition (fixed by feature name, not positional index)
def_col = int(np.argmax(A[:, FEATS.index("risk_orientation")]))
rem_cols = [i for i in range(3) if i != def_col]
gov_col = max(rem_cols, key=lambda i: A[i, FEATS.index("governance_orientation")])
voc_col = [i for i in range(3) if i not in (def_col, gov_col)][0]

W_named = pd.DataFrame({
    "Vocal": W[:, voc_col],
    "Governance": W[:, gov_col],
    "Defensive": W[:, def_col]
}, index=fit_pop["ticker"])

fit_pop["w_Vocal"] = W_named["Vocal"].values
fit_pop["w_Gov"] = W_named["Governance"].values
fit_pop["w_Def"] = W_named["Defensive"].values
fit_pop["sector"] = fit_pop["ticker"].map(fu_sector_map).fillna("Other / Diversified")
fit_pop["company_name"] = fit_pop["ticker"].map(fu_name_map)

# Coordinates in equilateral triangle
# Vocal: (0, 0)
# Gov: (1, 0)
# Def: (0.5, sqrt(3)/2)
w_v = fit_pop["w_Vocal"].values
w_g = fit_pop["w_Gov"].values
w_d = fit_pop["w_Def"].values
fit_pop["x_tern"] = w_g + 0.5 * w_d
fit_pop["y_tern"] = (np.sqrt(3) / 2) * w_d

# ==============================================================================
# MAIN CHAPTER FIGURE: Minimalist Simplex, Colored Centroids & Perimeter Tickers
# ==============================================================================
fig, ax = plt.subplots(figsize=(15, 14), dpi=300)

# Triangle border
ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], "k-", lw=2.2, zorder=3)

# w = 0.5 partition lines meeting at barycenter (0.5, sqrt(3)/6)
c_x, c_y = 0.5, np.sqrt(3)/6
ax.plot([0.5, c_x], [0, c_y], color="#94A3B8", lw=1.1, ls=":", zorder=2)
ax.plot([0.75, c_x], [np.sqrt(3)/4, c_y], color="#94A3B8", lw=1.1, ls=":", zorder=2)
ax.plot([0.25, c_x], [np.sqrt(3)/4, c_y], color="#94A3B8", lw=1.1, ls=":", zorder=2)

# Region watermark labels
ax.text(0.33, 0.06, "Pure Vocal (w > 0.5)", fontsize=7.2, color="#94A3B8", ha="center", style="italic")
ax.text(0.67, 0.06, "Pure Governance (w > 0.5)", fontsize=7.2, color="#94A3B8", ha="center", style="italic")
ax.text(0.50, 0.60, "Pure Defensive (w > 0.5)", fontsize=7.2, color="#94A3B8", ha="center", style="italic")
ax.text(0.50, 0.14, "Mixed Postures", fontsize=7.2, color="#94A3B8", ha="center", style="italic")

# 1. Cloud of 441 firms in light gray, no border, alpha 0.35
ax.scatter(fit_pop["x_tern"], fit_pop["y_tern"],
           c="#94A3B8", s=18, alpha=0.35, edgecolors="none", zorder=4)

# 2. Colors for the 10 Genuine Sectors (excluding single-firm Other)
SECTOR_COLORS = {
    "Technology": "#2563EB",          # Blue
    "Industrials & Mfg": "#475569",   # Slate dark
    "Financial Services": "#059669",   # Emerald green
    "Healthcare & Pharma": "#D97706", # Amber dark
    "Retail & Wholesale": "#DB2777",  # Pink
    "Utilities": "#0891B2",           # Cyan
    "Energy & Mining": "#B45309",     # Rust brown
    "Transportation": "#7C3AED",      # Violet
    "Business Services": "#0D9488",   # Teal
    "Communications": "#E11D48",      # Crimson red
}

# 3. Sector Centroids summary (filter out Other / Diversified)
sec_grp = fit_pop[fit_pop["sector"] != "Other / Diversified"].groupby("sector").agg(
    n=("ticker", "count"),
    x=("x_tern", "mean"),
    y=("y_tern", "mean")
).reset_index()

# Provably optimal, non-crossing assignment of centroids to perimeter labels:
# Ordered along the right edge from bottom (y=0.25) to top (y=0.83)
RIGHT_ORDER = [
    "Energy & Mining",
    "Utilities",
    "Financial Services",
    "Transportation",
    "Healthcare & Pharma",
    "Industrials & Mfg",
    "Retail & Wholesale",
    "Communications",
    "Business Services"
]

# Ordered along the left edge from bottom to top
LEFT_ORDER = [
    "Technology"
]

sec_dict = sec_grp.set_index("sector").to_dict(orient="index")

# Right-side Labels: y from 0.20 to 0.83 for 9 sectors
y_right = np.linspace(0.20, 0.83, len(RIGHT_ORDER))
for idx, sec in enumerate(RIGHT_ORDER):
    sid = idx + 1
    d = sec_dict[sec]
    sx, sy, n = d["x"], d["y"], int(d["n"])
    col = SECTOR_COLORS[sec]
    
    ly = y_right[idx]
    lx = (1.0 - ly / np.sqrt(3)) + 0.080
    
    # Colored Centroid Ball ("bola") with matching number
    marker_size = 75 + 18 * np.sqrt(n)
    ax.scatter(sx, sy, s=marker_size, facecolor=col, edgecolor="#0F172A",
               linewidth=1.4, zorder=6)
    ax.text(sx, sy, str(sid), fontsize=7.2, fontweight="bold",
            color="white", ha="center", va="center", zorder=7)
    
    # Non-crossing radial line
    ax.plot([sx, lx], [sy, ly], color=col, lw=1.2, alpha=0.85, zorder=5)
    
    # Outer Label Box with matching colored border & number
    label_text = f" {sid}  {sec} (n={n}) "
    ax.text(lx, ly, label_text,
            fontsize=7.8, fontweight="bold", ha="left", va="center",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.3, alpha=0.96),
            zorder=8)

# Left-side Labels: Technology
y_left = [0.20]
for idx, sec in enumerate(LEFT_ORDER):
    sid = len(RIGHT_ORDER) + idx + 1
    d = sec_dict[sec]
    sx, sy, n = d["x"], d["y"], int(d["n"])
    col = SECTOR_COLORS[sec]
    
    ly = y_left[idx]
    lx = (ly / np.sqrt(3)) - 0.085
    
    # Ball
    marker_size = 75 + 18 * np.sqrt(n)
    ax.scatter(sx, sy, s=marker_size, facecolor=col, edgecolor="#0F172A",
               linewidth=1.4, zorder=6)
    ax.text(sx, sy, str(sid), fontsize=7.2, fontweight="bold",
            color="white", ha="center", va="center", zorder=7)
    
    # Leader line
    ax.plot([sx, lx], [sy, ly], color=col, lw=1.2, alpha=0.85, zorder=5)
    
    # Label box
    label_text = f" {sid}  {sec} (n={n}) "
    ax.text(lx, ly, label_text,
            fontsize=7.8, fontweight="bold", ha="right", va="center",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, lw=1.3, alpha=0.96),
            zorder=8)

# 4. Perimeter Boundary Exemplars across ALL THREE EDGES (Demonstrating Simplex Clipping)
# Dynamic label generator: strictly argmax(w) with semantic suffix v/g/d
ticker_map = fit_pop.set_index("ticker").to_dict(orient="index")

def get_exemplar_label(t):
    r = ticker_map[t]
    weights = {"v": r["w_Vocal"], "g": r["w_Gov"], "d": r["w_Def"]}
    dom_code, max_val = max(weights.items(), key=lambda item: item[1])
    return f"{t} ({max_val:.2f}{dom_code})"

# A. Bottom Edge (w_def == 0.00): Vocal <-> Governance
BOTTOM_EXEMPLARS = [
    ("NVDA",  (0, -14), "center", "top"),
    ("MSFT",  (0, -14), "center", "top"),
    ("GOOGL", (0, -14), "center", "top"),
    ("AMZN",  (0, -14), "center", "top"),
    ("AAPL",  (0, -14), "center", "top"),
    ("RTX",   (0, -14), "center", "top"),
    ("VLO",   (0, -14), "center", "top"),
]

# B. Left Edge (w_gov == 0.00): Vocal <-> Defensive (outside to left)
LEFT_EXEMPLARS = [
    ("ANET", (-10, 0), "right", "center"),
    ("BKNG", (-10, 0), "right", "center"),
    ("CLX",  (-10, 0), "right", "center"),
    ("ETN",  (-10, 0), "right", "center"),
]

# C. Right Edge (w_voc == 0.00): Governance <-> Defensive (inside to left)
RIGHT_EXEMPLARS = [
    ("C",    (-10, 0), "right", "center"),
    ("AZO",  (-10, 0), "right", "center"),
    ("ADM",  (-10, 0), "right", "center"),
    ("ED",   (-10, 0), "right", "center"),
]

# D. Interior Center Exemplars (Mixed Postures near Barycenter)
CENTER_EXEMPLARS = [
    ("OXY",  (-8, -12), "right", "top"),
    ("JPM",  (12, -8),  "left",  "top"),
]

# Top Vertex Exemplar
dltr_lbl = get_exemplar_label("DLTR")
ax.scatter([0.5], [np.sqrt(3)/2], color="#0F172A", s=32, zorder=9)
ax.annotate(dltr_lbl, xy=(0.5, np.sqrt(3)/2), xytext=(0, 10),
            textcoords="offset points", fontsize=8.0, fontweight="bold",
            ha="center", va="bottom", color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E1", lw=0.8, alpha=0.95),
            zorder=10)

for t, off, ha, va in BOTTOM_EXEMPLARS:
    r = ticker_map[t]
    lbl = get_exemplar_label(t)
    ax.scatter([r["x_tern"]], [r["y_tern"]], color="#0F172A", s=28, zorder=9)
    ax.annotate(lbl, xy=(r["x_tern"], r["y_tern"]), xytext=off,
                textcoords="offset points", fontsize=7.6, fontweight="bold",
                ha=ha, va=va, color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E1", lw=0.8, alpha=0.95),
                zorder=10)

for t, off, ha, va in LEFT_EXEMPLARS:
    r = ticker_map[t]
    lbl = get_exemplar_label(t)
    ax.scatter([r["x_tern"]], [r["y_tern"]], color="#0F172A", s=28, zorder=9)
    ax.annotate(lbl, xy=(r["x_tern"], r["y_tern"]), xytext=off,
                textcoords="offset points", fontsize=7.6, fontweight="bold",
                ha=ha, va=va, color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E1", lw=0.8, alpha=0.95),
                zorder=10)

for t, off, ha, va in RIGHT_EXEMPLARS:
    r = ticker_map[t]
    lbl = get_exemplar_label(t)
    ax.scatter([r["x_tern"]], [r["y_tern"]], color="#0F172A", s=28, zorder=9)
    ax.annotate(lbl, xy=(r["x_tern"], r["y_tern"]), xytext=off,
                textcoords="offset points", fontsize=7.6, fontweight="bold",
                ha=ha, va=va, color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CBD5E1", lw=0.8, alpha=0.95),
                zorder=10)

for t, off, ha, va in CENTER_EXEMPLARS:
    r = ticker_map[t]
    lbl = get_exemplar_label(t)
    ax.scatter([r["x_tern"]], [r["y_tern"]], color="#0F172A", s=30, zorder=9)
    ax.annotate(lbl, xy=(r["x_tern"], r["y_tern"]), xytext=off,
                textcoords="offset points", fontsize=7.6, fontweight="bold",
                ha=ha, va=va, color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.2", fc="#F8FAFC", ec="#94A3B8", lw=0.9, alpha=0.95),
                zorder=10)

ax.set_xlim(-0.48, 1.52)
ax.set_ylim(-0.16, 1.05)
ax.set_aspect("equal")
ax.axis("off")

plt.title("The S&P 500 AI Disclosure Simplex: Industry Centroids & Boundary Exemplars (k = 3)\n441 Active Index Constituents (10-K & DEF 14A Statutory SEC Filings)",
          fontsize=13, fontweight="bold", pad=25)

# Vertex titles (flanking cleanly outside)
ax.text(-0.05, -0.07, "Vocal Substantives\n(Commercial Execution)",
        fontsize=11, fontweight="bold", ha="center", va="top", color="#0F172A")
ax.text(1.05, -0.07, "Governance-Led Disclosers\n(Institutional Guardrails)",
        fontsize=11, fontweight="bold", ha="center", va="top", color="#0F172A")
ax.text(0.5, np.sqrt(3)/2 + 0.05, "Defensive Disclosers\n(Risk-Framed)",
        fontsize=11, fontweight="bold", ha="center", va="bottom", color="#0F172A")

out_main = str(L.results_path("posture", "fig_archetypal_ternary_simplex.png"))
plt.tight_layout()
plt.savefig(out_main, dpi=300, bbox_inches="tight")
plt.close()
print("Saved clean Perimeter-Boundary Simplex Figure to:", out_main)
