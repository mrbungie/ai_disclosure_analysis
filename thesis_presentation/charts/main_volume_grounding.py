"""Chart 5: activity volume growth (with 2026 projection), firm-level grounding G,
and the five evidence attributes G averages, 2021 vs 2025."""
import sys

from style import C, fig, save, results, read, gold, REPO

sys.path.insert(0, str(REPO / "scripts" / "common"))
import layers as L  # noqa: E402
import pandas as pd  # noqa: E402

df = read(results("washing", "volume_vs_grounding.parquet")).set_index("year")

# hd_act_factor, computed exactly as thesis_document/thesis.qmd lines 174-199:
# full-2025 activity count / 2025 activities up to the same month-day as the
# latest 2026 filing, using the disclosure_volume covariate for the cutoff
# date and the activity spine (merged on accession_number) for the counts.
_dp_all = L.read_gold("document", ("covariates", "disclosure_volume"))
_dp = _dp_all[_dp_all["form"].isin(["10-K", "10-Q", "DEF 14A", "8-K"])].copy()
_dp["year"] = _dp["fecha"].dt.year
hd_cutoff = _dp.loc[_dp["year"] == 2026, "fecha"].max()
hd_cutoff_md = int(hd_cutoff.month * 100 + hd_cutoff.day)

_ad = pd.read_parquet(gold("spines", "activity", "activity"),
                       columns=["accession_number", "text_hash", "activity_id"]).drop_duplicates()
_ad = _ad.merge(_dp_all[["accession_number", "fecha"]], on="accession_number", how="inner")
_ad["year"] = _ad["fecha"].dt.year
_ad["md"] = _ad["fecha"].dt.month * 100 + _ad["fecha"].dt.day
hd_act_factor = float(((_ad["year"] == 2025).sum())
                       / (((_ad["year"] == 2025) & (_ad["md"] <= hd_cutoff_md)).sum()))

n2026_ytd = float(df.loc[2026, "n_activities"])
n2026_proj = n2026_ytd * hd_act_factor
ratio_5y = df.loc[2025, "n_activities"] / df.loc[2021, "n_activities"]

# G and its five attributes, from the filing-based firm-year activity counts
# that build_washing_score.py shrinks and averages into grounding_index.
COMPONENTS = [
    ("named_function", "Business function"),
    ("deployed_or_scaled", "Deployed or scaled"),
    ("named_product_or_process", "Named product or process"),
    ("quantified_outcome", "Quantified outcome"),
    ("third_party_named_provider", "Named provider"),
]
fy = L.read_gold("firm_year", ("covariates", "activities", ["n_activities"] + [c for c, _ in COMPONENTS]))
fy_sum = fy.groupby("year")[["n_activities"] + [c for c, _ in COMPONENTS]].sum()
comp_pct = fy_sum[[c for c, _ in COMPONENTS]].div(fy_sum["n_activities"], axis=0) * 100
ws = L.read_gold("firm_year", ("covariates", "washing_score"))
g_mean = ws[ws["w"].notna()].groupby("year")["grounding_index"].mean()

TITLE = dict(fontsize=21, color=C["ink_2"], fontweight="semibold", ha="left", va="bottom")

f = fig(1720, 500)
axL = f.add_axes([0.0, 0.1, 0.28, 0.76])
axR = f.add_axes([0.47, 0.1, 0.26, 0.76])
axM = f.add_axes([0.80, 0.1, 0.20, 0.76])

# ---- Left: activity volume bars ----
years = [2021, 2022, 2023, 2024, 2025]
vals = [df.loc[y, "n_activities"] for y in years]
xs = list(range(len(years)))
axL.bar(xs, vals, width=0.66, color=C["petrol"], zorder=3)
for x, v in zip(xs, vals):
    axL.annotate(f"{v / 1000:.1f}k", (x, v), xytext=(0, 7), textcoords="offset points",
                 ha="center", fontsize=18, fontweight="bold", color=C["petrol_dark"])

x26 = len(years)
axL.bar([x26], [n2026_ytd], width=0.66, color=C["petrol_light"], zorder=3)
axL.bar([x26], [n2026_proj - n2026_ytd], bottom=[n2026_ytd], width=0.66,
        facecolor="none", edgecolor=C["petrol_light"], hatch="////", lw=1.1, zorder=3)
axL.annotate(f"{n2026_proj / 1000:.1f}k", (x26, n2026_proj), xytext=(0, 7), textcoords="offset points",
             ha="center", fontsize=18, fontweight="bold", color=C["ink_3"])
axL.annotate("YTD", (x26, n2026_ytd / 2), ha="center", va="center",
             fontsize=16, fontweight="bold", color="white")

axL.set_xticks(xs + [x26])
axL.set_xticklabels(["'21", "'22", "'23", "'24", "'25", "'26"], fontsize=18, color=C["ink_2"])
axL.set_ylim(0, n2026_proj * 1.18)
axL.set_yticks([])
for s in ("left", "top", "right"):
    axL.spines[s].set_visible(False)
axL.spines["bottom"].set_color(C["rule"])
axL.tick_params(axis="x", length=0, pad=8)
axL.text(0.0, 1.05, "Disclosed AI activities per year", transform=axL.transAxes, **TITLE)
axL.annotate(f"{ratio_5y:.1f}×", (1.0, vals[-1] * 0.72), ha="center", va="center",
             fontsize=40, fontweight="bold", color=C["rust"])
axL.annotate("2021 → 2025", (1.0, vals[-1] * 0.72), xytext=(0, -34),
             textcoords="offset points", ha="center", va="center", fontsize=17, color=C["ink_3"])

# ---- Middle: mean firm-level G ----
gy = list(g_mean.index)
g_solid = [y for y in gy if y <= 2025]
axM.plot(g_solid, [g_mean[y] for y in g_solid], color=C["rust"], lw=3.2, marker="o", markersize=8,
         markeredgecolor=C["bg"], markeredgewidth=1.4, zorder=3)
axM.plot([2025, 2026], [g_mean[2025], g_mean[2026]], color=C["rust"], lw=2.6, linestyle=(0, (2, 2)), zorder=3)
axM.plot([2026], [g_mean[2026]], marker="o", markersize=8, markerfacecolor=C["bg"],
         markeredgecolor=C["rust"], markeredgewidth=2.0, zorder=4)
for y, va, dy in ((2021, "top", -16), (2025, "top", -16)):
    axM.annotate(f"{g_mean[y]:.2f}", (y, g_mean[y]), xytext=(0, dy), textcoords="offset points",
                 ha="center", va=va, fontsize=24, fontweight="bold", color=C["rust"])
axM.set_ylim(0.25, 0.55)
axM.set_yticks([0.3, 0.4, 0.5])
axM.set_yticklabels(["0.30", "0.40", "0.50"], fontsize=18, color=C["ink_3"])
axM.set_xlim(2020.6, 2026.4)
axM.set_xticks([2021, 2023, 2025])
axM.set_xticklabels(["'21", "'23", "'25"], fontsize=18, color=C["ink_2"])
axM.yaxis.grid(True, color=C["grid"], lw=1.0, zorder=0)
for s in ("left", "top", "right", "bottom"):
    axM.spines[s].set_visible(False)
axM.tick_params(axis="both", length=0, pad=8)
axM.text(0.0, 1.05, "Mean G per firm (0 to 1)", transform=axM.transAxes, **TITLE)

# ---- Middle: independent grounding attributes (Thesis Fig. 13, right) ----
lad = read(results("washing", "ladder_substance.parquet")).iloc[0]
attrs = [("Business function", lad["p_func"]), ("Deployed or scaled", lad["p_stage"]),
         ("Named product or process", lad["p_named"]), ("Quantified outcome", lad["p_metric"]),
         ("Named external vendor", lad["p_vendor"])]
ys = list(range(len(attrs)))[::-1]
for y, (lab, p_) in zip(ys, attrs):
    col = C["petrol"] if p_ >= 50 else C["rust"]
    axR.barh([y], [p_], height=0.62, color=col, zorder=3)
    axR.text(-3, y, lab, ha="right", va="center", fontsize=17, color=C["ink"])
    axR.text(p_ + 2, y, f"{p_:.0f}%" if p_ >= 10 else f"{p_:.1f}%", ha="left", va="center", fontsize=18,
             fontweight="bold", color=col)
axR.set_xlim(0, 120)
axR.set_ylim(-0.6, len(attrs) - 0.4)
axR.set_xticks([]); axR.set_yticks([])
for s_ in ("left", "top", "right", "bottom"):
    axR.spines[s_].set_visible(False)
axR.text(-0.62, 1.05, "% of activities with each piece of evidence", transform=axR.transAxes, **TITLE)

save(f, "volume_grounding")
