"""Chart 5: activity volume growth (with 2026 projection) and grounding trend."""
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
_dp["md"] = _dp["fecha"].dt.month * 100 + _dp["fecha"].dt.day
hd_cutoff = _dp.loc[_dp["year"] == 2026, "fecha"].max()
hd_cutoff_md = int(hd_cutoff.month * 100 + hd_cutoff.day)
hd_cutoff_str = hd_cutoff.strftime("%-d %B")

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

f = fig(1640, 560)
axL = f.add_axes([0.045, 0.13, 0.42, 0.8])
axR = f.add_axes([0.565, 0.13, 0.42, 0.8])

# ---- Left panel: activity volume bars ----
years = [2021, 2022, 2023, 2024, 2025]
vals = [df.loc[y, "n_activities"] for y in years]
xs = list(range(len(years)))
axL.bar(xs, vals, width=0.62, color=C["petrol"], zorder=3)
for x, v in zip(xs, vals):
    axL.annotate(f"{v:,.0f}", (x, v), xytext=(0, 8), textcoords="offset points",
                 ha="center", fontsize=13, fontweight="bold", color=C["petrol_dark"])

x26 = len(years)
axL.bar([x26], [n2026_ytd], width=0.62, color=C["petrol_light"], zorder=3)
axL.bar([x26], [n2026_proj - n2026_ytd], bottom=[n2026_ytd], width=0.62,
        facecolor="none", edgecolor=C["petrol_light"], hatch="////", lw=1.1, zorder=3)
axL.annotate(f"{n2026_ytd:,.0f}", (x26, n2026_ytd / 2), ha="center", va="center",
             fontsize=11.5, fontweight="bold", color="white")
axL.annotate(f"{n2026_proj:,.0f}", (x26, n2026_proj), xytext=(0, 8), textcoords="offset points",
             ha="center", fontsize=13, fontweight="bold", color=C["ink_2"])
axL.annotate("2026 YTD", (x26, n2026_ytd), xytext=(0, -18), textcoords="offset points",
             ha="center", fontsize=10.5, color="white")
axL.annotate("projected", (x26, n2026_proj - (n2026_proj - n2026_ytd) / 2), xytext=(0, 0),
             textcoords="offset points", ha="center", va="center", fontsize=10.5,
             color=C["ink_3"], rotation=90)

axL.set_xticks(xs + [x26])
axL.set_xticklabels([str(y) for y in years] + ["2026"], fontsize=13, color=C["ink_2"])
axL.set_ylim(0, n2026_proj * 1.2)
axL.set_yticks([])
for s in ("left", "top", "right"):
    axL.spines[s].set_visible(False)
axL.spines["bottom"].set_color(C["rule"])
axL.tick_params(axis="x", length=0, pad=8)
axL.text(0.0, 1.06, "Disclosed AI activities per year", transform=axL.transAxes,
         fontsize=13.5, color=C["ink_3"], ha="left", va="bottom")
axL.annotate(f"{ratio_5y:.1f}×", (1.3, vals[-1] * 0.62), ha="center", va="center",
             fontsize=30, fontweight="bold", color=C["rust"])
axL.annotate("2021 → 2025", (1.3, vals[-1] * 0.62), xytext=(0, -30),
             textcoords="offset points", ha="center", va="center", fontsize=11.5, color=C["ink_3"])

# ---- Right panel: grounding trend lines ----
series = [
    ("Deployed or scaled", "deployed_share_pct", C["petrol"]),
    ("Named product or process", "named_evidence_pct", C["rust"]),
]
yrs_all = list(df.index)
yrs_solid = [y for y in yrs_all if y <= 2025]
yrs_dot = [y for y in yrs_all if y >= 2025]

for name, col, color in series:
    v_solid = [df.loc[y, col] for y in yrs_solid]
    v_dot = [df.loc[y, col] for y in yrs_dot]
    axR.plot(yrs_solid, v_solid, color=color, lw=2.4, marker="o", markersize=5.5,
             markerfacecolor=color, markeredgecolor=C["bg"], markeredgewidth=1.1, zorder=3)
    axR.plot(yrs_dot, v_dot, color=color, lw=2.0, linestyle=(0, (2, 2)), zorder=3)
    axR.plot([yrs_dot[-1]], [v_dot[-1]], marker="o", markersize=5.5, markerfacecolor=C["bg"],
             markeredgecolor=color, markeredgewidth=1.6, zorder=4)
    start_v = df.loc[2021, col]
    end_v = df.loc[2025, col]
    axR.annotate(f"{name}  {start_v:.0f}% → {end_v:.0f}%",
                 (yrs_all[-1], end_v), xytext=(6, 0 if name == series[0][0] else 0),
                 textcoords="offset points", ha="left", va="center",
                 fontsize=12.5, fontweight="bold", color=color)

axR.set_xlim(yrs_all[0] - 0.3, yrs_all[-1] + 2.4)
axR.set_ylim(0, 100)
axR.set_yticks([0, 25, 50, 75, 100])
axR.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=12, color=C["ink_3"])
axR.set_xticks(yrs_all)
axR.set_xticklabels([str(y) for y in yrs_all], fontsize=13, color=C["ink_2"])
axR.yaxis.grid(True, color=C["grid"], lw=1.0, zorder=0)
for s in ("left", "top", "right", "bottom"):
    axR.spines[s].set_visible(False)
axR.tick_params(axis="both", length=0, pad=8)
axR.text(0.0, 1.06, "Grounding of disclosed AI activities", transform=axR.transAxes,
         fontsize=13.5, color=C["ink_3"], ha="left", va="bottom")

save(f, "volume_grounding")
