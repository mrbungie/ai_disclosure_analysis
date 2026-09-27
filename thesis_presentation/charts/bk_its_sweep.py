"""Backup chart: break-date sweep for the risk and governance frame series --
refitting the interrupted-time-series break at every candidate month shows
the best-fitting break predates the SEC's December 2023 warning."""
import pandas as pd

from style import C, fig, save, results, read

df = read(results("posture", "interrupted_series_break_sweep.csv"))
print(df.columns.tolist())

df["x"] = pd.PeriodIndex(df["event"], freq="M").to_timestamp()
sec_date = pd.Timestamp("2023-12-01")

SERIES = [("Risk frames", C["petrol"]), ("Governance frames", C["ochre"])]

f = fig(1300, 520)
ax = f.add_axes([0.07, 0.15, 0.88, 0.76])

for name, color in SERIES:
    s = df[df["outcome"] == name].sort_values("x")
    ax.plot(s["x"], s["r2"], "-", color=color, lw=2.2, zorder=3)
    best = s.loc[s["r2"].idxmax()]
    ax.scatter([best["x"]], [best["r2"]], s=80, color=color, edgecolor=C["bg"],
               linewidths=1.2, zorder=4)
    ax.annotate(f"{name}\nbest fit {best['event']}", xy=(best["x"], best["r2"]),
                xytext=(0, 14), textcoords="offset points", ha="center",
                fontsize=13, fontweight="semibold", color=color)

ax.axvline(sec_date, color=C["rust"], ls=(0, (1, 2)), lw=1.6, zorder=2)
ax.annotate("SEC warning (Dec 2023)", xy=(sec_date, ax.get_ylim()[0]),
            xytext=(8, 10), textcoords="offset points", fontsize=13,
            color=C["rust"], fontweight="semibold")

ax.set_ylabel("Model fit (R²)", fontsize=15, color=C["ink_2"])
ax.grid(axis="y", color=C["grid"], lw=0.9, zorder=0)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.margins(x=0.03)

save(f, "its_sweep")
