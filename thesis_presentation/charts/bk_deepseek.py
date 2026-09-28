"""Backup chart: open-source AI vocabulary on earnings calls around the
January 2025 DeepSeek-R1 release, interrupted time series."""
import pandas as pd
import matplotlib.dates as mdates

from style import C, fig, save, results, results_json, read

ds = read(results("posture", "deepseek_calls_series.csv"))
ds["x"] = pd.PeriodIndex(ds["month"], freq="M").to_timestamp()
dsj = results_json("posture", "deepseek_calls_its.json")
print("p_step =", dsj["p_step"])

RATE = 1e2  # per 100k words
for c in ["rate_per_1k", "trend", "trend_low", "trend_high", "counterfactual"]:
    ds[c] = ds[c] * RATE

brk = pd.Timestamp("2025-01-01")

f = fig(1290, 420)
ax = f.add_axes([0.08, 0.12, 0.9, 0.8])

ax.plot(ds["x"], ds["rate_per_1k"], "o", ms=4.5, color=C["rule"], alpha=0.65, zorder=2)

pre = ds[ds["post"] == 0]
ax.plot(pre["x"], pre["trend"], "-", lw=2.4, color=C["petrol"], zorder=3)
ax.fill_between(pre["x"], pre["trend_low"], pre["trend_high"], color=C["petrol"],
                 alpha=0.14, lw=0, zorder=1)

post = ds[ds["post"] == 1]
ax.plot(post["x"], post["trend"], "-", lw=2.4, color=C["rust"], zorder=3)
ax.fill_between(post["x"], post["trend_low"], post["trend_high"], color=C["rust"],
                 alpha=0.14, lw=0, zorder=1)

ax.plot(post["x"], post["counterfactual"], "--", lw=1.8, color=C["petrol"],
        alpha=0.6, zorder=3)

ax.axvline(brk, color=C["ink_2"], ls=(0, (1, 2)), lw=1.4, zorder=2)
ax.set_ylim(-0.15, ds["rate_per_1k"].max() * 1.18)

ev_row = ds[ds["month"] == "2025-01"].iloc[0]
fb_row = ds[ds["month"] == "2025-02"].iloc[0]

ax.annotate("DeepSeek-R1 · 20 Jan 2025", xy=(brk, ev_row["rate_per_1k"]),
            xytext=(14, 18), textcoords="offset points", fontsize=14,
            fontweight="semibold", color=C["ink"],
            arrowprops=dict(arrowstyle="->", color=C["ink_2"], lw=1.0,
                             shrinkA=0, shrinkB=5))

ax.annotate("two-month pulse", xy=(fb_row["x"], fb_row["rate_per_1k"]),
            xytext=(60, 6), textcoords="offset points", fontsize=14,
            color=C["rust"], fontweight="semibold",
            arrowprops=dict(arrowstyle="->", color=C["rust"], lw=1.0,
                             shrinkA=0, shrinkB=5))

late_post = post.iloc[len(post) // 2]
ax.annotate("no lasting level shift", xy=(late_post["x"], late_post["trend"]),
            xytext=(0, -64), textcoords="offset points", fontsize=14,
            color=C["rust"], fontweight="semibold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C["rust"], lw=1.0,
                             shrinkA=0, shrinkB=5))

ax.set_ylabel("Open-source AI mentions\nper 100k call words", fontsize=14,
              color=C["ink_2"])
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.margins(x=0.015)
ax.grid(axis="y", color=C["grid"], lw=0.9, zorder=0)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.tick_params(labelsize=14)

save(f, "deepseek")
