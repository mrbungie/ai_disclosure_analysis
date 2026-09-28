"""Backup chart: talk (disclosure intensity) vs. substance (grounded
activity) percentile ranks in 2025, with the decoupling tails labeled."""
from style import C, fig, save, L

YEAR = 2025

df = L.read_dataset("firm_year", ("covariates", "washing_score"),
                     ("covariates", "disclosure_volume", ["frames_per_1k"])
                     ).dropna(subset=["frames_per_1k", "substance", "w"])
df = df[df["year"] == YEAR].copy()
df["pct_d"] = df["pct_disclosure"]
df["pct_s"] = df["pct_substance"]

q_hi, q_lo = df["w"].quantile(0.95), df["w"].quantile(0.05)
tail_hi = df["w"] >= q_hi
tail_lo = df["w"] <= q_lo
body = ~(tail_hi | tail_lo)

f = fig(525, 525)
ax = f.add_axes([0.14, 0.13, 0.84, 0.84])

ax.scatter(df.loc[body, "pct_d"], df.loc[body, "pct_s"], s=10, color=C["rule"],
           alpha=0.45, linewidths=0, zorder=2)
ax.scatter(df.loc[tail_hi, "pct_d"], df.loc[tail_hi, "pct_s"], s=18,
           facecolor=C["rust"], edgecolor="none", alpha=0.9, zorder=3)
ax.scatter(df.loc[tail_lo, "pct_d"], df.loc[tail_lo, "pct_s"], s=18,
           facecolor=C["petrol"], edgecolor="none", alpha=0.9, zorder=3)

ax.plot([0, 1], [0, 1], color=C["ink"], lw=1.4, ls=(0, (5, 4)), zorder=1)
ax.text(0.6, 0.655, "W = 0: talk = substance", rotation=41, ha="center",
        va="center", fontsize=13.5, color=C["ink_2"], style="italic",
        rotation_mode="anchor")

ax.text(0.97, 0.05, "Talk ahead of substance (W > 0)", ha="right", va="bottom",
        fontsize=14, fontweight="semibold", color=C["rust"])
ax.text(0.03, 0.95, "Substance ahead of talk (W < 0)", ha="left", va="top",
        fontsize=14, fontweight="semibold", color=C["petrol"])

labels_hi = [("WM", (16, 10), "left"), ("DHR", (14, -14), "left"),
             ("IT", (14, 10), "left"), ("FOX", (14, 14), "left")]
labels_lo = [("MS", (14, -14), "left"), ("AIG", (-14, 14), "right"),
             ("CVX", (-14, 12), "right"), ("COF", (14, 12), "left")]
labels_diag = [("NVDA", (-34, -18), "right")]

for ticker, offset, ha in labels_hi:
    row = df[df["ticker"] == ticker].iloc[0]
    ax.scatter([row["pct_d"]], [row["pct_s"]], s=34, facecolor=C["rust"],
               edgecolor=C["ink"], lw=1.0, zorder=6)
    ax.annotate(ticker, xy=(row["pct_d"], row["pct_s"]), xytext=offset,
                textcoords="offset points", fontsize=14, fontweight="bold",
                color=C["rust"], ha=ha, va="center")

for ticker, offset, ha in labels_lo:
    row = df[df["ticker"] == ticker].iloc[0]
    ax.scatter([row["pct_d"]], [row["pct_s"]], s=34, facecolor=C["petrol"],
               edgecolor=C["ink"], lw=1.0, zorder=6)
    ax.annotate(ticker, xy=(row["pct_d"], row["pct_s"]), xytext=offset,
                textcoords="offset points", fontsize=14, fontweight="bold",
                color=C["petrol"], ha=ha, va="center")

for ticker, offset, ha in labels_diag:
    row = df[df["ticker"] == ticker].iloc[0]
    ax.scatter([row["pct_d"]], [row["pct_s"]], s=34, facecolor=C["ink"],
               edgecolor=C["bg"], lw=1.0, zorder=6)
    ax.annotate(ticker, xy=(row["pct_d"], row["pct_s"]), xytext=offset,
                textcoords="offset points", fontsize=14, fontweight="bold",
                color=C["ink"], ha=ha, va="center")

ax.set_xlim(-0.03, 1.03)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel(f"Disclosure intensity D (percentile, {YEAR})", fontsize=14,
              color=C["ink_2"])
ax.set_ylabel(f"Disclosed substance S (percentile, {YEAR})", fontsize=14,
              color=C["ink_2"])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.grid(color=C["grid"], lw=0.9, zorder=0)
ax.set_aspect("equal")
ax.tick_params(labelsize=13.5)

save(f, "w_scatter")
