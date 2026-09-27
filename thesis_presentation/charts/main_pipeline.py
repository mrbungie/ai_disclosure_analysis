"""Methodology diagram: sources -> text processing -> LLM extraction ->
constructs -> analyses, in the deck's design system. Coordinates are display
pixels (1720 x 640 on the slide); text sizes are points (1 pt = 1.333 px)."""
from matplotlib.patches import FancyBboxPatch

from style import C, fig, save, read, results_json, gold

W, H = 1750, 660

# Numbers quoted on the diagram, read from the same artefacts as the thesis.
fun = results_json("prefilter", "funnel.json")["funnel"]
hc = results_json("corpus", "headline_counts.json")
n_act = len(read(gold("spines", "activity", "activity"), columns=["ticker"]))
n_firms_act = read(gold("spines", "activity", "activity"), columns=["ticker"])["ticker"].nunique()

f = fig(W * 0.75, H * 0.75)
ax = f.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")

PAPER, LINE, SAND = "#FFFFFF", "#E4DED3", "#F1ECE3"


def box(x, y, w, h, title, lines=(), accent=None, fill=PAPER, edge=LINE, tcol=None,
        scol=None, dashed=False, big=None, bigcol=None):
    if accent:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                                    fc=accent, ec="none", zorder=2))
        ax.add_patch(FancyBboxPatch((x + 1, y + 5), w - 2, h - 6, boxstyle="round,pad=0,rounding_size=13",
                                    fc=fill, ec="none", zorder=2))
    else:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=14",
                                    fc=fill, ec=edge, lw=1.4, ls=(0, (4, 3)) if dashed else "-", zorder=2))
    ty = y + 30
    ax.text(x + 18, ty, title, ha="left", va="center", fontsize=13.5, fontweight="semibold",
            color=tcol or C["ink"], zorder=4)
    if big:
        ty += 36
        ax.text(x + 18, ty, big, ha="left", va="center", fontsize=21, fontweight="semibold",
                color=bigcol or C["petrol"], zorder=4)
        ty += 6
    for i, ln in enumerate(lines):
        ax.text(x + 18, ty + 26 + 21 * i, ln, ha="left", va="center", fontsize=11.2,
                color=scol or C["ink_3"], zorder=4)


def arrow(pts, color=None, lw=1.8):
    color = color or C["rule"]
    xs, ys = zip(*pts)
    ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=lw, zorder=1,
            solid_capstyle="butt", solid_joinstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.35", color=color,
                                lw=lw, shrinkA=0, shrinkB=0), zorder=1)


def stage_label(x, text):
    ax.text(x, 14, text, ha="left", va="center", fontsize=10.5, fontweight="semibold",
            color=C["ink_3"])


# ---- column 0: universe --------------------------------------------------
stage_label(0, "UNIVERSE")
box(0, 205, 190, 150, "S&P 500", ["fixed at 1 Jan 2021", f"{hc['hd_n_universe']} firms", "2021 – 2026"],
    fill=SAND, edge=SAND)

# ---- column 1: sources ---------------------------------------------------
stage_label(240, "SOURCES")
box(240, 40, 270, 170, "Filings & earnings calls",
    ["10-K · 10-Q · DEF 14A · 8-K · calls", f"{fun['n_raw'] / 1e6:.1f}M paragraphs"],
    accent=C["petrol"], big=f"{hc['hd_n_docs']:,} docs")
box(240, 400, 270, 105, "Returns & fundamentals", ["daily prices · beta · CAR", "quarterly financials"],
    fill=SAND, edge=SAND)
box(240, 525, 270, 105, "AI patents", ["OECD 2025 taxonomy", "external benchmark"], fill=SAND, edge=SAND)
arrow([(190, 280), (215, 280), (215, 125), (238, 125)])
arrow([(190, 290), (215, 290), (215, 452), (238, 452)])
arrow([(215, 452), (215, 577), (238, 577)])

# ---- column 2: text processing stack -------------------------------------
stage_label(560, "SCREEN")
box(560, 40, 280, 110, "Deduplicate", ["content hash · each text scored once"],
    accent=C["petrol"], big=f"{fun['n_unique'] / 1e6:.1f}M unique")
box(560, 175, 280, 90, "Screening features", ["39 signals + bge-m3 similarity"], accent=C["petrol"])
box(560, 290, 280, 110, "Gradient-boosted filter", ["recall-first · recall ≈ 0.97"],
    accent=C["petrol"], big=f"{fun['n_prefiltered']:,} AI texts")
box(560, 450, 280, 90, "Labelled benchmarks", ["training labels · held-out audits"],
    fill="#FBF3EF", edge=C["rust_light"], tcol=C["rust"], dashed=True)
arrow([(510, 95), (558, 95)])
arrow([(700, 150), (700, 173)])
arrow([(700, 265), (700, 288)])
arrow([(700, 450), (700, 402)], color=C["rust_light"])

# ---- column 3: LLM extraction ---------------------------------------------
stage_label(890, "EXTRACT · LLM")
box(890, 40, 290, 150, "Pass 1 · claims", ["zero-shot, 7 fields + sentence anchor", "→ disclosure D, posture P"],
    accent=C["teal"], big=f"{fun['n_frames_total']:,} frames", bigcol=C["teal"])
box(890, 250, 290, 150, "Pass 2 · activities", ["verb · AI object · function, stage,", f"evidence → A, G, S  ·  {n_firms_act} firms"],
    accent=C["teal"], big=f"{n_act:,} activities", bigcol=C["teal"])
arrow([(840, 345), (865, 345), (865, 115), (888, 115)])
arrow([(1035, 190), (1035, 248)], color=C["teal"])
ax.text(1047, 219, "when the firm acts", ha="left", va="center", fontsize=10.5, style="italic",
        color=C["teal"])

# ---- column 4: fusion ------------------------------------------------------
stage_label(1230, "MEASURE")
ax.add_patch(FancyBboxPatch((1230, 40), 220, 400, boxstyle="round,pad=0,rounding_size=14",
                            fc=C["petrol_dark"], ec="none", zorder=2))
ax.text(1250, 72, "Point-in-time fusion", fontsize=13.5, fontweight="semibold", color="#FFFFFF",
        va="center", zorder=4)
toks = [("D", C["petrol"]), ("P", C["petrol"]), ("A", C["teal"]), ("G", C["teal"]),
        ("S", C["teal"]), ("W", C["rust"])]
for i, (t, col) in enumerate(toks):
    cx, cy = 1262 + (i % 3) * 62, 118 + (i // 3) * 62
    ax.add_patch(FancyBboxPatch((cx, cy), 50, 50, boxstyle="round,pad=0,rounding_size=10",
                                fc=col, ec="none", zorder=3))
    ax.text(cx + 25, cy + 26, t, ha="center", va="center", fontsize=19, style="italic",
            fontweight="semibold", color="#FFFFFF", zorder=4, family="Source Serif 4 Display")
for i, ln in enumerate(["firm-year panel", "earnings-call panel", "call vs. filing cells",
                        "as-of joins: no look-ahead"]):
    ax.text(1250, 270 + 30 * i, ln, fontsize=11.5, color="#C9D6DA", va="center", zorder=4)
arrow([(1180, 115), (1228, 115)], color=C["teal"])
arrow([(1180, 325), (1228, 325)], color=C["teal"])
arrow([(510, 452), (540, 452), (540, 600), (1340, 600), (1340, 442)])
arrow([(510, 577), (540, 577)])

# ---- column 5: analyses ----------------------------------------------------
stage_label(1500, "ANSWER")
rqs = [("RQ1 · Evolution", "D, A, G over time", C["petrol"]),
       ("RQ2 · Heterogeneity", "posture P → archetypes", "#2E7A7A"),
       ("RQ3 · Credibility", "W; calls vs. filings", "#9C7A3A"),
       ("RQ4 · Economic information", "W, D, P → risk, returns", C["rust"])]
for i, (t, s, col) in enumerate(rqs):
    y = 40 + i * 112
    ax.add_patch(FancyBboxPatch((1500, y), 250, 96, boxstyle="round,pad=0,rounding_size=12",
                                fc=col, ec="none", zorder=2))
    ax.text(1516, y + 32, t, fontsize=12.3, fontweight="semibold", color="#FFFFFF", va="center", zorder=4)
    ax.text(1516, y + 64, s, fontsize=11, color="#FFFFFF", alpha=0.85, va="center", zorder=4)
    arrow([(1450, 260), (1472, 260), (1472, y + 48), (1498, y + 48)], color=C["rule"])

save(f, "pipeline")
