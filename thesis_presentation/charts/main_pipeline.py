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

PAPER, LINE, SAND = "#FFFFFF", "#E3E6E9", "#F1F2F4"
T_TITLE, T_BIG, T_SUB, T_STAGE = 15.5, 22, 14, 16.5   # pt; x1.333 = px on the slide


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
    ax.text(x + 16, ty, title, ha="left", va="center", fontsize=T_TITLE, fontweight="semibold",
            color=tcol or C["ink"], zorder=4)
    if big:
        ty += 36
        ax.text(x + 16, ty, big, ha="left", va="center", fontsize=T_BIG, fontweight="semibold",
                color=bigcol or C["petrol"], zorder=4)
        ty += 6
    for i, ln in enumerate(lines):
        ax.text(x + 16, ty + 27 + 25 * i, ln, ha="left", va="center", fontsize=T_SUB,
                color=scol or C["ink_3"], zorder=4)


def arrow(pts, color=None, lw=1.8):
    color = color or C["rule"]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=color, lw=lw, zorder=1, solid_capstyle="butt", solid_joinstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>,head_length=0.7,head_width=0.35", color=color,
                                lw=lw, shrinkA=0, shrinkB=0), zorder=1)


def stage_label(x, text):
    ax.text(x, 12, text, ha="left", va="center", fontsize=T_STAGE, fontweight="semibold",
            color=C["ink_3"])


# ---- column 1: sources (S&P 500 universe fixed at 1 Jan 2021) --------------
X1, W1 = 0, 310
stage_label(X1, "SOURCES")
box(X1, 40, W1, 175, "Filings & earnings calls",
    [f"S&P 500 · {hc['hd_n_universe']} firms", "2021 – 2026", f"{fun['n_raw'] / 1e6:.1f}M paragraphs"],
    accent=C["petrol"], big=f"{hc['hd_n_docs']:,} docs")
box(X1, 400, W1, 105, "Returns & fundamentals", ["daily prices · beta · CAR", "quarterly financials"],
    fill=SAND, edge=SAND)
box(X1, 525, W1, 105, "AI patents", ["OECD 2025 taxonomy", "external benchmark"], fill=SAND, edge=SAND)

# ---- column 2: text processing stack -------------------------------------
X2, W2 = 355, 320
stage_label(X2, "SCREEN")
box(X2, 40, W2, 120, "Deduplicate", ["each text scored once"],
    accent=C["petrol"], big=f"{fun['n_unique'] / 1e6:.1f}M unique")
box(X2, 185, W2, 80, "Screening features", ["39 signals + bge-m3"], accent=C["petrol"])
box(X2, 290, W2, 120, "Gradient-boosted filter", ["recall-first · recall ≈ 0.97"],
    accent=C["petrol"], big=f"{fun['n_prefiltered']:,} AI texts")
box(X2, 450, W2, 80, "Labelled benchmarks", ["training labels · audits"],
    fill="#F7ECE8", edge=C["rust_light"], tcol=C["rust"], dashed=True)
arrow([(W1, 95), (X2 - 2, 95)])
c2 = X2 + W2 / 2
arrow([(c2, 160), (c2, 183)])
arrow([(c2, 265), (c2, 288)])
arrow([(c2, 450), (c2, 412)], color=C["rust_light"])

# ---- column 3: LLM extraction ---------------------------------------------
X3, W3 = 720, 340
stage_label(X3, "EXTRACT · LLM")
box(X3, 40, W3, 150, "Pass 1 · claims", ["zero-shot, 7 fields + anchor", "→ disclosure D, posture P"],
    accent=C["teal"], big=f"{fun['n_frames_total']:,} frames", bigcol=C["teal"])
box(X3, 255, W3, 175, "Pass 2 · activities",
    ["verb · AI object · function,", "stage, evidence → A, G, S", f"{n_firms_act} firms"],
    accent=C["teal"], big=f"{n_act:,} activities", bigcol=C["teal"])
arrow([(X2 + W2, 350), (X2 + W2 + 22, 350), (X2 + W2 + 22, 115), (X3 - 2, 115)])
c3 = X3 + 150
arrow([(c3, 190), (c3, 253)], color=C["teal"])
ax.text(c3 + 12, 222, "if the firm reports acting", ha="left", va="center", fontsize=T_SUB, style="italic",
        color=C["teal"])

# ---- column 4: fusion ------------------------------------------------------
X4, W4 = 1105, 270
stage_label(X4, "MEASURE")
ax.add_patch(FancyBboxPatch((X4, 40), W4, 420, boxstyle="round,pad=0,rounding_size=14",
                            fc=C["petrol_dark"], ec="none", zorder=2))
ax.text(X4 + 18, 72, "Point-in-time fusion", fontsize=T_TITLE, fontweight="semibold", color="#FFFFFF",
        va="center", zorder=4)
toks = [("D", C["petrol"]), ("P", C["petrol"]), ("A", C["teal"]), ("G", C["teal"]),
        ("S", C["teal"]), ("W", C["rust"])]
for i, (t, col) in enumerate(toks):
    cx, cy = X4 + 30 + (i % 3) * 72, 108 + (i // 3) * 68
    ax.add_patch(FancyBboxPatch((cx, cy), 56, 56, boxstyle="round,pad=0,rounding_size=10",
                                fc=col, ec="none", zorder=3))
    ax.text(cx + 28, cy + 29, t, ha="center", va="center", fontsize=21, style="italic",
            fontweight="semibold", color="#FFFFFF", zorder=4, family="Source Serif 4 Display")
for i, ln in enumerate(["firm-year panel", "earnings-call panel", "call vs. filing cells",
                        "as-of joins,", "no look-ahead"]):
    ax.text(X4 + 18, 272 + 36 * i, ln, fontsize=T_SUB, color="#C9D6DA", va="center", zorder=4)
arrow([(X3 + W3, 115), (X4 - 2, 115)], color=C["teal"])
arrow([(X3 + W3, 340), (X4 - 2, 340)], color=C["teal"])
c4 = X4 + W4 / 2
arrow([(W1, 452), (W1 + 22, 452), (W1 + 22, 600), (c4, 600), (c4, 462)])
ax.plot([W1, W1 + 22], [577, 577], color=C["rule"], lw=1.8, zorder=1)

# ---- column 5: analyses ----------------------------------------------------
X5, W5 = 1420, 330
stage_label(X5, "EXPLORE")
rqs = [("RQ1 · Evolution", "Year-by-year trends", "D, A, G", C["petrol"]),
       ("RQ2 · Heterogeneity", "Archetypal analysis", "P → 3 postures", "#67958A"),
       ("RQ3 · Decoupling & venue", "Rank gap + firm-year FE", "W; calls vs. filings", "#B69A59"),
       ("RQ4 · Market relevance", "Regressions + BH-FDR", "W, D, P → markets", C["rust"])]
for i, (t, m_, s_, col) in enumerate(rqs):
    y = 30 + i * 152
    ax.add_patch(FancyBboxPatch((X5, y), W5, 138, boxstyle="round,pad=0,rounding_size=12",
                                fc=col, ec="none", zorder=2))
    ax.text(X5 + 16, y + 30, t, fontsize=T_TITLE, fontweight="semibold", color="#FFFFFF", va="center", zorder=4)
    ax.text(X5 + 16, y + 70, m_, fontsize=T_SUB, fontweight="bold", color="#FFFFFF", va="center", zorder=4)
    ax.text(X5 + 16, y + 106, s_, fontsize=T_SUB, color="#FFFFFF", alpha=0.85, va="center", zorder=4)
    arrow([(X4 + W4, 250), (X4 + W4 + 22, 250), (X4 + W4 + 22, y + 69), (X5 - 2, y + 69)], color=C["rule"])

save(f, "pipeline")
