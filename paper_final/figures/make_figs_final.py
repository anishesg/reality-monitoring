"""Final, minimal Figures 1 and 3 (same data; parsers from make_figs.py).
Run from paper_final/:  /usr/local/bin/python3 figures/make_figs_final.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figs import parse_out_b, parse_out_h, parse_base, save

INK, MUTED, GRID = "#1b1f24", "#8a9099", "#eceef1"
NAVY, CORAL, GOLD, TEAL, SKY = "#2b5aa8", "#e0605e", "#e8a33d", "#2a9d8f", "#e6eef8"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "xtick.major.size": 2.5, "ytick.major.size": 0, "xtick.major.width": 0.6,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})


def title(ax, letter, text):
    ax.set_title(f"$\\bf{{{letter}}}$   {text}", loc="left", fontsize=8.6, color=INK, pad=7)


def figure1():
    auroc = [("Qwen2.5-7B", .67, .66, .68), ("Llama-3.1-8B", .69, .70, .69), ("OLMo-2-7B", .59, .65, .71),
             ("Qwen2.5-14B", .65, .74, .69), ("Qwen3-8B", .60, .78, .68), ("R1-Distill-7B", .62, .71, .60)]
    base = [("Qwen\n7B", parse_base("out_e_qwen7b.txt")), ("Llama\n8B", parse_base("out_e_llama.txt")),
            ("Qwen\n14B", parse_base("out_base_q14.txt"))]
    H = parse_out_h()
    fig, axes = plt.subplots(1, 3, figsize=(5.2, 2.6), gridspec_kw={"width_ratios": [0.92, 1.18, 1.1], "wspace": 0.8})

    # (a) they know
    ax = axes[0]
    ys = np.arange(len(auroc))[::-1]
    for y, (n, v, p, b) in zip(ys, auroc):
        lo, hi = min(v, p, b), max(v, p, b)
        ax.plot([lo, hi], [y, y], color="#c9d6ec", lw=4, solid_capstyle="round", zorder=1)
        ax.plot(hi, y, "o", ms=4.5, color=NAVY, zorder=2)
    ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_yticks(ys); ax.set_yticklabels([a[0] for a in auroc], color=INK)
    ax.set_xlim(0.47, 0.82); ax.set_xticks([0.5, 0.6, 0.7, 0.8]); ax.set_ylim(-0.6, len(auroc) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("AUROC for own correctness")
    title(ax, "a", "They know")

    # (b) they don't act
    ax = axes[1]
    x = np.arange(len(base)) * 1.45
    for xi, (n, d) in zip(x, base):
        ax.plot([xi, xi], [d["init"], d["none"]], color="#f3c1c0", lw=2.2, zorder=1, solid_capstyle="round")
        ax.plot(xi, d["init"], "o", ms=5, color="white", mec=INK, mew=1.1, zorder=3)
        ax.plot(xi, d["none"], "o", ms=5.5, color=CORAL, zorder=3)
        ax.plot(xi + 0.25, d["oracle"], "D", ms=4.2, color=TEAL, zorder=4)
    d0 = base[0][1]
    ax.text(x[0] - 0.14, d0["init"], "before", fontsize=7, color=INK, ha="right", va="center")
    ax.text(x[0] - 0.14, d0["none"] + 0.02, "after", fontsize=7, color=CORAL, ha="right", va="center")
    ax.text(x[2] + 0.28, base[2][1]["oracle"] - 0.08, "perfect\nnote", fontsize=7, color=TEAL, ha="center", va="top", linespacing=0.95)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in base], color=INK, fontsize=7); ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.95, 3.25); ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1.0])
    for yv in (0.5, 1.0): ax.axhline(yv, color=GRID, lw=0.6, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylabel("accuracy")
    title(ax, "b", "They don't act")

    # (c) training makes them act
    ax = axes[2]
    ax.axvspan(0.50, 0.68, color=SKY, lw=0, zorder=0)
    ax.text(0.59, 0.30, "own\nself-report", ha="center", va="bottom", fontsize=7, color=NAVY)
    rkeys = [k for k in ("rule_s0", "rule_s1", "rule_s2") if k in H]
    rarr = np.array([[s[:2] for s in H[k]["syn"]] for k in rkeys])
    own = np.array([[s[:2] for s in H[k]["syn"]] for k in ("own_s0", "own_s1", "own_s2")])
    best = np.array([(s[0], s[2]) for s in H["rule_s0"]["syn"]])
    unt = np.array([(s[0], s[1]) for s in H["base"]["syn"]])
    for yv in (0.25, 0.5, 0.75, 1.0): ax.axhline(yv, color=GRID, lw=0.6, zorder=0)
    ax.plot(own[:, :, 0].mean(0), own[:, :, 1].mean(0), color=GOLD, lw=1.6, zorder=3)
    ax.plot(rarr[:, :, 0].mean(0), rarr[:, :, 1].mean(0), color=NAVY, lw=2.0, zorder=4)
    ax.plot(best[:, 0], best[:, 1], color=INK, lw=0.9, ls=(0, (3, 2)), zorder=5)
    ax.plot(unt[:, 0], unt[:, 1], color=CORAL, lw=2.0, zorder=3)
    R = 1.01
    ax.text(R, 0.985, "rule-\ntrained", fontsize=7, color=NAVY, va="center", linespacing=0.95)
    ax.text(R, 0.83, "own-\nconfidence", fontsize=7, color="#b87a12", va="center", linespacing=0.95)
    ax.text(R, unt[-1, 1], "untrained", fontsize=7, color=CORAL, va="center")
    ax.text(0.73, 0.905, "best possible", fontsize=7, color=INK, ha="center")
    ax.set_xlim(0.48, 1.0); ax.set_ylim(0.15, 1.04); ax.set_xticks([0.5, 0.75, 1.0])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("AUROC of confidence note"); ax.set_ylabel("accuracy after challenge", labelpad=2)
    title(ax, "c", "Training makes them act")
    save(fig, "fig1_overview")


def figure3():
    B = parse_out_b()
    rows = [("Llama-3.1-8B", "llama8b"), ("Qwen2.5-7B", "qwen7b"), ("OLMo-2-7B", "olmo"), None,
            ("Qwen2.5-14B", "qwen14b_v2"), ("Qwen2.5-32B", "qwen32b_awq_v2"),
            ("Qwen3-8B", "qwen3_8b_nothink_v2"), ("Qwen3-8B, thinking", "qwen3_8b"), None,
            ("R1-Distill-7B*", "r1_7b")]
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.45), sharey=True, gridspec_kw={"wspace": 0.38})
    ypos, y = [], 0.0
    for r in rows:
        if r is None: y += 0.55; continue
        ypos.append((y, r)); y += 1
    for ax, cue, lab in ((axes[0], "counter", "strong challenge"), (axes[1], "weak", "tentative suggestion")):
        for xv in (0.25, 0.5, 0.75): ax.axvline(xv, color=GRID, lw=0.6, zorder=0)
        for yy, (name, key) in ypos:
            c = B[key][cue]; near, far, D = c["FFnear"], c["FFfar"], c["D"]
            faded = key == "r1_7b"; a = 0.35 if faded else 1.0
            sig = ((D[1] > 0) or (D[2] < 0)) and not faded
            ax.plot([far[0], near[0]], [yy, yy], color=NAVY if sig else "#cfd4da", lw=2.0, zorder=1, alpha=a, solid_capstyle="round")
            ax.plot(near[0], yy, "o", ms=5, color=CORAL, zorder=3, alpha=a)
            ax.plot(far[0], yy, "o", ms=5, color="white", mec=NAVY, mew=1.3, zorder=4, alpha=a)
            ax.text(1.05, yy, f"{D[0]:+.2f}".replace("-", "−"), fontsize=7.2, va="center", ha="left", clip_on=False,
                    color=NAVY if sig else MUTED, fontweight="bold" if sig else "normal")
        ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(f"switch rate, {lab}")
    axes[0].set_yticks([p[0] for p in ypos]); axes[0].set_yticklabels([p[1][0] for p in ypos], color=INK)
    axes[0].invert_yaxis()
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=CORAL, ms=5, label="suggestion = model's 2nd choice"),
                        plt.Line2D([], [], marker="o", ls="", color="white", mec=NAVY, mew=1.3, ms=5, label="suggestion = its last choice")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.06), fontsize=7.5, handletextpad=0.2, columnspacing=1.4)
    save(fig, "fig3_wrong_for_wrong")


if __name__ == "__main__":
    figure1(); figure3(); print("ok")
