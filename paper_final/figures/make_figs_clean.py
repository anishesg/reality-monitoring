"""Clean versions of Figure 1 and Figure 3 (same data as make_figs.py).
Run from paper_final/:  /usr/local/bin/python3 figures/make_figs_clean.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figs import parse_out_b, parse_out_h, parse_base, save  # data parsers only

INK, MUTED, LIGHT = "#1a1a1a", "#6b6b6b", "#d9d9d9"
BLUE, ORANGE, GREEN, VERM = "#0072B2", "#E69F00", "#009E73", "#D55E00"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.3, "ytick.labelsize": 8.3, "legend.fontsize": 8.3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})


def title(ax, letter, text):
    ax.set_title(f"$\\bf{{{letter}}}$  {text}", loc="left", fontsize=9.4, color=INK, pad=9)


def figure1():
    auroc = [("Qwen2.5-7B", .67, .66, .68), ("Llama-3.1-8B", .69, .70, .69), ("OLMo-2-7B", .59, .65, .71),
             ("Qwen2.5-14B", .65, .74, .69), ("Qwen3-8B", .60, .78, .68), ("R1-Distill-7B", .62, .71, .60)]
    base = {"Qwen\n7B": parse_base("out_e_qwen7b.txt"), "Llama\n8B": parse_base("out_e_llama.txt"),
            "Qwen\n14B": parse_base("out_base_q14.txt")}
    H = parse_out_h()

    fig = plt.figure(figsize=(5.5, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.0, 1.0], hspace=0.78, wspace=0.42)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]

    # (a) models know
    ax = axes[0]
    ys = np.arange(len(auroc))[::-1]
    for y, (n, v, p, b) in zip(ys, auroc):
        lo, hi = min(v, p, b), max(v, p, b)
        ax.plot([lo, hi], [y, y], color=LIGHT, lw=3.2, solid_capstyle="round", zorder=1)
        ax.plot(v, y, "o", ms=3.6, color=BLUE, zorder=3)
        ax.plot(p, y, "s", ms=3.2, color=ORANGE, zorder=3)
        ax.plot(b, y, "D", ms=3.0, color=GREEN, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([a[0] for a in auroc])
    ax.tick_params(axis="y", length=0)
    ax.plot([0.5, 0.5], [-1.1, len(auroc) - 0.45], color=MUTED, lw=0.6, ls=(0, (2, 2)))
    ax.set_xlim(0.44, 0.82); ax.set_xticks([0.5, 0.6, 0.7, 0.8]); ax.set_ylim(-1.0, len(auroc) + 0.9)
    ax.set_xlabel("AUROC for own correctness")
    ax.text(0.51, -0.45, "chance", fontsize=7, color=MUTED, va="center")
    for x, y0, t, c, m in ((0.795, 5.55, "stated", BLUE, "o"), (0.795, 4.95, "P(True)", ORANGE, "s"), (0.795, 4.35, "belief", GREEN, "D")):
        pass
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=3.6, label="stated"),
                       plt.Line2D([], [], marker="s", ls="", color=ORANGE, ms=3.2, label="P(True)"),
                       plt.Line2D([], [], marker="D", ls="", color=GREEN, ms=3.0, label="belief")],
              loc="upper right", ncol=3, frameon=False, handletextpad=0.05, columnspacing=0.45, borderaxespad=0.0,
              fontsize=7.3, bbox_to_anchor=(1.05, 1.0), handlelength=0.9)
    title(ax, "a", "Models know")

    # (b) they do not act on it: drop plot
    ax = axes[1]
    names = list(base); x = np.arange(len(names)) * 1.35
    for xi, n in zip(x, names):
        d = base[n]
        ax.annotate("", xy=(xi, d["none"] + 0.035), xytext=(xi, d["init"] - 0.03),
                    arrowprops=dict(arrowstyle="-|>", color=LIGHT, lw=1.6, mutation_scale=7))
        ax.plot(xi, d["init"], "o", ms=5, color=INK, mfc="white", mew=1.1, zorder=3)
        ax.plot(xi - 0.13, d["none"], "o", ms=5, color=MUTED, zorder=3)
        ax.plot(xi + 0.13, d["oracle"], "D", ms=4.6, color=GREEN, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.9, linespacing=1.0); ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 3.3); ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("accuracy")
    d0 = base[names[0]]
    ax.text(x[1], 0.97, "before challenge", fontsize=7.9, color=INK, va="center", ha="center")
    d2 = base[names[2]]
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=5, label="no note"),
                       plt.Line2D([], [], marker="D", ls="", color=GREEN, ms=4.6, label="perfect note")],
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8,
              handletextpad=0.1, columnspacing=1.0, borderaxespad=0.0, title=None)
    ax.text(x[1], 0.13, "after challenge", fontsize=7.9, color=MUTED, ha="center", va="center")
    title(ax, "b", "They ignore a perfect note")

    # (c) training makes them act, up to what they know
    ax = axes[2]
    ax.axvspan(0.50, 0.68, color="#eef3f8", lw=0, zorder=0)
    ax.text(0.59, 0.36, "range of models' own\nstated confidence", ha="center", va="bottom", fontsize=7.9, color=BLUE)
    best = H["rule_s0"]["syn"]
    ax.plot([s[0] for s in best], [s[2] for s in best], color=INK, lw=1.0, ls=(0, (3, 2)), zorder=4)
    rule = H["rule_s0"]["syn"]
    ax.plot([s[0] for s in rule], [s[1] for s in rule], color=BLUE, lw=1.8, zorder=3)
    own = np.array([[s[:2] for s in H[k]["syn"]] for k in ("own_s0", "own_s1", "own_s2")])  # seeds x 7 x (auroc, acc)
    ox, oy = own[:, :, 0].mean(0), own[:, :, 1]
    ax.fill_between(ox, oy.min(0), oy.max(0), color=ORANGE, alpha=0.18, lw=0, zorder=2)
    ax.plot(ox, oy.mean(0), color=ORANGE, lw=1.4, zorder=3)
    unt = H["base"]["syn"]
    ax.plot([s[0] for s in unt], [s[1] for s in unt], color=MUTED, lw=1.4, zorder=3)
    lp = H["rule_s0"]["real"]["letter"]
    ax.errorbar([lp[0]], [lp[1]], yerr=[[lp[1] - lp[2]], [lp[3] - lp[1]]], fmt="o", ms=3.4, color=BLUE, mfc="white", mew=1.0, capsize=1.5, lw=0.8, zorder=5)
    ax.annotate("rule-trained model given\na letter-probe note", (lp[0], lp[1]), xytext=(0.93, 0.58), fontsize=7.9, color=BLUE, ha="center",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5, shrinkA=0, shrinkB=3))
    # direct labels at the right edge
    ax.annotate("best possible (keep iff note ≥ 50%)", (0.80, 0.785), xytext=(0.70, 0.93), fontsize=7.9, color=INK, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.5, shrinkA=0, shrinkB=2))
    ax.text(1.005, rule[-1][1] + 0.025, "trained on rule", fontsize=7.9, color=BLUE, va="center")
    ax.text(1.005, oy.mean(0)[-1] - 0.075, "trained on own\nconfidence (3 seeds)", fontsize=7.9, color="#b07800", va="center")
    ax.text(1.005, unt[-1][1], "untrained", fontsize=7.9, color=MUTED, va="center")
    ax.set_xlim(0.48, 1.0); ax.set_ylim(0.15, 1.02); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("AUROC of the confidence note"); ax.set_ylabel("accuracy after challenge")
    title(ax, "c", "Training makes them act, up to what their confidence is worth")
    save(fig, "fig1_overview")


def figure3():
    B = parse_out_b()
    rows = [("Llama-3.1-8B", "llama8b"), ("Qwen2.5-7B", "qwen7b"), ("OLMo-2-7B", "olmo"), None,
            ("Qwen2.5-14B", "qwen14b_v2"), ("Qwen2.5-32B", "qwen32b_awq_v2"), ("Qwen3-8B, no thinking", "qwen3_8b_nothink_v2"),
            ("Qwen3-8B, thinking", "qwen3_8b"), None, ("R1-Distill-7B$^\\dagger$", "r1_7b")]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.6), sharey=True, gridspec_kw={"wspace": 0.34})
    y = 0.6; ypos = []; headers = [(0.0, "7–8B instruction-tuned")]
    for r in rows:
        if r is None:
            y += 0.9
            headers.append((y - 0.6, "larger or newer" if len(headers) == 1 else "reasoning distill (noisy)"))
            continue
        ypos.append((y, r)); y += 1
    for ax, cue, lab in ((axes[0], "counter", "strong challenge"), (axes[1], "weak", "tentative suggestion")):
        for yy, (name, key) in ypos:
            c = B[key][cue]; near, far, D = c["FFnear"], c["FFfar"], c["D"]
            faded = key == "r1_7b"; a = 0.4 if faded else 1.0
            ax.plot([far[0], near[0]], [yy, yy], color=LIGHT, lw=2.6, solid_capstyle="round", zorder=1, alpha=a)
            ax.errorbar(near[0], yy, xerr=[[near[0] - near[1]], [near[2] - near[0]]], fmt="o", ms=4, color=VERM, lw=0.7, capsize=0, zorder=3, alpha=a)
            ax.errorbar(far[0], yy, xerr=[[far[0] - far[1]], [far[2] - far[0]]], fmt="o", ms=4, color=BLUE, mfc="white", mew=1.1, lw=0.7, capsize=0, zorder=4, alpha=a)
            sig = (D[1] > 0) or (D[2] < 0)
            ax.text(1.03, yy, f"{D[0]:+.2f}".replace("-", "−"), fontsize=7.9, va="center", ha="left",
                    color=(INK if sig and not faded else MUTED), fontweight=("bold" if sig and not faded else "normal"), clip_on=False)
        ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
        ax.set_xlabel(f"switch rate, {lab}")
        ax.grid(axis="x", color="#eeeeee", lw=0.6); ax.set_axisbelow(True)
        ax.text(1.03, 0.0, "last − 2nd", fontsize=7, color=MUTED, ha="left", va="center", clip_on=False)
    axes[0].set_yticks([p[0] for p in ypos]); axes[0].set_yticklabels([p[1][0] for p in ypos])
    axes[0].tick_params(axis="y", length=0); axes[1].tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    for ax in axes: ax.spines["left"].set_visible(False)
    for hy, ht in headers:
        axes[0].text(-0.02, hy, ht, transform=axes[0].get_yaxis_transform(), fontsize=7.9, color=MUTED,
                     style="italic", ha="right", va="center")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=VERM, ms=4, label="challenged with the model's 2nd choice"),
                        plt.Line2D([], [], marker="o", ls="", color=BLUE, mfc="white", mew=1.1, ms=4, label="challenged with its last choice")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05), fontsize=7.8, handletextpad=0.2, columnspacing=1.6)
    save(fig, "fig3_wrong_for_wrong")


if __name__ == "__main__":
    figure1(); figure3(); print("ok")
