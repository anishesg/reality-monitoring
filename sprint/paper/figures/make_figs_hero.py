"""Hero versions of Figure 1 and Figure 3. Same data as make_figs.py (parsers imported); only presentation differs.
Run from paper_final/:  /usr/local/bin/python3 figures/make_figs_hero.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figs import parse_out_b, parse_out_h, parse_base, save

INK, MUTED, GRID = "#16181d", "#7a7f87", "#e9ebee"
NAVY, GOLD, TEAL, RED, SKY = "#1f4e99", "#e0a100", "#0f9d78", "#d1495b", "#dbe7f5"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 8.8, "xtick.labelsize": 8.2, "ytick.labelsize": 8.2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.linewidth": 0.7, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 4, "ytick.major.pad": 4,
    "pdf.fonttype": 42, "savefig.dpi": 300,
})


def head(ax, letter, bold, rest=""):
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(-26, 30 if rest else 14), textcoords="offset points",
                fontsize=11, fontweight="bold", color=INK, va="bottom", ha="left")
    ax.annotate(bold, (0, 1), xycoords="axes fraction", xytext=(-12, 30 if rest else 14), textcoords="offset points",
                fontsize=9.4, fontweight="bold", color=INK, va="bottom", ha="left")
    if rest:
        ax.annotate(rest, (0, 1), xycoords="axes fraction", xytext=(-12, 16), textcoords="offset points",
                    fontsize=7.8, color=MUTED, va="bottom", ha="left")


def ygrid(ax, ys):
    for y in ys:
        ax.axhline(y, color=GRID, lw=0.7, zorder=0)


def figure1():
    auroc = [("Qwen2.5-7B", .67, .66, .68), ("Llama-3.1-8B", .69, .70, .69), ("OLMo-2-7B", .59, .65, .71),
             ("Qwen2.5-14B", .65, .74, .69), ("Qwen3-8B", .60, .78, .68), ("R1-Distill-7B", .62, .71, .60)]
    base = [("Qwen2.5-7B", parse_base("out_e_qwen7b.txt")), ("Llama-3.1-8B", parse_base("out_e_llama.txt")),
            ("Qwen2.5-14B", parse_base("out_base_q14.txt"))]
    H = parse_out_h()

    fig = plt.figure(figsize=(5.0, 5.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], width_ratios=[1.0, 1.08], hspace=1.0, wspace=0.62)
    axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    # ---------------- (a) models know
    ax = axA
    ys = np.arange(len(auroc))[::-1]
    ax.axvspan(0.44, 0.5, color="#f3f4f6", lw=0, zorder=0)
    ax.text(0.47, len(auroc) - 0.35, "chance", fontsize=7.4, color=MUTED, ha="center", rotation=90, va="top")
    for y, (n, v, p, b) in zip(ys, auroc):
        lo, hi = min(v, p, b), max(v, p, b)
        ax.plot([0.5, lo], [y, y], color=GRID, lw=1.0, zorder=1)
        ax.plot([lo, hi], [y, y], color="#b9c6db", lw=4.5, solid_capstyle="round", zorder=2)
        ax.plot(hi, y, "o", ms=5.6, color=NAVY, zorder=3)
        ax.text(hi + 0.012, y, f"{hi:.2f}", fontsize=7.6, color=NAVY, va="center", fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels([a[0] for a in auroc]); ax.tick_params(axis="y", colors=INK)
    ax.set_xlim(0.44, 0.84); ax.set_xticks([0.5, 0.6, 0.7, 0.8]); ax.set_ylim(-0.6, len(auroc) - 0.4)
    ax.spines["bottom"].set_visible(True)
    ax.set_xlabel("AUROC for own correctness")
    head(ax, "a", "Models know", "bar: 3 signals · dot: highest")

    # ---------------- (b) they ignore a perfect note
    ax = axB
    ygrid(ax, [0.25, 0.5, 0.75, 1.0])
    x = np.arange(len(base)) * 1.0
    for xi, (n, d) in zip(x, base):
        drop = d["init"] - d["none"]
        ax.add_patch(FancyArrowPatch((xi, d["init"] - 0.03), (xi, d["none"] + 0.035), arrowstyle="-|>,head_width=3.2,head_length=4.5",
                                     color=RED, lw=2.0, zorder=2, shrinkA=0, shrinkB=0))
        ax.plot(xi, d["init"], "o", ms=7, color="white", mec=INK, mew=1.4, zorder=3)
        ax.plot(xi - 0.11, d["none"], "o", ms=6.5, color=MUTED, zorder=3)
        ax.plot(xi + 0.11, d["oracle"], "D", ms=6, color=TEAL, zorder=4, mec="white", mew=0.6)
        ax.text(xi + 0.09, (d["init"] + d["none"]) / 2, f"−{100*drop:.0f}\npts", fontsize=8.4, color=RED, fontweight="bold",
                va="center", ha="left", linespacing=0.95)
    ax.set_xticks(x); ax.set_xticklabels([n.replace("-", "\n", 1) for n, _ in base], fontsize=7.8, color=INK, linespacing=1.0)
    ax.set_xlim(-0.45, 2.55); ax.set_ylim(0, 1.04); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("accuracy")
    ax.text(1.0, 0.075, "perfect note (95% if right, 5% if wrong): ±0.01", fontsize=7.6, color=TEAL, ha="center", fontweight="bold")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=6, color="white", mec=INK, mew=1.3, label="before"),
                       plt.Line2D([], [], marker="o", ls="", ms=6, color=MUTED, label="after, no note"),
                       plt.Line2D([], [], marker="D", ls="", ms=5.5, color=TEAL, label="after, perfect note")],
              loc="upper center", bbox_to_anchor=(0.45, -0.25), ncol=3, frameon=False, fontsize=7.4, handletextpad=0.1, columnspacing=0.6)
    head(ax, "b", "A perfect note is ignored", "after “Actually, another source says … X.”")

    # ---------------- (c) training makes them act, up to what confidence is worth
    ax = axC
    ygrid(ax, [0.25, 0.5, 0.75, 1.0])
    best = np.array([(s[0], s[2]) for s in H["rule_s0"]["syn"]])
    rkeys = [k for k in ("rule_s0", "rule_s1", "rule_s2") if k in H]
    rarr = np.array([[s[:2] for s in H[k]["syn"]] for k in rkeys])
    rule = np.stack([rarr[:, :, 0].mean(0), rarr[:, :, 1].mean(0)], 1)
    ubest = np.array([(s[0], s[2]) for s in H["base"]["syn"]])
    unt = np.array([(s[0], s[1]) for s in H["base"]["syn"]])
    own = np.array([[s[:2] for s in H[k]["syn"]] for k in ("own_s0", "own_s1", "own_s2")])
    ox, oy = own[:, :, 0].mean(0), own[:, :, 1]
    ax.axvspan(0.50, 0.68, color=SKY, lw=0, zorder=0)
    ax.text(0.59, 0.40, "AUROC of models'\nown stated confidence", fontsize=7.6, color=NAVY, ha="center", va="center")
    # headroom above keep-all that only a better monitor unlocks
    keep = best[0, 1]
    ax.fill_between(best[:, 0], keep, best[:, 1], where=best[:, 1] >= keep, color="#cfc3e6", alpha=0.75, lw=0, zorder=1)
    ax.annotate("gain that needs\na better monitor", (0.93, 0.79), xytext=(0.87, 0.62), fontsize=7.6, color="#5b3f8c", ha="center",
                arrowprops=dict(arrowstyle="-", color="#5b3f8c", lw=0.6, shrinkA=0, shrinkB=1))
    # training gain bracket at the right edge (AUROC 0.99 synthetic note)
    xr = rule[-1, 0]
    ax.add_patch(FancyArrowPatch((xr - 0.012, unt[-1, 1] + 0.01), (xr - 0.012, rule[-1, 1] - 0.01), arrowstyle="<|-|>,head_width=2.4,head_length=3.4",
                                 color=NAVY, lw=1.0, zorder=6, shrinkA=0, shrinkB=0))
    ax.text(xr - 0.022, 0.52, f"+{rule[-1, 1] - unt[-1, 1]:.2f}\nfrom\ntraining", fontsize=8, color=NAVY, fontweight="bold", ha="right", va="center", linespacing=0.95)
    ax.fill_between(ox, oy.min(0), oy.max(0), color=GOLD, alpha=0.22, lw=0, zorder=2)
    ax.plot(ox, oy.mean(0), color=GOLD, lw=1.8, zorder=3)
    if len(rkeys) > 1:
        ax.fill_between(rule[:, 0], rarr[:, :, 1].min(0), rarr[:, :, 1].max(0), color=NAVY, alpha=0.16, lw=0, zorder=3)
    ax.plot(rule[:, 0], rule[:, 1], color=NAVY, lw=2.6, zorder=4)
    ax.plot(ubest[:, 0], ubest[:, 1], color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=4)
    ax.text(0.515, ubest[0, 1] + 0.035, "best possible for the untrained model", fontsize=7.6, color=MUTED, va="center")
    ax.plot(best[:, 0], best[:, 1], color=INK, lw=1.0, ls=(0, (3, 2)), zorder=5)
    ax.plot(unt[:, 0], unt[:, 1], color=RED, lw=2.2, zorder=3)
    lp = H["rule_s0"]["real"]["letter"]
    ax.errorbar([lp[0]], [lp[1]], yerr=[[lp[1] - lp[2]], [lp[3] - lp[1]]], fmt="*", ms=11, color=NAVY, mfc=GOLD, mec=NAVY, mew=0.8,
                capsize=0, lw=0.8, zorder=6)
    ax.annotate("letter-probability\nnote", (lp[0], lp[1]), xytext=(0.72, 0.60), fontsize=7.6, color=NAVY, ha="center",
                arrowprops=dict(arrowstyle="-", color=NAVY, lw=0.6, shrinkA=0, shrinkB=5))
    R = 1.006
    ax.text(R, 1.12, "best possible\n(dashed)", fontsize=7.6, color=INK, va="center")
    ax.text(R, 0.94, "trained on rule" + (f"\n({len(rkeys)} seeds)" if len(rkeys) > 1 else ""), fontsize=7.6, color=NAVY, va="center", fontweight="bold")
    ax.text(R, 0.795, "trained on own\nconfidence (3 seeds)", fontsize=7.6, color="#9a6b00", va="center")
    ax.text(R, unt[-1, 1], "untrained:\nflat at 0.24", fontsize=7.9, color=RED, va="center", fontweight="bold")
    ax.set_xlim(0.48, 1.0); ax.set_ylim(0.15, 1.08); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.spines["bottom"].set_visible(True)
    ax.set_xlabel("quality of the confidence note in the model's turn (AUROC)"); ax.set_ylabel("accuracy after challenge")
    head(ax, "c", "Training makes them act, up to what their confidence is worth",
         "Qwen2.5-7B given calibrated notes of controlled quality")
    save(fig, "fig1_overview")


def figure3():
    B = parse_out_b()
    groups = [("do not compare", "7–8B instruction-tuned", [("Llama-3.1-8B", "llama8b"), ("Qwen2.5-7B", "qwen7b"), ("OLMo-2-7B", "olmo")]),
              ("compare", "larger or newer", [("Qwen2.5-14B", "qwen14b_v2"), ("Qwen2.5-32B", "qwen32b_awq_v2"),
                                               ("Qwen3-8B, no thinking", "qwen3_8b_nothink_v2"), ("Qwen3-8B, thinking", "qwen3_8b")]),
              ("noisy", "reasoning distill (noisy)", [("R1-Distill-7B$^\\dagger$", "r1_7b")])]
    fig, axes = plt.subplots(1, 2, figsize=(4.75, 3.25), sharey=True, gridspec_kw={"wspace": 0.42})
    rows, y, bands = [], 0.0, []
    for tag, gname, items in groups:
        y0 = y
        for name, key in items:
            rows.append((y, name, key)); y += 1
        bands.append((y0 - 0.5, y - 0.5, tag, gname)); y += 0.7
    for ax, cue, lab in ((axes[0], "counter", "strong challenge"), (axes[1], "weak", "tentative suggestion")):
        for (b0, b1, tag, gname), col in zip(bands, ("#f4f5f7", "#eef4fb", "#fafafa")):
            ax.axhspan(b0, b1, color=col, lw=0, zorder=0)
        for yy, name, key in rows:
            c = B[key][cue]; near, far, D = c["FFnear"], c["FFfar"], c["D"]
            noisy = key == "r1_7b"
            sig = ((D[1] > 0) or (D[2] < 0)) and not noisy
            col = NAVY if sig else MUTED
            ax.plot([near[1], near[2]], [yy - 0.13, yy - 0.13], color=RED, lw=0.8, alpha=0.7, zorder=2)
            ax.plot([far[1], far[2]], [yy + 0.13, yy + 0.13], color=NAVY, lw=0.8, alpha=0.7, zorder=2)
            if abs(near[0] - far[0]) > 0.02:
                ax.add_patch(FancyArrowPatch((near[0], yy), (far[0], yy), arrowstyle="-|>,head_width=2.6,head_length=3.6",
                                             color=col, lw=1.6 if sig else 1.0, zorder=3, shrinkA=4, shrinkB=4))
            al = 0.4 if noisy else 1.0
            ax.plot(near[0], yy, "o", ms=6, color=RED, zorder=4, mec="white", mew=0.6, alpha=al)
            ax.plot(far[0], yy, "o", ms=6, color="white", mec=NAVY, mew=1.4, zorder=5, alpha=al)
            ax.text(1.04, yy, f"{D[0]:+.2f}".replace("-", "−"), fontsize=8, va="center", ha="left", clip_on=False,
                    color=(NAVY if sig else MUTED), fontweight=("bold" if sig else "normal"))
        ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", ".25", ".50", ".75", "1"])
        for xv in (0.25, 0.5, 0.75): ax.axvline(xv, color="white", lw=0.8, zorder=1)
        ax.spines["bottom"].set_visible(True)
        ax.set_xlabel(f"switch rate, {lab}")
        ax.text(1.04, -0.95, "last − 2nd", fontsize=7.4, color=MUTED, ha="left", clip_on=False)
    axes[0].set_yticks([r[0] for r in rows]); axes[0].set_yticklabels([r[1] for r in rows]); axes[0].tick_params(axis="y", colors=INK)
    axes[0].set_ylim(y - 0.55, -1.2)
    for b0, b1, tag, gname in bands:
        axes[0].text(-0.02, b0 - 0.05, gname, transform=axes[0].get_yaxis_transform(), fontsize=7.6, color=MUTED, style="italic", ha="right", va="bottom")
    axes[1].text(0.03, bands[0][0] + 0.2, "no gap: suggestion\nnot weighed\nagainst belief", fontsize=7.4, color=MUTED, va="top")
    yq = [r[0] for r in rows if r[2] == "qwen3_8b_nothink_v2"][0]
    axes[1].text(0.97, yq + 0.5, "gap: switching falls\nwhen the suggestion\nis the last choice", fontsize=7.4, color=NAVY, va="center", ha="right")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=RED, ms=6, label="challenged with the model's 2nd choice"),
                        plt.Line2D([], [], marker="o", ls="", color="white", mec=NAVY, mew=1.4, ms=6, label="challenged with its last choice")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.52, 1.035), fontsize=8.2, handletextpad=0.2, columnspacing=1.6)
    save(fig, "fig3_wrong_for_wrong")


if __name__ == "__main__":
    figure1(); figure3(); print("ok")
