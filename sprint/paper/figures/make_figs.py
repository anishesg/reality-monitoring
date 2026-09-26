"""Generate all figures for the paper.

Data sources (read-only):
  results/out_b.txt          -- Exp B, wrong-for-wrong test (cell rates + 95% CIs)
  results/out_h.txt          -- Exp H, accuracy vs. AUROC of a calibrated note
  results/out_e_qwen7b.txt   -- base Qwen2.5-7B note conditions
  results/out_e_llama.txt    -- base Llama-3.1-8B note conditions
  results/out_base_q14.txt   -- base Qwen2.5-14B note conditions
  DOSSIER.md section 1       -- AUROC table (hard bank), hard-coded below

Run:  /usr/local/bin/python3 figures/make_figs.py   (from paper_final/)
"""
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

# Okabe-Ito colorblind-safe palette, fixed roles across all figures
BLUE = "#0072B2"      # rule-trained / verbal confidence / last choice
ORANGE = "#E69F00"    # own-confidence trained / P(True)
GREEN = "#009E73"     # belief score / perfect note
PURPLE = "#CC79A7"    # shuffled-confidence trained
VERM = "#D55E00"      # second choice
GRAY = "#7f7f7f"      # untrained base / no note
INK = "#222222"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.edgecolor": "#444444",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(HERE, name + ".png"), bbox_inches="tight", pad_inches=0.02, dpi=200)
    plt.close(fig)


def panel_label(ax, s, dx=0.0):
    ax.text(dx, 1.03, s, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left")


# ----------------------------------------------------------------------------- parsers
def parse_out_b():
    """Return {model: {cue: {cell: (rate, lo, hi)}}} and D_FF per cue."""
    txt = open(os.path.join(RES, "out_b.txt")).read()
    out = {}
    for block in txt.split("====================")[1:]:
        name = block.split()[0]
        cues = {}
        for cue in ("counter", "weak", "pressure"):
            m = re.search(r"\[%s\](.*)\n\s+D_FF\(far-near\) ([+-][\d.na]+) \[([+-][\d.na]+),([+-][\d.na]+)\]" % cue, block)
            if not m:
                continue
            cells = {}
            for c in re.finditer(r"(FT|FFnear|FFfar|TFnear|TFfar) ([\d.na]+)\[([\d.na]+),([\d.na]+)\]", m.group(1)):
                cells[c.group(1)] = tuple(float(c.group(i)) for i in (2, 3, 4))
            cells["D"] = tuple(float(m.group(i)) for i in (2, 3, 4))
            cues[cue] = cells
        out[name] = cues
    return out


def parse_out_h():
    txt = open(os.path.join(RES, "out_h.txt")).read()
    out = {}
    for block in txt.split("===== ")[1:]:
        name = block.split()[0]
        syn = [(float(a), float(b), float(c)) for a, b, c in re.findall(
            r"realized AUROC ([\d.]+)\s+acc ([\d.]+)\s+best-possible ([\d.]+)", block)]
        real = {k: (float(a), float(b), float(lo), float(hi)) for k, a, b, lo, hi in re.findall(
            r"real (\w+)\s+AUROC ([\d.]+)\s+acc ([\d.]+) \[([\d.]+),([\d.]+)\]", block)}
        out[name] = {"syn": syn, "real": real}
    return out


def parse_base(fname):
    for line in open(os.path.join(RES, fname)):
        if line.startswith("base"):
            v = [float(x) for x in line.split()[1:7]]
            return dict(zip(["init", "none", "raw", "calib", "shuffled", "oracle"], v))


# ----------------------------------------------------------------------------- Figure 1
def figure1():
    # (a) DOSSIER section 1: hard-bank AUROC (verbal conf / P(True) / belief)
    auroc = [
        ("Qwen2.5-7B", .67, .66, .68),
        ("Llama-3.1-8B", .69, .70, .69),
        ("OLMo-2-7B", .59, .65, .71),
        ("Qwen2.5-14B", .65, .74, .69),
        ("Qwen3-8B", .60, .78, .68),
        ("R1-Distill-7B", .62, .71, .60),
    ]
    base = {
        "Qwen2.5-7B": parse_base("out_e_qwen7b.txt"),
        "Llama-3.1-8B": parse_base("out_e_llama.txt"),
        "Qwen2.5-14B": parse_base("out_base_q14.txt"),
    }
    H = parse_out_h()

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05),
                             gridspec_kw={"width_ratios": [1.0, 0.82, 1.18], "wspace": 0.62})

    # ---- (a) they know
    ax = axes[0]
    ys = list(range(len(auroc)))[::-1]
    for y, (name, v, p, b) in zip(ys, auroc):
        ax.plot([min(v, p, b), max(v, p, b)], [y, y], color="#cccccc", lw=1.0, zorder=1)
        ax.scatter(v, y, s=16, marker="o", color=BLUE, zorder=3, lw=0)
        ax.scatter(p, y, s=18, marker="s", color=ORANGE, zorder=3, lw=0)
        ax.scatter(b, y, s=22, marker="^", color=GREEN, zorder=3, lw=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([a[0] + ("*" if a[0] in ("Qwen3-8B", "R1-Distill-7B") else "") for a in auroc])
    ax.axvline(0.5, color="#999999", lw=0.6, ls=":")
    ax.text(0.505, -0.9, "chance", fontsize=6.5, color="#666666", va="bottom")
    ax.set_xlim(0.48, 0.82)
    ax.set_ylim(-1.0, len(auroc) - 0.4)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel("AUROC for own correctness")
    handles = [Line2D([], [], marker="o", color=BLUE, lw=0, ms=4, label="stated"),
               Line2D([], [], marker="s", color=ORANGE, lw=0, ms=4, label="P(True)"),
               Line2D([], [], marker="^", color=GREEN, lw=0, ms=4.5, label="belief")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              handletextpad=0.05, columnspacing=0.5, borderaxespad=0.1)
    panel_label(ax, "a", dx=-0.62)

    # ---- (b) they don't act on it
    ax = axes[1]
    names = ["Qwen2.5-7B", "Llama-3.1-8B", "Qwen2.5-14B"]
    short = ["Qwen\n7B", "Llama\n8B", "Qwen\n14B"]
    w = 0.36
    for i, n in enumerate(names):
        d = base[n]
        ax.bar(i - w / 2 - 0.01, d["none"], w, color=GRAY, lw=0)
        ax.bar(i + w / 2 + 0.01, d["oracle"], w, color=GREEN, lw=0)
        ax.plot([i - w - 0.04, i + w + 0.04], [d["init"]] * 2, color=INK, lw=1.2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(short)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("Accuracy after strong challenge")
    handles = [Line2D([], [], color=INK, lw=1.2, label="before challenge"),
               plt.Rectangle((0, 0), 1, 1, color=GRAY, label="no note"),
               plt.Rectangle((0, 0), 1, 1, color=GREEN, label="perfect note")]
    ax.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.55), ncol=1, frameon=False,
              handlelength=1.2, handletextpad=0.4, borderaxespad=0.1)
    panel_label(ax, "b", dx=-0.30)

    # ---- (c) training makes them act, up to what they know
    ax = axes[2]

    def xy(run, idx):
        s = H[run]["syn"]
        return [p[0] for p in s], [p[idx] for p in s]

    x, y = xy("rule_s0", 2)
    ax.plot(x, y, color=INK, lw=1.0, ls="--", zorder=2)
    for seed in ("own_s0", "own_s1", "own_s2"):
        x, y = xy(seed, 1)
        ax.plot(x, y, color=ORANGE, lw=1.0, alpha=0.75, zorder=3)
    x, y = xy("rule_s0", 1)
    ax.plot(x, y, color=BLUE, lw=1.6, zorder=4)
    x, y = xy("shuffled_s0", 1)
    ax.plot(x, y, color=PURPLE, lw=1.4, zorder=3)
    x, y = xy("base", 1)
    ax.plot(x, y, color=GRAY, lw=1.4, zorder=3)
    # real signals for the rule-trained model
    for k, mk in (("verbal", "o"), ("ptrue", "s"), ("belief", "^"), ("letter", "D")):
        a, acc, lo, hi = H["rule_s0"]["real"][k]
        ax.errorbar(a, acc, yerr=[[acc - lo], [hi - acc]], fmt=mk, ms=3.6, color=BLUE, mfc="white",
                    mew=0.9, elinewidth=0.6, capsize=0, zorder=5)
    a, acc, _, _ = H["rule_s0"]["real"]["letter"]
    ax.annotate("letter probe", xy=(a, acc), xytext=(0.655, 0.86), fontsize=6.5, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5))
    ax.annotate("self-report", xy=(0.58, 0.74), xytext=(0.52, 0.55), fontsize=6.5, color=BLUE,
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.5))
    # direct labels
    ax.text(1.005, 0.96, "best possible", fontsize=6.5, color=INK, va="center")
    ax.text(1.005, 0.905, "rule-trained", fontsize=6.5, color=BLUE, va="center")
    ax.text(1.005, 0.845, "own-trained\n(3 seeds)", fontsize=6.5, color="#a87400", va="center",
            linespacing=0.9)
    ax.text(1.005, 0.755, "shuffled-\ntrained", fontsize=6.5, color=PURPLE, va="center", linespacing=0.9)
    ax.text(1.005, 0.235, "untrained", fontsize=6.5, color=GRAY, va="center")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.15, 1.0)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("AUROC of calibrated confidence note")
    ax.set_ylabel("Accuracy after strong challenge")
    panel_label(ax, "c", dx=-0.3)

    save(fig, "fig1_overview")


# ----------------------------------------------------------------------------- Figure 3
def figure3():
    B = parse_out_b()
    rows = [  # (label, key, group)
        ("Llama-3.1-8B", "llama8b", 0),
        ("Qwen2.5-7B", "qwen7b", 0),
        ("OLMo-2-7B", "olmo", 0),
        ("Qwen2.5-14B", "qwen14b_v2", 1),
        ("Qwen2.5-32B", "qwen32b_awq_v2", 1),
        ("Qwen3-8B, no think", "qwen3_8b_nothink_v2", 1),
        ("Qwen3-8B, think", "qwen3_8b", 1),
        ("R1-Distill-7B$^\\dagger$", "r1_7b", 2),
    ]
    # y positions with gaps between groups
    ypos, y = [], 0.0
    prev = 0
    for _, _, g in rows:
        if g != prev:
            y += 0.7
            prev = g
        ypos.append(y)
        y += 1.0
    ypos = [max(ypos) - v for v in ypos]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.55), sharey=True, gridspec_kw={"wspace": 0.22})
    for ax, cue, lab in ((axes[0], "counter", "strong challenge"), (axes[1], "weak", "tentative suggestion")):
        for (name, key, g), yy in zip(rows, ypos):
            c = B[key][cue]
            near, far, D = c["FFnear"], c["FFfar"], c["D"]
            muted = g == 2
            ax.plot([near[0], far[0]], [yy, yy], color="#bbbbbb", lw=1.4, zorder=1)
            ax.errorbar(near[0], yy + 0.13, xerr=[[near[0] - near[1]], [near[2] - near[0]]], fmt="o", ms=4,
                        color=VERM, elinewidth=0.7, capsize=0, zorder=3, alpha=0.55 if muted else 1)
            ax.errorbar(far[0], yy - 0.13, xerr=[[far[0] - far[1]], [far[2] - far[0]]], fmt="o", ms=4,
                        color=BLUE, mfc="white", mew=1.0, elinewidth=0.7, capsize=0, zorder=3,
                        alpha=0.55 if muted else 1)
            ax.text(1.02, yy, ("%+.3f" % D[0]).replace("0.", ".").replace("-", "\u2212"), fontsize=6.5, va="center", ha="left",
                    color="#666666" if muted else INK, transform=ax.get_yaxis_transform())
        ax.text(1.02, max(ypos) + 0.95, "last \u2212\nsecond", linespacing=0.9, fontsize=6.5, ha="left", va="center",
                transform=ax.get_yaxis_transform(), color=INK)
        ax.set_xlim(-0.02, 1.02)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Switch rate, %s" % lab)
        ax.grid(axis="x", color="#eeeeee", lw=0.5, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([r[0] for r in rows])
    axes[0].tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", length=0)
    for ax in axes:
        ax.set_ylim(min(ypos) - 0.6, max(ypos) + 0.6)
    # group brackets
    g0 = [yy for (_, _, g), yy in zip(rows, ypos) if g == 0]
    g1 = [yy for (_, _, g), yy in zip(rows, ypos) if g == 1]
    for grp, txt in ((g0, "7-8B"), (g1, "larger /\nnewer")):
        axes[0].annotate("", xy=(-0.60, min(grp) - 0.3), xytext=(-0.60, max(grp) + 0.3),
                         xycoords=("axes fraction", "data"),
                         arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7))
        axes[0].text(-0.63, (min(grp) + max(grp)) / 2, txt, transform=axes[0].get_yaxis_transform(),
                     rotation=90, ha="right", va="center", fontsize=6.8, color="#555555", linespacing=0.9)
    handles = [Line2D([], [], marker="o", color=VERM, lw=0, ms=4, label="wrong answer challenged with the model's second choice"),
               Line2D([], [], marker="o", color=BLUE, mfc="white", mew=1.0, lw=0, ms=4,
                      label="... with the model's last choice")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.96), ncol=2, frameon=False,
               handletextpad=0.2, columnspacing=1.2)
    save(fig, "fig3_wrong_for_wrong")


if __name__ == "__main__":
    figure1()
    figure3()
    print("ok")
