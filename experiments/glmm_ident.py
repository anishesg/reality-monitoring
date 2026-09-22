#!/usr/bin/env python3
"""Pooled logistic regression on the 3-cell identification data (counter + pressure).
switch ~ cue(counter) + claim_true + alt_true + z(belief margin) + z(own conf) + model FE,
cluster-robust SEs by (model,qid). Also per-model average marginal effects."""
import json, glob, os
import numpy as np, pandas as pd
import statsmodels.api as sm

BASE = os.path.expanduser("~/reality-monitoring-repo/results_ident")
rows = []
for d in sorted(glob.glob(BASE + "/*")):
    tag = os.path.basename(d)
    if not os.path.exists(d + "/ident.jsonl"): continue
    sig = {r["qid"]: r for r in (json.loads(l) for l in open(d + "/signals.jsonl"))}
    for l in open(d + "/ident.jsonl"):
        r = json.loads(l)
        if r["outcome"] not in ("retain", "switch_alt", "switch_other"): continue
        s = sig.get(r["qid"], {})
        b = {"true_answer": s.get("b_true"), "d1": s.get("b_d1"), "d2": s.get("b_d2")}
        cmap = {"TF": ("true_answer", "d1"), "FT": ("d1", "true_answer"), "FF": ("d1", "d2")}
        ck, ak = cmap[r["cell"]]
        bm = (b[ak] - b[ck]) if (b[ak] is not None and b[ck] is not None) else None
        rows.append({
            "model": tag, "qid": r["qid"], "cell": r["cell"],
            "switch": int(r["outcome"] != "retain"),
            "counter": int(r["chal"] == "counter"),
            "claim_true": int(r["cell"] == "TF"),
            "alt_true": int(r["cell"] == "FT"),
            "bmargin": bm, "conf": s.get("conf"),
        })
df = pd.DataFrame(rows)
print("trials:", len(df))
d = df.dropna(subset=["bmargin", "conf"]).copy()
d["z_bm"] = (d.bmargin - d.bmargin.mean()) / d.bmargin.std()
d["z_conf"] = (d.conf - d.conf.mean()) / d.conf.std()
X = pd.get_dummies(d[["counter", "claim_true", "alt_true", "z_bm", "z_conf", "model"]],
                   columns=["model"], drop_first=True).astype(float)
X = sm.add_constant(X)
groups = d.model + "_" + d.qid.astype(str)
m = sm.Logit(d.switch.astype(float), X).fit(disp=0, cov_type="cluster",
                                            cov_kwds={"groups": groups})
print("\n=== pooled logit (cluster-robust by model x item), n=%d ===" % len(d))
out = pd.DataFrame({"coef": m.params, "se": m.bse, "p": m.pvalues})
print(out.loc[[i for i in out.index if not i.startswith("model_")]].round(3))
# average marginal effects for the interpretable predictors
ame = []
for var in ["counter", "claim_true", "alt_true", "z_bm", "z_conf"]:
    X1, X0 = X.copy(), X.copy()
    if var.startswith("z_"):
        X1[var] += 1.0  # +1 SD
    else:
        X1[var], X0[var] = 1.0, 0.0
    p1 = m.predict(X1).mean(); p0 = m.predict(X0).mean()
    ame.append((var, round(p1 - p0, 4)))
print("\naverage marginal effects on P(switch):")
for v, a in ame: print(f"  {v:12s} {a:+.4f}")
# alt-truth vs belief within FF+FT only (the identification contrast)
for tag in sorted(d.model.unique()):
    sub = d[(d.model == tag) & (d.counter == 1)]
    ff = sub[sub.cell == "FF"]; ft = sub[sub.cell == "FT"]; tf = sub[sub.cell == "TF"]
    print(f"{tag:12s} counter: TF {tf.switch.mean():.3f} FT {ft.switch.mean():.3f} FF {ff.switch.mean():.3f}"
          f" | pressure: {d[(d.model==tag)&(d.counter==0)].switch.mean():.3f}")
