import json, glob, os
import numpy as np, pandas as pd
import statsmodels.api as sm

BASE = os.path.expanduser("~/reality-monitoring-repo/results_ident")
rows = []
for d in sorted(glob.glob(BASE + "/*")):
    tag = os.path.basename(d)
    if not os.path.exists(d + "/ident.jsonl"): continue
    sig = {r["qid"]: r for r in (json.loads(l) for l in open(d + "/signals.jsonl"))}
    srcs = [("ident.jsonl", None), ("weak.jsonl", "weak")]
    for fn, forced in srcs:
        if not os.path.exists(d + "/" + fn): continue
        for l in open(d + "/" + fn):
            r = json.loads(l)
            if r["outcome"] not in ("retain", "switch_alt", "switch_other"): continue
            s = sig.get(r["qid"], {})
            b = {"true_answer": s.get("b_true"), "d1": s.get("b_d1"), "d2": s.get("b_d2")}
            cmap = {"TF": ("true_answer", "d1"), "FT": ("d1", "true_answer"), "FF": ("d1", "d2")}
            ck, ak = cmap[r["cell"]]
            bm = (b[ak] - b[ck]) if (b[ak] is not None and b[ck] is not None) else None
            chal = forced or r["chal"]
            rows.append({"model": tag, "qid": r["qid"], "cell": r["cell"],
                         "switch": int(r["outcome"] != "retain"),
                         "chal": chal,
                         "claim_true": int(r["cell"] == "TF"),
                         "alt_true": int(r["cell"] == "FT"),
                         "bmargin": bm, "conf": s.get("conf")})
df = pd.DataFrame(rows)
d = df.dropna(subset=["bmargin", "conf"]).copy()
d["z_bm"] = (d.bmargin - d.bmargin.mean()) / d.bmargin.std()
d["z_conf"] = (d.conf - d.conf.mean()) / d.conf.std()
d["counter"] = (d.chal == "counter").astype(int)
d["weak"] = (d.chal == "weak").astype(int)
X = pd.get_dummies(d[["counter", "weak", "claim_true", "alt_true", "z_bm", "z_conf", "model"]],
                   columns=["model"], drop_first=True).astype(float)
X = sm.add_constant(X)
groups = d.model + "_" + d.qid.astype(str)
m = sm.Logit(d.switch.astype(float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups})
print("n =", len(d), " (pressure is reference challenge)")
out = pd.DataFrame({"coef": m.params, "se": m.bse, "p": m.pvalues})
print(out.loc[[i for i in out.index if not i.startswith("model_")]].round(3))
print("\naverage marginal effects on P(switch):")
for var in ["counter", "weak", "claim_true", "alt_true", "z_bm", "z_conf"]:
    X1, X0 = X.copy(), X.copy()
    if var.startswith("z_"): X1[var] += 1.0
    else: X1[var], X0[var] = 1.0, 0.0
    print("  %-11s %+.4f" % (var, m.predict(X1).mean() - m.predict(X0).mean()))
# off-ceiling only (weak trials): same regression
w = d[d.chal == "weak"]
Xw = pd.get_dummies(w[["claim_true", "alt_true", "z_bm", "z_conf", "model"]], columns=["model"], drop_first=True).astype(float)
Xw = sm.add_constant(Xw)
mw = sm.Logit(w.switch.astype(float), Xw).fit(disp=0, cov_type="cluster",
                                              cov_kwds={"groups": w.model + "_" + w.qid.astype(str)})
print("\nweak-cue-only AMEs (off-ceiling, n=%d):" % len(w))
for var in ["claim_true", "alt_true", "z_bm", "z_conf"]:
    X1, X0 = Xw.copy(), Xw.copy()
    if var.startswith("z_"): X1[var] += 1.0
    else: X1[var], X0[var] = 1.0, 0.0
    print("  %-11s %+.4f" % (var, mw.predict(X1).mean() - mw.predict(X0).mean()))
