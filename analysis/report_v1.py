#!/usr/bin/env python3
import json

def line(name, v, ra, rb):
    return f"  {name:16s} {v['delta']:+.3f}  [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]  rates: {v[ra]:.3f} vs {v[rb]:.3f}  n={v['n']}"

for d in ["qwen25_1p5b", "qwen25_7b", "mistral7b", "olmo2_7b"]:
    r = json.load(open(f"results/{d}/analysis.json"))
    print("=" * 70)
    print(f"{d}  n={r['n_analyzed']}  fc_acc={r['baseline_acc']:.3f}  excluded={r['unparsed_or_ambiguous_frac']:.3f}")
    print(line("delta_revision", r["delta_revision"], "switch_given_false", "switch_given_true"))
    print(line("conf_use_self", r["conf_use_self"], "abandon_low", "abandon_high"))
    print(line("conf_use_user", r["conf_use_user"], "abandon_low", "abandon_high"))
    print(line("origin_effect", r["origin_effect"], "abandon_self", "abandon_user"))
    print(f"  pressure_abandon_true={r['pressure_abandon_true_claims']:.3f}   conf_use_self|knows={r['conf_use_self_knows']:+.3f}")
