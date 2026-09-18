#!/usr/bin/env python3
import json, os
rows = json.load(open("results_v16/all.json")) if os.path.exists("results_v16/all.json") else []
order = ["h_qwen05", "h_qwen15", "h_qwen3b", "h_qwen7b", "h_qwen14b", "h_qwen32b",
         "h_llama1b", "h_llama3b", "h_llama8b", "h_phi35", "h_mistral",
         "h_zephyr_sft", "h_zephyr_dpo", "h_olmo_sft", "h_olmo_dpo", "h_olmo_rlvr",
         "h_tulu_sft", "h_tulu_dpo", "h_tulu_rlvr"]
bydir = {r["dir"]: r for r in rows}
print(f"{'model':14s} {'accFC':>6} {'dmg':>7} {'swTrue':>7} {'cUseSelf':>9} {'cUseUser':>9} {'AUROCc':>7}")
for d in order:
    r = bydir.get(d)
    if not r: continue
    v = r.get("value_counter", {}); dc = r.get("discrim_counter", {})
    dmg = None
    if v.get("keep_all_acc") is not None and v.get("post_challenge_acc") is not None:
        dmg = round(v["keep_all_acc"] - v["post_challenge_acc"], 3)
    ius = r.get("inj_conf_use_self", {}); iuu = r.get("inj_conf_use_user", {})
    ds = None if ius.get("lo") is None or ius.get("hi") is None else round(ius["lo"] - ius["hi"], 3)
    du = None if iuu.get("lo") is None or iuu.get("hi") is None else round(iuu["lo"] - iuu["hi"], 3)
    au = r["auroc"]["conf"][0] if r["auroc"].get("conf") and r["auroc"]["conf"][0] else None
    print(f"{d:14s} {str(r.get('acc_fc')):>6} {str(dmg):>7} {str(dc.get('sw_true')):>7} {str(ds):>9} {str(du):>9} {str(au):>7}")
