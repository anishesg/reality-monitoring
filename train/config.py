#!/usr/bin/env python3
"""Tiny YAML config loader: `extends:` chaining + dotted CLI overrides (a.b=1e-5 -> float).
Usage: cfg = load_config("train/configs/A3_olmo.yaml", ["grpo.lr=5e-6", "arm=A3"])
"""
import os, yaml

def _coerce(s):
    if not isinstance(s, str): return s
    low = s.strip().lower()
    if low in ("true", "false"): return low == "true"
    if low in ("null", "none", "~"): return None
    try: return int(s)
    except ValueError: pass
    try: return float(s)
    except ValueError: return s

def _merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

def _load(path, seen=()):
    path = os.path.abspath(path)
    assert path not in seen, f"extends cycle at {path}"
    with open(path) as f: d = yaml.safe_load(f) or {}
    parent = d.pop("extends", None)
    if parent:
        d = _merge(_load(os.path.join(os.path.dirname(path), parent), seen + (path,)), d)
    return d

def set_dotted(d, key, value):
    parts = key.split("."); cur = d
    for p in parts[:-1]: cur = cur.setdefault(p, {})
    cur[parts[-1]] = _coerce(value)

def load_config(path, overrides=()):
    cfg = _load(path)
    for o in overrides or ():
        k, _, v = o.partition("="); set_dotted(cfg, k.strip(), v)
    return cfg

def get(cfg, dotted, default=None):
    cur = cfg
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur: return default
        cur = cur[p]
    return cur

if __name__ == "__main__":
    import sys, json
    print(json.dumps(load_config(sys.argv[1], sys.argv[2:]), indent=1))
