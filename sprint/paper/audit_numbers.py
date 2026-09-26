#!/usr/bin/env python3
"""Flag every decimal / percentage in main.tex that does not appear in DOSSIER.md, the results/ files, or the prior papers.
Usage: python3 audit_numbers.py [main.tex]"""
import glob, re, sys

tex = open(sys.argv[1] if len(sys.argv) > 1 else "main.tex").read()
tex = re.sub(r"(?m)%.*$", "", tex)  # drop comments
body = tex.split("\\begin{document}", 1)[-1]

sources = open("DOSSIER.md").read()
for f in glob.glob("results/*") + ["prior/main_icml_original.tex"]:
    try:
        sources += open(f, errors="ignore").read()
    except IsADirectoryError:
        pass


def variants(x):
    """Strictly equivalent spellings: 0.233 ~ .233; percent 23.3 ~ 0.233; 23 ~ 0.23."""
    out = {x}
    if x.startswith("0."): out.add(x[1:])
    if x.startswith("."): out.add("0" + x)
    if "." in x:
        a, b = x.split(".")
        if a in ("", "0") and len(b) >= 2:           # 0.233 -> 23.3 ; 0.23 -> 23
            p = b[:2].lstrip("0") or "0"
            out.add(p + ("." + b[2:] if len(b) > 2 else ""))
        elif a not in ("", "0"):                      # 23.3 -> 0.233
            out.add("0." + a.zfill(2) + b); out.add("." + a.zfill(2) + b)
    else:
        if len(x) <= 2: out.add("0." + x.zfill(2)); out.add("." + x.zfill(2))
    return out


nums = re.findall(r"(?<![\w.])[-−+]?(\d*\.\d+|\d+(?:,\d{3})+|\d+)(?![\w.])", body)
seen, missing = set(), []
for n in nums:
    n = n.replace(",", "")
    if n in seen:
        continue
    seen.add(n)
    try:
        if "." not in n and int(n) < 20:
            continue  # small integers (counts, section numbers)
    except ValueError:
        pass
    if not any(re.search(r"(?<![\d])" + re.escape(v) + r"(?![\d])", sources) for v in variants(n)):
        missing.append(n)
print(f"checked {len(seen)} distinct numbers; {len(missing)} not found in sources:")
for m in missing:
    ctx = re.search(r".{0,60}" + re.escape(m) + r".{0,40}", body.replace("\n", " "))
    print(f"  {m:>10}  …{ctx.group(0) if ctx else ''}…")
