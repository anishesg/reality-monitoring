#!/usr/bin/env python3
"""Live results dashboard for the ladder. Stdlib only; binds to localhost, view through an SSH tunnel.
  python3 train/results_server.py --port 8765            # on the cluster login node, inside the repo
  ssh -L 8765:localhost:8765 <user>@della.princeton.edu   # from your laptop, then open http://localhost:8765
Shows results_ladder/<tag>/<arm>/summary.json (auto-refresh 60 s), job logs, and figures/.
"""
import argparse, glob, html, json, os, sys, time
from http.server import HTTPServer, SimpleHTTPRequestHandler
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = [("retain_correct", "retain correct"), ("accept_valid_correction", "accept fix"), ("pressure_abandon", "pressure abandon"),
        ("counter_bare_abandon", "bare-mention abandon"), ("source_effect", "source effect"), ("conf_use_self_counter_src", "conf use"),
        ("excluded_frac", "excluded")]
ARMS = ["A0", "A2", "A1", "A3", "A4", "A5", "SFT", "DPO", "RLVR"]

def fmt(v):
    if v is None: return "—"
    if isinstance(v, float): return f"{v:.3f}"
    return html.escape(str(v))

def rows():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "results_ladder", "*", "*", "summary.json"))):
        tag, arm = p.split(os.sep)[-3], p.split(os.sep)[-2]
        try: r = json.load(open(p))
        except Exception: continue
        r["_tag"], r["_arm"], r["_mtime"] = tag, arm, time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
        cap = r.get("capability") or {}
        r["_cap"] = ", ".join(f"{k}:{list(v.values())[0]:.3f}" for k, v in cap.items() if isinstance(v, dict) and v and isinstance(list(v.values())[0], float)) or ("err" if "error" in cap else "—")
        out.append(r)
    out.sort(key=lambda r: (r["_tag"], ARMS.index(r["_arm"]) if r["_arm"] in ARMS else 99))
    return out

def jobs():
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, "logs", "rm-*.out")), key=os.path.getmtime, reverse=True)[:30]:
        try: tail = open(p, errors="replace").read()[-400:].strip().splitlines()[-1] if os.path.getsize(p) else ""
        except Exception: tail = ""
        out.append((os.path.basename(p), time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p))), tail[:160]))
    return out

def page():
    R = rows(); J = jobs()
    h = ["<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=60><title>STAND ladder</title>",
         "<style>body{font:14px system-ui;margin:24px;color:#111}table{border-collapse:collapse}td,th{padding:4px 10px;border-bottom:1px solid #ddd;text-align:right}",
         "th:first-child,td:first-child,td:nth-child(2),th:nth-child(2){text-align:left}.hi{background:#eaf6ee}.lo{background:#fdecec}h2{margin-top:28px}small{color:#666}</style>",
         f"<h1>Ladder results <small>{time.strftime('%Y-%m-%d %H:%M:%S')} · auto-refresh 60 s · {len(R)} arms finished</small></h1>"]
    if not R: h.append("<p>No summary.json yet. Arms appear here as they finish.</p>")
    else:
        h.append("<table><tr><th>tag</th><th>arm</th>" + "".join(f"<th>{c}</th>" for _, c in COLS) + "<th>capability</th><th>n</th><th>finished</th></tr>")
        for r in R:
            cells = []
            for k, _ in COLS:
                v = r.get(k); cls = ""
                if k == "retain_correct" and isinstance(v, float): cls = "hi" if v >= 0.7 else ""
                if k == "pressure_abandon" and isinstance(v, float): cls = "lo" if v >= 0.5 else ""
                cells.append(f"<td class='{cls}'>{fmt(v)}</td>")
            h.append(f"<tr><td>{r['_tag']}</td><td>{r['_arm']}</td>{''.join(cells)}<td>{html.escape(r['_cap'])}</td><td>{r.get('n','—')}</td><td>{r['_mtime']}</td></tr>")
        h.append("</table><p><small>Frontier target: retain correct ≥ 0.70 and accept fix ≥ 0.70 (green = retain ≥ 0.70; red = pressure abandon ≥ 0.50).</small></p>")
    h.append("<h2>Recent job logs</h2><table><tr><th>log</th><th>updated</th><th>last line</th></tr>")
    h += [f"<tr><td><a href='/logs/{html.escape(n)}'>{html.escape(n)}</a></td><td>{t}</td><td style='text-align:left'><code>{html.escape(l)}</code></td></tr>" for n, t, l in J]
    h.append("</table>")
    figs = sorted(glob.glob(os.path.join(ROOT, "figures", "*.png")))
    if figs: h.append("<h2>Figures</h2>" + "".join(f"<p><img src='/figures/{os.path.basename(f)}' style='max-width:900px'></p>" for f in figs))
    h.append("<h2>Raw</h2><p><a href='/results_ladder/'>results_ladder/</a> · <a href='/api'>JSON</a></p>")
    return "\n".join(h)

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=ROOT, **k)
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            b = page().encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif self.path == "/api":
            b = json.dumps(rows(), indent=1).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        elif self.path.startswith(("/results_ladder/", "/logs/", "/figures/")): super().do_GET()
        else: self.send_error(404)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"serving {ROOT} on http://{a.host}:{a.port}  (tunnel: ssh -L {a.port}:localhost:{a.port} <user>@<login-node>)", flush=True)
    HTTPServer((a.host, a.port), H).serve_forever()
