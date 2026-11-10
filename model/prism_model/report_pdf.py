"""
Render an arsenal report (``arsenal.analyze`` output) as a PDF for players and coaches.

Laid out like the Garnet Pitch Lab pages: a Development card, pitch by pitch,
the shape picture with a fastball card, a card for the pitch that needs work,
the arsenal plan, counts, grades explained, and progress through the season.
The page is HTML printed with headless Chromium, so the same template can feed
a dashboard later.

    python -m prism_model.report_pdf --reports reps.pkl --name "Grayson Black"
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from prism_model.fastball_traits import ordinal

CRIMSON = "#8a1f2d"
GOLD = "#e6a817"
PITCH_COLORS = {"Four-seam": "#d22d3d", "Sinker": "#f08a24", "Cutter": "#8a4b2c", "Slider": "#2b2b2b",
                "Sweeper": "#5b5b5b", "Curveball": "#2f6db5", "Changeup": "#2e9e5b", "Splitter": "#14a3a3"}
VERDICT_TONE = {"Working": "ok", "Struggling": "bad", "No clear job yet": "warn", "Too few game pitches": "muted",
                "Bullpen only": "muted", "Rarely used": "muted"}


def _e(x) -> str:
    return "–" if x is None or x == "N/A" else html.escape(str(x))


def _color(pitch: str) -> str:
    return PITCH_COLORS.get(pitch, "#777")


def grade_cell(v) -> str:
    if not isinstance(v, (int, float)):
        return '<td class="g na">N/A</td>'
    tone = "lo" if v <= 92 else "hi" if v >= 108 else ""
    return f'<td class="g {tone}">{int(round(v))}</td>'


CSS = f"""
@page {{ size: Letter; margin: 0.5in 0.55in 0.6in; }}
* {{ box-sizing: border-box; }}
body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #1d1d1f; font-size: 9pt; line-height: 1.42; margin: 0; }}
.serif {{ font-family: Georgia, 'Times New Roman', serif; }}
.top {{ display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 8px; }}
.top h1 {{ font-family: Georgia, serif; color: {CRIMSON}; font-size: 25pt; margin: 0; line-height: 1.05; }}
.top .meta {{ color: #666; font-size: 9pt; margin-top: 2px; }}
.top .brand {{ text-align: right; font-size: 8pt; color: #777; }}
.top .brand b {{ font-family: Georgia, serif; color: {CRIMSON}; font-size: 13pt; letter-spacing: 2px; }}
.lede {{ font-size: 10.5pt; color: #333; margin: 2px 0 12px; }}
h2.step {{ font-family: Georgia, serif; font-size: 13pt; margin: 16px 0 6px; }}
.sec {{ break-inside: avoid; }}
.card {{ border: 1px solid #e3dede; border-radius: 8px; padding: 11px 14px 12px; margin: 0 0 6px; break-inside: avoid;
         background: linear-gradient(#fbf7f7, #fff 40px); }}
.card.accent {{ border-color: #d9aeb4; border-top: 3px solid {CRIMSON}; }}
.card h3 {{ font-family: Georgia, serif; font-size: 12.5pt; margin: 0 0 2px; }}
.card .sub {{ color: #777; font-size: 7.8pt; margin-bottom: 8px; }}
.tiles {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-bottom: 8px; }}
.tile {{ background: #f4f2f2; border-radius: 6px; padding: 7px 9px; }}
.tile .k {{ font-size: 7.5pt; color: #777; }} .tile .v {{ font-weight: 700; font-size: 10.5pt; margin: 1px 0 2px; }}
.tile .d {{ font-size: 7.8pt; color: #444; }}
.label {{ font-size: 7.5pt; letter-spacing: 1px; color: #888; text-transform: uppercase; margin: 9px 0 4px; font-weight: 600; }}
.item {{ border-left: 3px solid {CRIMSON}; background: #fbf6f6; padding: 5px 9px; margin: 4px 0; border-radius: 0 4px 4px 0; }}
.item.good {{ border-left-color: #2e8b57; background: #f4faf6; }}
.item b.t {{ font-size: 9pt; }}
.pill {{ display: inline-block; font-size: 7pt; padding: 1px 7px; border-radius: 9px; background: #eee; color: #444;
         margin-left: 5px; vertical-align: 1px; font-weight: 600; }}
.pill.ok {{ background: #e3f3e8; color: #22663f; }} .pill.bad {{ background: #f8e1e4; color: {CRIMSON}; }}
.pill.warn {{ background: #fbf0d6; color: #8a6200; }} .pill.muted {{ background: #eee; color: #666; }}
.pitch {{ border-left: 3px solid #999; padding: 3px 0 3px 10px; margin: 8px 0; }}
.pitch .name {{ font-weight: 700; font-size: 10pt; }}
.pitch .pp {{ font-size: 7.8pt; color: #555; margin-left: 6px; }}
.pitch p {{ margin: 2px 0; }}
ul.sit {{ margin: 3px 0 0; padding-left: 15px; }} ul.sit li {{ margin: 1px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 4px 0; }}
th {{ text-align: left; font-size: 7pt; letter-spacing: 0.8px; text-transform: uppercase; color: {CRIMSON};
      border-bottom: 1.5px solid {CRIMSON}; padding: 3px 6px; font-weight: 700; }}
td {{ padding: 3.5px 6px; border-bottom: 1px solid #eee; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.g {{ text-align: center; font-weight: 700; font-variant-numeric: tabular-nums; }}
td.g.lo {{ color: {CRIMSON}; background: #f8e6e8; }} td.g.hi {{ color: #1f6b40; background: #e6f4ec; }}
td.g.na {{ color: #aaa; font-weight: 400; }}
td.in {{ color: #1f6b40; font-weight: 700; }} td.out {{ color: {CRIMSON}; font-weight: 700; }}
.why {{ font-size: 8pt; color: #333; margin: 1px 0 7px; }}
.why .muted {{ color: #777; }}
.dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; vertical-align: 0; }}
.cols2 {{ display: grid; grid-template-columns: 1.15fr 1fr; gap: 12px; align-items: start; }}
.legend div {{ margin: 0 0 6px; font-size: 8.2pt; }} .legend .s {{ color: #666; font-size: 7.6pt; margin-left: 14px; }}
.note {{ color: #777; font-size: 7.6pt; margin-top: 4px; }}
.fbline {{ display: flex; gap: 6px; flex-wrap: wrap; margin: 6px 0; }}
.stat {{ background: #f4f2f2; border-radius: 5px; padding: 4px 8px; font-size: 8pt; }}
.stat b {{ display: block; font-size: 10pt; }}
.charts {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 4px 0 6px; }}
.chart .t {{ font-size: 6.8pt; letter-spacing: 0.8px; color: #888; text-transform: uppercase; font-weight: 700; }}
.plan {{ font-size: 9.6pt; }}
.oneline {{ background: #faf6ef; border-left: 4px solid {GOLD}; padding: 9px 13px; font-size: 10pt; margin-top: 14px; break-inside: avoid; }}
"""


# --------------------------------------------------------------------------
# Charts (inline SVG)
# --------------------------------------------------------------------------
def movement_svg(points: list[dict], profile: list[dict], size: int = 270) -> str:
    lim = 25
    pad = 26
    span = size - 2 * pad
    X = lambda hb: pad + (hb + lim) / (2 * lim) * span
    Y = lambda ivb: pad + (lim - ivb) / (2 * lim) * span
    s = [f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg" '
         'font-family="Helvetica,Arial,sans-serif">']
    for v in (-20, -10, 10, 20):
        s.append(f'<line x1="{X(v)}" y1="{pad}" x2="{X(v)}" y2="{size - pad}" stroke="#f0eded"/>')
        s.append(f'<line x1="{pad}" y1="{Y(v)}" x2="{size - pad}" y2="{Y(v)}" stroke="#f0eded"/>')
        s.append(f'<text x="{X(v)}" y="{size - pad + 11}" font-size="7" fill="#999" text-anchor="middle">{v:+d}</text>')
        s.append(f'<text x="{pad - 4}" y="{Y(v) + 2.5}" font-size="7" fill="#999" text-anchor="end">{v:+d}</text>')
    s.append(f'<line x1="{X(0)}" y1="{pad}" x2="{X(0)}" y2="{size - pad}" stroke="#aaa"/>')
    s.append(f'<line x1="{pad}" y1="{Y(0)}" x2="{size - pad}" y2="{Y(0)}" stroke="#aaa"/>')
    s.append(f'<text x="{pad}" y="{pad - 8}" font-size="7.5" fill="#666">Ride ↑ (IVB, in)</text>')
    s.append(f'<text x="{size - pad}" y="{size - 4}" font-size="7.5" fill="#666" text-anchor="end">Arm side (HB, in) →</text>')
    s.append(f'<text x="{pad}" y="{size - 4}" font-size="7.5" fill="#666">← Glove side</text>')
    for p in points:
        hb, ivb = max(-lim, min(lim, p["hb_arm"])), max(-lim, min(lim, p["InducedVertBreak"]))
        s.append(f'<circle cx="{X(hb):.1f}" cy="{Y(ivb):.1f}" r="2.2" fill="{_color(p["shape_family"])}" fill-opacity="0.35"/>')
    placed = []
    for m in profile:
        if m.get("ivb") is None or m.get("hb_arm") is None:
            continue
        x, y = X(float(m["hb_arm"])), Y(float(m["ivb"]))
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{_color(m["pitch"])}" stroke="#fff" stroke-width="1.5"/>')
        ly = y + 3
        while any(abs(ly - py) < 10 and abs(x - px) < 60 for px, py in placed):
            ly += 10  # nudge labels of pitches that sit on top of each other
        placed.append((x, ly))
        s.append(f'<text x="{x + 9:.1f}" y="{ly:.1f}" font-size="8" font-weight="700" fill="#222">{_e(m["pitch"])}</text>')
    s.append("</svg>")
    return "".join(s)


def line_svg(values: list, labels: list[str], color: str, w: int = 125, h: int = 62, band=None) -> str:
    vals = [v for v in values if isinstance(v, (int, float))]
    if len(vals) < 2:
        return '<div class="note">Not enough sessions</div>'
    lo, hi = min(vals), max(vals)
    if band:
        lo, hi = min(lo, band[0]), max(hi, band[1])
    rng = (hi - lo) or 1
    lo, hi = lo - 0.15 * rng, hi + 0.15 * rng
    px = lambda i: 8 + i * (w - 16) / max(len(values) - 1, 1)
    py = lambda v: 6 + (hi - v) / (hi - lo) * (h - 18)
    s = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" font-family="Helvetica,Arial,sans-serif">']
    if band:
        s.append(f'<rect x="4" y="{py(band[1]):.1f}" width="{w - 8}" height="{py(band[0]) - py(band[1]):.1f}" fill="#e3f3e8"/>')
    pts = [(px(i), py(v)) for i, v in enumerate(values) if isinstance(v, (int, float))]
    s.append('<polyline fill="none" stroke="{}" stroke-width="1.6" points="{}"/>'.format(
        color, " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)))
    for (x, y), v in zip(pts, vals):
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.4" fill="{color}"/>')
    s.append(f'<text x="2" y="{h - 1}" font-size="6.5" fill="#999">{_e(labels[0])}</text>')
    s.append(f'<text x="{w - 2}" y="{h - 1}" font-size="6.5" fill="#999" text-anchor="end">{_e(labels[-1])}</text>')
    s.append(f'<text x="{w - 2}" y="8" font-size="6.5" fill="#999" text-anchor="end">{max(vals):g}–{min(vals):g}</text>')
    s.append("</svg>")
    return "".join(s)


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------
def development_card(r: dict) -> str:
    d, c = r["development"], r["development"]["command"]
    st = d["stuff"]
    tog = d["pitches_together"]
    cmd_d = ", ".join(x for x in [
        f"{c['strike_pct']:.0f}% strikes" if c.get("strike_pct") is not None else None,
        f"{c['bb_pct']:.0f}% walks in {c['pa']} PA" if c.get("bb_pct") is not None else None,
        f"Location+ {c['location_plus']}" if c.get("location_plus") is not None else None] if x)
    H = ['<div class="card accent"><h3>Development</h3>'
         '<div class="sub">Who he is as a pitcher and what\'s holding him back, from his fall outings.</div>',
         '<div class="tiles">',
         f'<div class="tile"><div class="k">Stuff</div><div class="v">{_e(str(st["label"]).capitalize())}</div>'
         f'<div class="d">Stuff+ {_e(st["stuff_plus"])} across his pitches</div></div>',
         f'<div class="tile"><div class="k">How his pitches work together</div><div class="v">'
         f'{len(r["pitch_by_pitch"])} pitches</div><div class="d">{_e("; ".join(tog[:2]))}</div></div>',
         f'<div class="tile"><div class="k">Command</div><div class="v">{_e(str(c.get("label", "Not enough data")).capitalize())}'
         f'</div><div class="d">{_e(cmd_d)}</div></div></div>']
    H.append('<div class="label">What makes him</div>')
    H += [f'<div class="item good">{_e(x)}</div>' for x in d["what_makes_him"]] or ['<div class="note">No clear strength yet.</div>']
    H.append('<div class="label">What\'s holding him back</div>')
    if d["holding_back"]:
        for i, x in enumerate(d["holding_back"], 1):
            H.append(f'<div class="item"><b class="t">{i}.</b> {_e(x)}</div>')
    else:
        H.append('<div class="note">Nothing clearly holding him back in this data.</div>')
    if d["priority_order"]:
        H.append('<div class="label">What helps, in order</div><ol style="margin:0;padding-left:18px">' +
                 "".join(f"<li>{_e(x)}</li>" for x in d["priority_order"]) + "</ol>")
    H.append("</div>")
    return "".join(H)


def pitch_by_pitch_card(r: dict) -> str:
    H = ['<div class="card"><div class="label" style="margin-top:0">Pitch by pitch</div>']
    for p in r["pitch_by_pitch"]:
        v = p.get("verdict", "")
        g = p["grades"]
        H.append(f'<div class="pitch" style="border-left-color:{_color(p["pitch"])}">'
                 f'<span class="name">{_e(p["pitch"])}</span>'
                 f'<span class="pp">Pitch+ {_e(g["pitch_plus"])}</span>'
                 f'<span class="pill {VERDICT_TONE.get(v, "muted")}">{_e(v.lower())}</span>')
        parts = []
        if p.get("jobs"):
            j = p["jobs"][0]
            parts.append(f"<b>Its job:</b> {_e(j['job'].lower())} ({_e(j['evidence'])}).")
        if p.get("great"):
            parts.append(f"<b>Also:</b> {_e('; '.join(x[0].lower() + x[1:] for x in p['great']))}.")
        if not p.get("jobs") and p.get("verdict") not in ("Rarely used",):
            parts.append("<b>What's great:</b> nothing clearly above the staff yet.")
        fails = [f"{b['issue'].lower()} ({b['evidence']})" for b in p.get("deal_breakers", [])]
        if fails:
            parts.append(f"<b>Where it's failing:</b> {_e('; '.join(fails))}.")
        if p.get("not_strength"):
            parts.append(f"<b>Not its strength:</b> {_e('; '.join(p['not_strength']))}.")
        parts.append(f"<b>So:</b> {_e(p.get('so', ''))}")
        H.append(f"<p>{' '.join(parts)}</p></div>")
    s = r["situations"]
    sit = []
    fade = s.get("fade", {})
    if fade.get("strike_late") is not None:
        t = f"Pitches 1–25 of an outing: {fade['strike_early']:.0f}% strikes; after: {fade['strike_late']:.0f}%"
        if fade.get("fb_velo_late") is not None:
            t += f" (fastball {fade['fb_velo_early']} → {fade['fb_velo_late']} mph)"
        sit.append(t + ".")
    fp = s.get("first_pitch_strike")
    if fp:
        sit.append(f"First-pitch strikes: {fp['pct']}% of {fp['n']} (MLB middle half {fp['pro_range']}).")
    if sit:
        H.append('<div class="label">Situations</div><ul class="sit">' + "".join(f"<li>{_e(x)}</li>" for x in sit) + "</ul>")
    H.append("</div>")
    return "".join(H)


def shape_cards(r: dict) -> str:
    prof = r["movement_profile"]
    legend = []
    for m in prof:
        legend.append(f'<div><span class="dot" style="background:{_color(m["pitch"])}"></span><b>{_e(m["pitch"])}</b>'
                      f'<div class="s">{_e(m["velo"])} mph · IVB {_e(m["ivb"])}" · HB {_e(m["hb_arm"])}" · '
                      f'{_e(m["spin"])} rpm · tilt {_e(m["tilt"])}</div></div>')
    H = ['<div class="card"><h3>Movement profile</h3>'
         '<div class="sub">Every pitch from his fall outings; large markers are each pitch\'s average. '
         'HB is arm side positive.</div>',
         f'<div class="cols2"><div>{movement_svg(r.get("points", []), prof)}</div>'
         f'<div class="legend"><div class="label" style="margin-top:0">Pitches</div>{"".join(legend)}</div></div></div>']
    t, role = r.get("fastball_traits"), r.get("fastball_role")
    if t or role:
        fb = t["pitch"] if t else role["pitch"]
        H.append(f'<div class="card"><h3>{_e(fb)} <span style="font-family:Helvetica;font-size:8pt;color:#777;font-weight:400">'
                 f'{_e(role["subtype"].lower() + " fastball") if role and role.get("subtype") else ""}</span></h3>')
        if role:
            H.append(f'<p style="margin:2px 0 6px">{_e(role["role"])}</p>')
        if t:
            stats = [("Velocity", f"{t['velo']} mph", f"{ordinal(t['velo_pct_d3'])} pct D3" if t.get("velo_pct_d3") is not None else ""),
                     ("Extension", f"{t['extension_ft']} ft", f"{ordinal(t['extension_pct_mlb'])} pct MLB"),
                     ("Perceived", f"{t['perceived_velo']} mph",
                      f"{ordinal(t['perceived_pct_d3'])} pct D3" if t.get("perceived_pct_d3") is not None else "")]
            if t.get("show_vaa"):
                stats.append(("Approach angle", f"{t['vaa_above_expected']:+.1f}° flatter" if t["vaa_above_expected"] > 0
                              else f"{t['vaa_above_expected']:+.1f}°", f"{ordinal(t['vaa_ae_pct_mlb'])} pct MLB"))
            H.append('<div class="fbline">' + "".join(
                f'<div class="stat">{k}<b>{v}</b>{_e(sub)}</div>' for k, v, sub in stats) + "</div>")
            if t.get("show_vaa"):
                H.append('<div class="note">Approach angle is compared with an average fastball at his velocity and plate '
                         'height, so it shows what his slot, extension and carry add. It\'s shown when it helps him.</div>')
        H.append("</div>")
    return "".join(H)


def needs_work_card(r: dict) -> str:
    w = r.get("needs_work")
    if not w:
        return ""
    p = next(x for x in r["pitch_by_pitch"] if x["pitch"] == w["pitch"])
    sh = p.get("shape", {})
    H = [f'<div class="card accent"><h3>{_e(w["pitch"])} <span class="pill {VERDICT_TONE.get(w["verdict"], "muted")}">'
         f'{_e(w["verdict"].lower())}</span></h3>'
         f'<div class="sub">Avg {_e(sh.get("velo"))} mph, IVB {_e(sh.get("ivb"))}", HB {_e(sh.get("hb_arm"))}" · '
         f'Pitch+ {_e(p["grades"]["pitch_plus"])}</div>']
    t = w.get("targets")
    if t:
        H.append('<div class="label">Right now → target</div><table><tr><th>Shape</th><th class="num">Right now</th>'
                 '<th class="num">Target (versions that work)</th></tr>')
        for row in t["rows"]:
            cls = "in" if row["in_range"] else "out"
            H.append(f'<tr><td>{_e(row["label"])}</td><td class="num {cls}">{row["now"]:+.1f} {row["unit"]}</td>'
                     f'<td class="num">{row["lo"]:+.1f} to {row["hi"]:+.1f} {row["unit"]}</td></tr>')
        rates = p.get("rates") or {}
        if rates.get("strike") is not None:
            ref = (p.get("rates_ref") or {}).get("strike")
            ok = ref is not None and rates["strike"] >= ref
            H.append(f'<tr><td>Strike %</td><td class="num {"in" if ok else "out"}">{rates["strike"]:.0f}%</td>'
                     f'<td class="num">{ref:.0f}%+</td></tr>' if ref is not None else "")
        H.append("</table>")
        H.append(f'<div class="note">Target: the middle half of MLB {w["pitch"].lower()}s from pitchers with a similar release '
                 f'height whose version worked ({t["n_comps"]} pitcher-seasons). Green is in range, red is outside it.</div>')
        H.append(f'<p style="margin:6px 0 0"><b>The plan:</b> {_e(t["plan"])}</p>')
    else:
        H.append(f'<p>{_e(p.get("so", ""))}</p>')
    H.append("</div>")
    return "".join(H)


def arsenal_card(r: dict) -> str:
    rows = []
    for p in r["pitch_by_pitch"]:
        rt = p.get("rates")
        if not rt or not p["game_pitches"]:
            continue
        f = lambda k: f"{rt[k]:.0f}%" if rt.get(k) is not None else "–"
        rows.append(f'<tr><td><span class="dot" style="background:{_color(p["pitch"])}"></span><b>{_e(p["pitch"])}</b> '
                    f'<span class="note">n={p["game_pitches"]}</span></td><td class="num">{f("strike")}</td>'
                    f'<td class="num">{f("whiff")}</td><td class="num">{f("called_strike")}</td><td class="num">{f("chase")}</td>'
                    f'<td class="num">{f("putaway")}</td></tr>')
    H = [f'<div class="card accent"><h3>Arsenal recommendation</h3><p class="plan" style="margin:2px 0 0">'
         f'{_e(r.get("arsenal_plan", ""))}</p></div>']
    if rows:
        H.append('<div class="card"><h3>Real results</h3><div class="sub">Against hitters in fall intrasquads (AWRE).</div>'
                 '<table><tr><th>Pitch</th><th class="num">Strike %</th><th class="num">Whiff % (of swings)</th>'
                 '<th class="num">Called strike %</th><th class="num">Chase %</th><th class="num">Putaway %</th></tr>'
                 + "".join(rows) + "</table></div>")
    return "".join(H)


def counts_card(r: dict) -> str:
    s = r["situations"]
    if not s.get("by_count"):
        return ""
    H = ['<div class="card"><h3>By count</h3>']
    fp = s.get("first_pitch_strike")
    if fp:
        H.append(f'<div class="sub">First-pitch strikes {fp["pct"]}% of {fp["n"]} (MLB middle half {fp["pro_range"]}).</div>')
    H.append('<table><tr><th>Count</th><th class="num">n</th><th class="num">Strike %</th><th class="num">Whiff %</th><th>Most used</th></tr>')
    for row in s["by_count"]:
        H.append(f'<tr><td>{_e(row["count"])}</td><td class="num">{row["n"]}</td><td class="num">{_e(row["strike_pct"])}</td>'
                 f'<td class="num">{_e(row["whiff_pct"])}</td><td>{_e(row["most_used"])}</td></tr>')
    H.append("</table>")
    if s.get("splits"):
        fmt = lambda v: "–" if v is None else f"{v:.3f}".lstrip("0")
        H.append('<table style="margin-top:8px"><tr><th>Split</th><th class="num">PA</th><th class="num">AVG</th>'
                 '<th class="num">OBP</th><th class="num">SLG</th><th class="num">K%</th><th class="num">BB%</th><th>Mix</th></tr>')
        for label, sp in s["splits"].items():
            mix = ", ".join(f"{k} {v}%" for k, v in sp.get("mix", {}).items())
            H.append(f'<tr><td>{label}</td><td class="num">{sp["pa"]}</td><td class="num">{fmt(sp.get("avg"))}</td>'
                     f'<td class="num">{fmt(sp.get("obp"))}</td><td class="num">{fmt(sp.get("slg"))}</td>'
                     f'<td class="num">{_e(sp.get("k_pct"))}</td><td class="num">{_e(sp.get("bb_pct"))}</td><td>{_e(mix)}</td></tr>')
        H.append("</table>")
    H.append("</div>")
    return "".join(H)


def grades_card(r: dict) -> str:
    H = ['<div class="card"><h3>Pitch grades</h3>'
         '<div class="sub"><b>Pitch+</b> combines three parts, each graded so <b>100 = average</b> and every 10 points is a real step. '
         '<b>Stuff</b> is the pitch itself (velocity, movement, spin, release); velocity is judged against D3 pitchers, the shape '
         'against MLB pitches of its type. <b>Location</b> is how often it\'s competitive and stays out of the middle. '
         '<b>Results</b> is what happened against hitters, and counts more as his sample grows.</div>',
         '<table><tr><th>Pitch</th><th style="text-align:center">Pitch+</th><th style="text-align:center">Stuff</th>'
         '<th style="text-align:center">Shape</th><th style="text-align:center">Location</th>'
         '<th style="text-align:center">Results</th><th class="num">Total n</th><th class="num">Game n</th></tr>']
    for p in r["pitch_by_pitch"]:
        g = p["grades"]
        H.append(f'<tr><td><span class="dot" style="background:{_color(p["pitch"])}"></span><b>{_e(p["pitch"])}</b></td>'
                 f'{grade_cell(g["pitch_plus"])}{grade_cell(g["stuff_plus"])}{grade_cell(g["shape_plus"])}'
                 f'{grade_cell(g["location_plus"])}{grade_cell(g["results_plus"])}'
                 f'<td class="num">{p["n"]}</td><td class="num">{p["game_pitches"]}</td></tr>')
        if not isinstance(g["pitch_plus"], (int, float)):
            continue
        why = (f"<b>Pitch+ {g['pitch_plus']}:</b> built like a {_e(g['stuff_plus'])} (Stuff), located like a "
               f"{_e(g['location_plus'])}, plays like a {_e(g['results_plus'])} in games. With {p['game_pitches']} game "
               f"pitches, results count for {p.get('results_weight', 0)}% of it.")
        parts = p.get("stuff_parts")
        if parts:
            why += (f" <b>Stuff {_e(g['stuff_plus'])}</b> vs a typical {p['pitch'].lower()}: " +
                    ", ".join(f"{k} {v:+d}" for k, v in parts.items()) + ".")
        if p.get("comparable_mlb"):
            why += f' <span class="muted">{_e(p["comparable_mlb"]["text"])}</span>'
        H.append(f'<tr><td colspan="8" class="why">{why}</td></tr>')
    H.append('</table><div class="note">Green is clearly above average (108+), red clearly below (92 or lower). '
             'Pitch types need 10 pitches to be graded.</div></div>')
    return "".join(H)


def progress_cards(r: dict) -> str:
    prog = r.get("progress") or []
    by_pitch: dict[str, list] = {}
    for x in prog:
        by_pitch.setdefault(x["pitch"], []).append(x)
    H = []
    for pitch, rows in by_pitch.items():
        if len(rows) < 2:
            continue
        col = _color(pitch)
        lab = [str(x["outing"])[5:] for x in rows]
        charts = [("Stuff+", [x["stuff_plus"] for x in rows]), ("Velo", [x["velo"] for x in rows]),
                  ("IVB (ride)", [x["ivb"] for x in rows]), ("HB (run)", [x["hb"] for x in rows])]
        H.append(f'<div class="card"><h3><span class="dot" style="background:{col}"></span>{_e(pitch)} through the fall</h3>'
                 '<div class="charts">' + "".join(
                     f'<div class="chart"><div class="t">{k}</div>{line_svg(v, lab, col)}</div>' for k, v in charts) +
                 '</div><table><tr><th>Date</th><th class="num">n</th><th class="num">Velo</th><th class="num">IVB</th>'
                 '<th class="num">HB</th><th class="num">Spin</th><th style="text-align:center">Stuff+</th></tr>')
        for x in rows:
            H.append(f'<tr><td>{_e(x["outing"])}</td><td class="num">{x["n"]}</td><td class="num">{_e(x["velo"])}</td>'
                     f'<td class="num">{_e(x["ivb"])}"</td><td class="num">{_e(x["hb"])}"</td><td class="num">{_e(x["spin"])}</td>'
                     f'{grade_cell(x["stuff_plus"])}</tr>')
        H.append("</table></div>")
    return "".join(H)


def to_html(r: dict, scale_note: str = "") -> str:
    def sec(title: str, body: str) -> str:
        # Heading and its first card stay on one page; the rest may flow.
        cut = body.find('<div class="card', 1)
        first, rest = (body, "") if cut == -1 else (body[:cut], body[cut:])
        return f'<div class="sec"><h2 class="step">{title}</h2>{first}</div>{rest}'

    H = [f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>",
         f'<div class="top"><div><h1>{_e(r["pitcher"])}</h1><div class="meta">{_e(r["throws"])}HP · {r["pitches"]} pitches · '
         f'{r["game_pitches"]} against hitters · {r["outings"]} outings</div></div>'
         '<div class="brand"><b>PRISM</b><br>Oberlin Baseball pitcher report</div></div>',
         f'<div class="lede">{_e(r["summary"])}</div>',
         development_card(r),
         sec("Pitch by pitch", pitch_by_pitch_card(r)),
         sec("The shape picture", shape_cards(r))]
    if r.get("needs_work"):
        H.append(sec("The pitch that needs work", needs_work_card(r)))
    H += [sec("The arsenal plan", arsenal_card(r)), sec("The counts", counts_card(r)),
          sec("The grades, explained", grades_card(r)), sec("Progress through the fall", progress_cards(r))]
    d = r["development"]
    if d["priority_order"]:
        H.append(f'<div class="oneline"><b class="serif">His page in one sentence.</b> {_e(r["summary"])} '
                 f'{_e(r.get("arsenal_plan", ""))}</div>')
    H.append("</body></html>")
    return "".join(H)


def write_pdf(r: dict, path: Path, scale_note: str = "") -> Path:
    from playwright.sync_api import sync_playwright

    path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.set_content(to_html(r, scale_note), wait_until="load")
        foot = ('<div style="font-size:7px;color:#999;width:100%;padding:0 0.55in;display:flex;justify-content:space-between;'
                'font-family:Helvetica,Arial,sans-serif"><span>PRISM · Oberlin Baseball</span>'
                '<span><span class="pageNumber"></span></span></div>')
        page.pdf(path=str(path), format="Letter", print_background=True, display_header_footer=True,
                 header_template="<div></div>", footer_template=foot,
                 margin={"top": "0.5in", "bottom": "0.6in", "left": "0.55in", "right": "0.55in"})
        browser.close()
    return path


def main() -> None:
    import pickle

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reports", required=True, help="pickle of arsenal.analyze output")
    ap.add_argument("--name", help="only this pitcher")
    ap.add_argument("--out", default="pdf_reports")
    args = ap.parse_args()
    from prism_model.arsenal import SCALE_NOTE, _slug

    for r in pickle.load(open(args.reports, "rb")):
        if args.name and r["pitcher"] != args.name:
            continue
        print(write_pdf(r, Path(args.out) / f"{_slug(r['pitcher'])}.pdf", SCALE_NOTE.get(r["scale"], "")))


if __name__ == "__main__":
    main()
