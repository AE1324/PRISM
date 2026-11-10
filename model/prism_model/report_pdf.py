"""
Render an arsenal report (``arsenal.analyze`` output) as a PDF for players and coaches.

The page is built as HTML and printed with headless Chromium (Playwright), so the
layout lives in one template here and the same HTML can later feed a dashboard.

    python -m prism_model.report_pdf --name "Grayson Black"
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path

from prism_model.fastball_traits import ordinal

CRIMSON = "#a6192e"
GOLD = "#ffc72c"


def _e(x) -> str:
    return "–" if x is None or x == "N/A" else html.escape(str(x))


def grade_cell(v) -> str:
    """Grade with a background that runs blue (below 100) to red (above 100)."""
    if not isinstance(v, (int, float)):
        return '<td class="g na">–</td>'
    d = max(-30, min(30, v - 100)) / 30
    if d >= 0:
        bg = f"rgba(166,25,46,{0.08 + 0.55 * d:.2f})"
    else:
        bg = f"rgba(37,99,175,{0.08 + 0.55 * -d:.2f})"
    color = "#fff" if abs(d) > 0.6 else "#111"
    return f'<td class="g" style="background:{bg};color:{color}">{int(round(v))}</td>'


CSS = f"""
@page {{ size: Letter; margin: 0.45in 0.5in 0.55in; }}
* {{ box-sizing: border-box; }}
body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #1a1a1a; font-size: 9.6pt; line-height: 1.38; margin: 0; }}
header {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 3px solid {CRIMSON}; padding-bottom: 6px; margin-bottom: 10px; }}
header h1 {{ margin: 0; font-size: 22pt; letter-spacing: -0.3px; }}
header .meta {{ color: #555; font-size: 9pt; }}
header .brand {{ text-align: right; font-size: 8.5pt; color: #555; }}
header .brand b {{ color: {CRIMSON}; font-size: 11pt; letter-spacing: 1px; }}
.summary {{ background: #faf6ef; border-left: 4px solid {GOLD}; padding: 8px 12px; font-size: 10.5pt; margin: 0 0 10px; }}
h2 {{ font-size: 11.5pt; color: {CRIMSON}; text-transform: uppercase; letter-spacing: 0.8px; margin: 14px 0 6px; border-bottom: 1px solid #e4d9d9; padding-bottom: 2px; }}
h3 {{ font-size: 10.5pt; margin: 10px 0 3px; }}
h3 .verdict {{ font-weight: normal; color: #666; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 18px; }}
.box h4 {{ margin: 4px 0 3px; font-size: 9.6pt; }}
ul {{ margin: 0; padding-left: 15px; }} li {{ margin: 1px 0; }}
ol {{ margin: 0; padding-left: 18px; }} ol li {{ margin: 2px 0; }}
.kv {{ display: flex; flex-wrap: wrap; gap: 4px 18px; }} .kv span b {{ color: #333; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 3px 0 6px; }}
th {{ background: #f1ecec; text-align: left; padding: 3px 5px; font-weight: 600; border-bottom: 1px solid #d8cccc; }}
td {{ padding: 3px 5px; border-bottom: 1px solid #eee; }}
td.g {{ text-align: center; font-weight: 600; width: 46px; }}
td.na {{ color: #999; font-weight: normal; }}
.note {{ color: #666; font-size: 8pt; }}
.pitch {{ break-inside: avoid; border: 1px solid #eee; border-radius: 4px; padding: 5px 9px 6px; margin: 6px 0; }}
.pitch .line {{ color: #444; margin: 1px 0 3px; }}
.so {{ margin-top: 3px; }}
.tag {{ display: inline-block; font-size: 7.8pt; padding: 0 5px; border-radius: 8px; background: #eee; color: #444; margin-left: 4px; }}
.good {{ color: #1d6b37; }} .bad {{ color: {CRIMSON}; }}
.pagebreak {{ break-before: page; }}
"""


def _list(items, cls="") -> str:
    if not items:
        return '<p class="note">Nothing clear in this data yet.</p>'
    return f'<ul class="{cls}">' + "".join(f"<li>{_e(x)}</li>" for x in items) + "</ul>"


def to_html(r: dict, scale_note: str = "") -> str:
    d = r["development"]
    c = d["command"]
    H = [f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"]
    H.append(f"""<header><div><h1>{_e(r['pitcher'])} <span style="font-weight:300;color:#666">{_e(r['throws'])}HP</span></h1>
      <div class="meta">{r['pitches']} pitches · {r['game_pitches']} vs hitters · {r['outings']} outings</div></div>
      <div class="brand"><b>PRISM</b><br>Oberlin Baseball pitch report</div></header>""")
    H.append(f'<div class="summary">{_e(r["summary"])}</div>')

    # Development
    H.append("<h2>Development</h2>")
    cmd = [f"{c['strike_pct']:.0f}% strikes" if c.get("strike_pct") is not None else None,
           f"{c['bb_pct']:.0f}% walks ({c['pa']} PA)" if c.get("bb_pct") is not None else None,
           f"{c['location_plus']} Location+" if c.get("location_plus") is not None else None]
    H.append('<div class="kv">'
             f"<span><b>Stuff:</b> {_e(d['stuff']['label'])} ({_e(d['stuff']['stuff_plus'])} Stuff+)</span>"
             f"<span><b>Command:</b> {_e(c.get('label', 'not enough data'))} ({', '.join(x for x in cmd if x)})</span></div>")
    if d["pitches_together"]:
        H.append(f"<p style='margin:4px 0'><b>How the pitches work together:</b> {_e('; '.join(d['pitches_together']))}</p>")
    H.append('<div class="cols">')
    H.append(f'<div class="box"><h4 class="good">What makes him</h4>{_list(d["what_makes_him"])}</div>')
    H.append(f'<div class="box"><h4 class="bad">What\'s holding him back</h4>{_list(d["holding_back"])}</div>')
    H.append("</div>")
    if d["priority_order"]:
        H.append("<h4 style='margin:8px 0 3px'>Priority order</h4><ol>" +
                 "".join(f"<li>{_e(x)}</li>" for x in d["priority_order"]) + "</ol>")

    # Grades
    H.append("<h2>Pitch grades</h2><table><tr><th>Pitch</th><th>Type</th><th>n</th><th>vs hitters</th><th>Usage</th>"
             "<th>Pitch+</th><th>Stuff+</th><th>Shape+</th><th>Location+</th><th>Results+</th><th>Staff rank</th></tr>")
    for p in r["pitch_by_pitch"]:
        g = p["grades"]
        H.append(f"<tr><td><b>{_e(p['pitch'])}</b></td><td>{_e(p['subtype'])}</td><td>{p['n']}</td><td>{p['game_pitches']}</td>"
                 f"<td>{_e(p['usage_pct'])}%</td>{grade_cell(g['pitch_plus'])}{grade_cell(g['stuff_plus'])}"
                 f"{grade_cell(g['shape_plus'])}{grade_cell(g['location_plus'])}{grade_cell(g['results_plus'])}"
                 f"<td>{_e(g.get('staff_rank'))}</td></tr>")
    H.append(f"</table><p class='note'>{_e(scale_note)} Pitch types need {10} pitches to be graded.</p>")

    # Fastball
    t = r.get("fastball_traits")
    if t or r.get("fastball_role"):
        H.append("<h2>Fastball</h2>")
        if r.get("fastball_role"):
            H.append(f"<p style='margin:2px 0 4px'>{_e(r['fastball_role']['role'])}</p>")
        if t:
            pct = (f" · {ordinal(t['velo_pct_d3'])} → {ordinal(t['perceived_pct_d3'])} percentile among D3 pitchers"
                   if t.get("velo_pct_d3") is not None else "")
            H.append("<table><tr><th>Velocity</th><th>Extension</th><th>Perceived</th><th>Approach angle</th>"
                     "<th>VAA vs expected</th><th>Plate height</th><th>Release height</th></tr>"
                     f"<tr><td>{t['velo']} mph</td><td>{t['extension_ft']} ft ({ordinal(t['extension_pct_mlb'])} pct MLB)</td>"
                     f"<td>{t['perceived_velo']} mph ({t['extension_bonus_mph']:+.1f})</td><td>{t['vaa']}°</td>"
                     f"<td>{t['vaa_above_expected']:+.2f}° ({ordinal(t['vaa_ae_pct_mlb'])} pct MLB)</td>"
                     f"<td>{t['avg_plate_height_ft']} ft</td><td>{t['release_height_ft']} ft</td></tr></table>")
            H.append(f"<p class='note'>Velocity{pct}. VAA vs expected compares his approach angle with an average fastball at "
                     "the same velocity and plate height; positive = flatter, which plays up in the zone.</p>")

    # Pitch by pitch
    H.append("<h2>Pitch by pitch</h2>")
    for p in r["pitch_by_pitch"]:
        g = p["grades"]
        sub = f" ({_e(p['subtype'])})" if p.get("subtype") else ""
        H.append(f"<div class='pitch'><h3>{_e(p['pitch'])}{sub} <span class='verdict'>— {_e(p.get('verdict', ''))}</span></h3>")
        H.append(f"<div class='line'>Pitch+ {_e(g['pitch_plus'])} · Stuff+ {_e(g['stuff_plus'])} · Location+ {_e(g['location_plus'])}"
                 f" · Results+ {_e(g['results_plus'])} <span class='tag'>staff rank {_e(g.get('staff_rank'))}</span></div><ul>")
        if p.get("stuff_explained"):
            H.append(f"<li>Why the Stuff grade: {_e(p['stuff_explained'])}</li>")
        if p.get("comparable_mlb"):
            H.append(f"<li>{_e(p['comparable_mlb']['text'])}</li>")
        for j in p.get("jobs", []):
            H.append(f"<li><b class='good'>Job — {_e(j['job'])}:</b> {_e(j['evidence'])}</li>")
        for b in p.get("deal_breakers", []):
            H.append(f"<li><b class='bad'>Where it's failing — {_e(b['issue'])}:</b> {_e(b['evidence'])}</li>")
        for x in p.get("not_strength", []):
            H.append(f"<li>Not its strength: {_e(x)}</li>")
        if p.get("add_on_idea"):
            H.append(f"<li>{_e(p['add_on_idea'])}</li>")
        H.append(f"</ul><div class='so'><b>So:</b> {_e(p.get('so', ''))}</div></div>")

    # Situations
    s = r["situations"]
    H.append("<h2>Situations</h2>")
    bits = []
    if s.get("first_pitch_strike"):
        f = s["first_pitch_strike"]
        bits.append(f"First-pitch strikes: {_e(f['pct'])}% of {f['n']} (MLB middle half {f['pro_range']})")
    fade = s.get("fade", {})
    if fade.get("strike_late") is not None:
        txt = f"Pitches 1–25: {fade['strike_early']:.0f}% strikes; after: {fade['strike_late']:.0f}%"
        if fade.get("fb_velo_late") is not None:
            txt += f" (fastball {fade['fb_velo_early']} → {fade['fb_velo_late']} mph)"
        bits.append(txt)
    H.append(_list(bits))
    if s.get("by_count"):
        has_hard = any(row.get("hard_pct") is not None for row in s["by_count"])
        H.append("<table><tr><th>Count</th><th>n</th><th>Strike%</th><th>Whiff%</th>" +
                 ("<th>Hard%</th>" if has_hard else "") + "<th>Most used</th></tr>")
        for row in s["by_count"]:
            H.append(f"<tr><td>{_e(row['count'])}</td><td>{row['n']}</td><td>{_e(row['strike_pct'])}</td>"
                     f"<td>{_e(row['whiff_pct'])}</td>" + (f"<td>{_e(row['hard_pct'])}</td>" if has_hard else "") +
                     f"<td>{_e(row['most_used'])}</td></tr>")
        H.append("</table>")
    if s.get("splits"):
        H.append("<table><tr><th>Split</th><th>PA</th><th>AVG</th><th>OBP</th><th>SLG</th><th>K%</th><th>BB%</th><th>Mix</th></tr>")
        fmt = lambda v: "–" if v is None else f"{v:.3f}".lstrip("0")
        for label, sp in s["splits"].items():
            mix = ", ".join(f"{k} {v}%" for k, v in sp.get("mix", {}).items())
            H.append(f"<tr><td>{label}</td><td>{sp['pa']}</td><td>{fmt(sp.get('avg'))}</td><td>{fmt(sp.get('obp'))}</td>"
                     f"<td>{fmt(sp.get('slg'))}</td><td>{_e(sp.get('k_pct'))}</td><td>{_e(sp.get('bb_pct'))}</td><td>{_e(mix)}</td></tr>")
        H.append("</table>")

    # Movement + progress
    H.append("<h2>Movement profile</h2><table><tr><th>Pitch</th><th>Velo (max)</th><th>IVB</th><th>HB (arm +)</th><th>Spin</th>"
             "<th>Tilt</th><th>VAA</th><th>Rel H</th><th>Rel S</th><th>Ext</th></tr>")
    for m in r["movement_profile"]:
        H.append(f"<tr><td><b>{_e(m['pitch'])}</b></td><td>{_e(m['velo'])} ({_e(m['max_velo'])})</td><td>{_e(m['ivb'])}</td>"
                 f"<td>{_e(m['hb_arm'])}</td><td>{_e(m['spin'])}</td><td>{_e(m['tilt'])}</td><td>{_e(m['vaa'])}</td>"
                 f"<td>{_e(m['rel_height'])}</td><td>{_e(m['rel_side_arm'])}</td><td>{_e(m['extension'])}</td></tr>")
    H.append("</table>")
    if r.get("progress"):
        H.append("<h2>Progress by outing</h2><table><tr><th>Outing</th><th>Pitch</th><th>n</th><th>Velo</th><th>IVB</th>"
                 "<th>HB</th><th>Spin</th><th>Stuff+</th></tr>")
        for x in r["progress"]:
            H.append(f"<tr><td>{_e(x['outing'])}</td><td>{_e(x['pitch'])}</td><td>{x['n']}</td><td>{_e(x['velo'])}</td>"
                     f"<td>{_e(x['ivb'])}</td><td>{_e(x['hb'])}</td><td>{_e(x['spin'])}</td>{grade_cell(x['stuff_plus'])}</tr>")
        H.append("</table>")
    H.append("</body></html>")
    return "".join(H)


def write_pdf(r: dict, path: Path, scale_note: str = "") -> Path:
    from playwright.sync_api import sync_playwright

    path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.set_content(to_html(r, scale_note), wait_until="load")
        foot = ('<div style="font-size:7px;color:#888;width:100%;padding:0 0.5in;display:flex;justify-content:space-between;'
                'font-family:Helvetica,Arial,sans-serif"><span>PRISM · Pitch Rating Integrated Stuff Model · Oberlin Baseball</span>'
                '<span>Page <span class="pageNumber"></span> of <span class="totalPages"></span></span></div>')
        page.pdf(path=str(path), format="Letter", print_background=True, display_header_footer=True,
                 header_template="<div></div>", footer_template=foot,
                 margin={"top": "0.45in", "bottom": "0.55in", "left": "0.5in", "right": "0.5in"})
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
