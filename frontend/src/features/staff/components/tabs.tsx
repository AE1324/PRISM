import type { Pitch, PitcherReport } from "../types";
import { num, ordinal, pct, signed } from "../format";
import { Dot, GradeCell, MovementChart, Status, TargetTable, TrendChart } from "./ui";

/* ---------------- Overview ---------------- */

export function OverviewTab({ r }: { r: PitcherReport }) {
  const d = r.development, fb = r.fastball_traits;
  return (
    <div className="pr-grid">
      <div className="pr-grid pr-cols3">
        <div className="pr-panel"><h2>Strengths</h2>{d.what_makes_him.length ? <ul>{d.what_makes_him.map((x) => <li key={x}>{x}</li>)}</ul> : <p className="soft" style={{ margin: 0 }}>No clear strength yet.</p>}</div>
        <div className="pr-panel"><h2>Limiting factors</h2>{d.holding_back.length ? <ul>{d.holding_back.map((x) => <li key={x}>{x}</li>)}</ul> : <p className="soft" style={{ margin: 0 }}>Nothing clearly holding him back.</p>}</div>
        <div className="pr-panel"><h2>Priorities</h2>{d.priority_order.length ? <ol>{d.priority_order.map((x) => <li key={x}>{x}</li>)}</ol> : <p className="soft" style={{ margin: 0 }}>Keep doing what he's doing.</p>}</div>
      </div>

      <div className="pr-panel">
        <h2>Arsenal grades</h2>
        <div className="pr-scroll">
          <table>
            <thead>
              <tr><th>Pitch</th><th className="num">Usage</th><th className="c">Pitch+</th><th className="c">Stuff</th><th className="c">Shape</th><th className="c">Location</th><th className="c">Results</th><th className="num">Velo</th><th className="num">IVB</th><th className="num">HB</th><th className="num">Spin</th></tr>
            </thead>
            <tbody>
              {r.pitch_by_pitch.map((p) => (
                <tr key={p.pitch}>
                  <td style={{ whiteSpace: "nowrap" }}><Dot pitch={p.pitch} />{p.pitch} <span className="soft">{p.subtype}</span></td>
                  <td className="num">{p.usage_pct}%</td>
                  <GradeCell v={p.grades.pitch_plus} /><GradeCell v={p.grades.stuff_plus} /><GradeCell v={p.grades.shape_plus} />
                  <GradeCell v={p.grades.location_plus} /><GradeCell v={p.grades.results_plus} />
                  <td className="num">{num(p.shape?.velo)}</td><td className="num">{num(p.shape?.ivb)}</td><td className="num">{num(p.shape?.hb_arm)}</td><td className="num">{p.shape?.spin ?? "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="pr-note" style={{ margin: "8px 0 0" }}>Red is above average, blue below; 100 is average. Results and location count more as the sample grows. Pitch types need 10 pitches to be graded.</p>
      </div>

      <div className="pr-grid pr-split">
        <div className="pr-panel"><h2>Movement</h2><MovementChart report={r} /></div>
        <div className="pr-panel">
          <h2>Fastball</h2>
          {r.fastball_role && <p className="body" style={{ margin: "0 0 12px" }}>{r.fastball_role.role}</p>}
          {fb && (
            <table>
              <thead><tr><th>Trait</th><th className="num">Value</th><th className="num">Percentile</th></tr></thead>
              <tbody>
                <tr><td>Velocity</td><td className="num">{num(fb.velo)} mph</td><td className="num">{ordinal(fb.velo_pct_d3)} D3</td></tr>
                <tr><td>Perceived velocity</td><td className="num">{num(fb.perceived_velo)} mph</td><td className="num">{ordinal(fb.perceived_pct_d3)} D3</td></tr>
                <tr><td>Extension</td><td className={`num ${(fb.extension_pct_mlb ?? 50) <= 15 ? "out" : (fb.extension_pct_mlb ?? 50) >= 85 ? "in" : ""}`}>{num(fb.extension_ft, 2)} ft</td><td className="num">{ordinal(fb.extension_pct_mlb)} MLB</td></tr>
                {fb.show_vaa && typeof fb.vaa_above_expected === "number" && (
                  <tr><td>Approach angle vs expected</td><td className="num">{signed(fb.vaa_above_expected, 2)}°</td><td className="num">{ordinal(fb.vaa_ae_pct_mlb)} MLB</td></tr>
                )}
              </tbody>
            </table>
          )}
          <div className="pr-sub" style={{ marginTop: 12 }}>
            <span className="pr-label">How his pitches work together</span>
            <div className="body" style={{ marginTop: 3 }}>{d.pitches_together.join("; ")}.</div>
          </div>
          {r.arsenal_plan && (
            <div className="pr-sub" style={{ marginTop: 10 }}>
              <span className="pr-label">Arsenal plan</span>
              <div className="body" style={{ marginTop: 3 }}>{r.arsenal_plan}</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------------- Pitches ---------------- */

const RATES: [keyof NonNullable<Pitch["rates"]>, string][] = [
  ["strike", "Strike"], ["zone", "Zone"], ["called_strike", "Called strike"], ["whiff", "Whiff"], ["chase", "Chase"], ["putaway", "Putaway"],
];

export function PitchTab({ r, p }: { r: PitcherReport; p: Pitch }) {
  const sh = p.shape, sa = p.spin_axis, g = p.grades;
  const graded = typeof g.pitch_plus === "number";
  return (
    <div className="pr-grid pr-detail">
      <div className="pr-panel">
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16, flexWrap: "wrap" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <Dot pitch={p.pitch} />
              <h2 style={{ margin: 0, fontSize: 17 }}>{p.pitch}</h2>
              <span className="soft">{p.subtype}</span>
              <Status verdict={p.verdict} />
            </div>
            {sh && <div className="soft mono" style={{ fontSize: 12, marginTop: 6 }}>{num(sh.velo)} mph · {num(sh.ivb)}" IVB · {num(sh.hb_arm)}" HB · {sh.spin} rpm · {p.n} thrown · {p.game_pitches} vs hitters</div>}
          </div>
          <table style={{ width: "auto" }}>
            <thead><tr><th className="c">Pitch+</th><th className="c">Stuff</th><th className="c">Loc</th><th className="c">Res</th></tr></thead>
            <tbody><tr><GradeCell v={g.pitch_plus} /><GradeCell v={g.stuff_plus} /><GradeCell v={g.location_plus} /><GradeCell v={g.results_plus} /></tr></tbody>
          </table>
        </div>

        {!graded ? (
          <p className="body" style={{ marginTop: 14 }}>{p.so}</p>
        ) : (
          <>
            <div style={{ margin: "14px 0 4px" }}>
              <span className="pr-label">Role</span>
              <div style={{ marginTop: 2 }}>
                {p.jobs?.length ? <>{p.jobs[0].job} <span className="soft">— {p.jobs[0].evidence}</span></> : <span className="soft">No clear job yet</span>}
              </div>
              {!!p.great?.length && <div className="body" style={{ marginTop: 2 }}>{p.great.join(" · ")}</div>}
              {!!p.deal_breakers?.length && (
                <div className="out" style={{ marginTop: 2 }}>{p.deal_breakers.map((x) => `${x.issue}: ${x.evidence}`).join(" · ")}</div>
              )}
            </div>
            <div style={{ marginTop: 14 }}><div className="pr-label" style={{ marginBottom: 4 }}>Targets</div><TargetTable pitch={p} /></div>
            {p.targets && (
              <div className="pr-sub" style={{ marginTop: 12 }}>
                <span className="pr-label">Plan</span>
                <div className="body" style={{ marginTop: 3 }}>{p.targets.plan}</div>
              </div>
            )}
            <div className="pr-grid pr-cols2" style={{ marginTop: 12 }}>
              <div className="pr-sub">
                <span className="pr-label">Spin axis</span>
                <div className="body" style={{ marginTop: 3 }}>
                  {sa?.normal ? (
                    <>
                      Normal <span className="mono" style={{ color: "var(--pr-ink)" }}>{sa.normal}</span>, middle half within {sa.spread_deg}°.{" "}
                      {sa.call && sa.good_word ? `${sa.good_word.charAt(0).toUpperCase()}${sa.good_word.slice(1)} plays better (${sa.why}).` : "No direction clearly plays better for this pitch type."}
                      {sa.good && <> His good version: <span className="mono">{sa.good.clock}</span> ({sa.good.share}% of them).</>}
                      {sa.bad && <> Watch for <span className="mono">{sa.bad.clock}</span> ({sa.bad.share}%).</>}
                    </>
                  ) : "Not enough pitches to read an axis yet."}
                </div>
              </div>
              <div className="pr-sub">
                <span className="pr-label">Whiff vs contact</span>
                <div className="body" style={{ marginTop: 3 }}>{p.swing_split || "Not enough swings yet."}</div>
              </div>
            </div>
            {(p.stuff_parts || p.comparable_mlb?.text) && (
              <div className="pr-sub" style={{ marginTop: 12 }}>
                <span className="pr-label">Why the Stuff grade</span>
                <div className="body" style={{ marginTop: 3 }}>
                  {p.stuff_parts && <>vs a typical {p.pitch.toLowerCase()}: {Object.entries(p.stuff_parts).map(([k, v]) => `${k} ${v > 0 ? "+" : ""}${v}`).join(", ")}. </>}
                  {p.comparable_mlb?.text}
                </div>
              </div>
            )}
            <div style={{ marginTop: 12, borderTop: "1px solid var(--pr-line)", paddingTop: 10 }}>
              <span className="pr-label">Bottom line</span>
              <div style={{ marginTop: 2 }}>{p.so}</div>
            </div>
          </>
        )}
      </div>

      <div className="pr-grid">
        <div className="pr-panel"><h2>Movement</h2><MovementChart report={r} highlight={p.pitch} width={340} height={300} /></div>
        {p.rates && (
          <div className="pr-panel">
            <h2>Rates</h2>
            <table>
              <thead><tr><th>Rate</th><th className="num">His</th><th className="num">Reference</th></tr></thead>
              <tbody>
                {RATES.map(([k, label]) => (
                  <tr key={k}><td>{label}</td><td className="num">{pct(p.rates?.[k] ?? null)}</td><td className="num soft">{pct(p.rates_ref?.[k] ?? null)}</td></tr>
                ))}
              </tbody>
            </table>
            <p className="pr-note" style={{ margin: "8px 0 0" }}>Strike and whiff compare to D3 pitchers; the rest to the Oberlin staff.</p>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------------- Counts ---------------- */

export function CountsTab({ r }: { r: PitcherReport }) {
  const s = r.situations, fps = s.first_pitch_strike, fade = s.fade;
  const splits = Object.entries(s.splits ?? {});
  return (
    <div className="pr-grid">
      <div className="pr-panel">
        <h2>By count</h2>
        {fps && <p className="body" style={{ margin: "0 0 10px" }}>First-pitch strikes: <span className="mono">{pct(fps.pct, 1)}</span> of {fps.n} <span className="soft">(MLB middle half {fps.pro_range})</span></p>}
        <div className="pr-scroll">
          <table>
            <thead><tr><th>Count</th><th className="num">N</th><th className="num">Strike %</th><th className="num">Whiff %</th><th>Most used</th></tr></thead>
            <tbody>{s.by_count.map((c) => (
              <tr key={c.count}><td>{c.count}</td><td className="num">{c.n}</td><td className="num">{num(c.strike_pct)}</td><td className="num">{num(c.whiff_pct)}</td><td className="body">{c.most_used}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </div>
      <div className="pr-grid pr-split">
        <div className="pr-panel">
          <h2>Platoon splits</h2>
          {splits.length ? (
            <div className="pr-scroll">
              <table>
                <thead><tr><th>Split</th><th className="num">PA</th><th className="num">AVG</th><th className="num">OBP</th><th className="num">SLG</th><th className="num">K%</th><th className="num">BB%</th><th>Mix</th></tr></thead>
                <tbody>{splits.map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td><td className="num">{v.pa}</td><td className="num">{num(v.avg, 3)}</td><td className="num">{num(v.obp, 3)}</td><td className="num">{num(v.slg, 3)}</td>
                    <td className="num">{num(v.k_pct)}</td><td className="num">{num(v.bb_pct)}</td>
                    <td className="body" style={{ whiteSpace: "nowrap" }}>{Object.entries(v.mix).map(([p, n]) => `${p} ${n}%`).join(", ")}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ) : <p className="soft" style={{ margin: 0 }}>No batter-side data yet.</p>}
        </div>
        <div className="pr-panel">
          <h2>Early vs late in an outing</h2>
          {fade ? (
            <table>
              <thead><tr><th></th><th className="num">Pitches 1–25</th><th className="num">After 25</th></tr></thead>
              <tbody>
                <tr><td>Pitches</td><td className="num">{fade.early_n}</td><td className="num">{fade.late_n}</td></tr>
                <tr><td>Strike %</td><td className="num">{num(fade.strike_early)}</td><td className="num">{num(fade.strike_late)}</td></tr>
                <tr><td>Fastball velo</td><td className="num">{num(fade.fb_velo_early)}</td><td className="num">{num(fade.fb_velo_late)}</td></tr>
              </tbody>
            </table>
          ) : <p className="soft" style={{ margin: 0 }}>Not enough long outings yet.</p>}
        </div>
      </div>
    </div>
  );
}

/* ---------------- Progress ---------------- */

export function ProgressTab({ r }: { r: PitcherReport }) {
  const outings = [...new Set(r.progress.map((x) => x.outing))].sort();
  const pitches = r.pitch_by_pitch.map((p) => p.pitch).filter((p) => r.progress.some((x) => x.pitch === p));
  const series = (key: "velo" | "ivb" | "hb" | "spin" | "stuff_plus") =>
    pitches.map((p) => ({ pitch: p, values: outings.map((o) => r.progress.find((x) => x.outing === o && x.pitch === p)?.[key] ?? null) }));
  if (!outings.length) return <div className="pr-panel"><p className="soft" style={{ margin: 0 }}>Not enough pitches per outing to chart progress yet.</p></div>;
  return (
    <div className="pr-grid">
      <div className="pr-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
        <TrendChart title="Stuff+" series={series("stuff_plus")} outings={outings} />
        <TrendChart title="Velocity" unit="mph" series={series("velo")} outings={outings} />
        <TrendChart title="Induced vertical break" unit="in" series={series("ivb")} outings={outings} />
        <TrendChart title="Horizontal break" unit="in" series={series("hb")} outings={outings} />
      </div>
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }} className="body">
        {pitches.map((p) => <span key={p} style={{ display: "inline-flex", alignItems: "center" }}><Dot pitch={p} />{p}</span>)}
      </div>
      <div className="pr-panel">
        <h2>By outing</h2>
        <div className="pr-scroll">
          <table>
            <thead><tr><th>Outing</th><th>Pitch</th><th className="num">N</th><th className="num">Velo</th><th className="num">IVB</th><th className="num">HB</th><th className="num">Spin</th><th className="c">Stuff+</th></tr></thead>
            <tbody>
              {[...r.progress].sort((a, b) => a.outing.localeCompare(b.outing)).map((x, i) => (
                <tr key={i}>
                  <td className="mono">{x.outing}</td><td style={{ whiteSpace: "nowrap" }}><Dot pitch={x.pitch} />{x.pitch}</td><td className="num">{x.n}</td>
                  <td className="num">{num(x.velo)}</td><td className="num">{num(x.ivb)}</td><td className="num">{num(x.hb)}</td><td className="num">{x.spin ?? "–"}</td>
                  <GradeCell v={x.stuff_plus} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="pr-note" style={{ margin: "8px 0 0" }}>Stuff+ is shown when he threw at least 5 of a pitch type in that outing.</p>
      </div>
    </div>
  );
}
