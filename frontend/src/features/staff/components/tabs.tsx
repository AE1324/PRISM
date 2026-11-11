import type { Pitch, PitcherReport } from "../types";
import { num, ordinal, pct, pitchColor, signed } from "../format";
import { GradeCell, MovementChart, Section, Status, Swatch, TargetTable, TrendChart } from "./ui";

const List = ({ items, ordered, empty }: { items: string[]; ordered?: boolean; empty: string }) =>
  !items.length ? <p className="soft" style={{ margin: 0 }}>{empty}</p> : ordered ? <ol>{items.map((x) => <li key={x}>{x}</li>)}</ol> : <ul>{items.map((x) => <li key={x}>{x}</li>)}</ul>;

/* ---------------- Overview ---------------- */

export function OverviewTab({ r }: { r: PitcherReport }) {
  const d = r.development, fb = r.fastball_traits;
  return (
    <>
      <Section idx="01" title="Development">
        <div className="pr-grid pr-cols3">
          <div><div className="pr-colhead">Strengths</div><List items={d.what_makes_him} empty="No clear strength yet." /></div>
          <div><div className="pr-colhead">Limiting factors</div><List items={d.holding_back} empty="Nothing clearly holding him back." /></div>
          <div><div className="pr-colhead">Priorities</div><List items={d.priority_order} ordered empty="Keep doing what he's doing." /></div>
        </div>
      </Section>

      <Section idx="02" title="Arsenal grades" aside="100 = average · red above, blue below">
        <div className="pr-scroll">
          <table>
            <thead>
              <tr><th>Pitch</th><th className="num">Usage</th><th className="c">Pitch+</th><th className="c">Stuff</th><th className="c">Shape</th><th className="c">Location</th><th className="c">Results</th><th className="num">Velo</th><th className="num">IVB</th><th className="num">HB</th><th className="num">Spin</th></tr>
            </thead>
            <tbody>
              {r.pitch_by_pitch.map((p) => (
                <tr key={p.pitch}>
                  <td style={{ whiteSpace: "nowrap" }}><Swatch pitch={p.pitch} /><b>{p.pitch}</b> <span className="soft">{p.subtype}</span></td>
                  <td className="num">{p.usage_pct}%</td>
                  <GradeCell v={p.grades.pitch_plus} /><GradeCell v={p.grades.stuff_plus} /><GradeCell v={p.grades.shape_plus} />
                  <GradeCell v={p.grades.location_plus} /><GradeCell v={p.grades.results_plus} />
                  <td className="num">{num(p.shape?.velo)}</td><td className="num">{num(p.shape?.ivb)}</td><td className="num">{num(p.shape?.hb_arm)}</td><td className="num">{p.shape?.spin ?? "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="pr-note" style={{ margin: "10px 0 0" }}>Results and location count more as the sample grows. Pitch types need 10 pitches to be graded.</p>
      </Section>

      <Section idx="03" title="Shapes and fastball">
        <div className="pr-grid pr-split">
          <div className="pr-frame"><MovementChart report={r} /></div>
          <div>
            {r.fastball_role && <p className="pr-callout" style={{ margin: "0 0 16px", ["--pc" as string]: pitchColor(r.fastball_role.pitch) }}>{r.fastball_role.role}</p>}
            {fb && (
              <table>
                <thead><tr><th>Fastball</th><th className="num">Value</th><th className="num">Percentile</th></tr></thead>
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
            <div style={{ marginTop: 18 }}>
              <div className="pr-block"><span className="pr-label">How his pitches work together</span><span className="body">{d.pitches_together.join("; ")}.</span></div>
              {r.arsenal_plan && <div className="pr-block"><span className="pr-label">Arsenal plan</span><span className="body">{r.arsenal_plan}</span></div>}
            </div>
          </div>
        </div>
      </Section>
    </>
  );
}

/* ---------------- Pitches ---------------- */

const RATES: [keyof NonNullable<Pitch["rates"]>, string][] = [
  ["strike", "Strike"], ["zone", "Zone"], ["called_strike", "Called strike"], ["whiff", "Whiff"], ["chase", "Chase"], ["putaway", "Putaway"],
];

export function PitchTab({ r, p }: { r: PitcherReport; p: Pitch }) {
  const sh = p.shape, sa = p.spin_axis, g = p.grades;
  const graded = typeof g.pitch_plus === "number";
  const pc = { ["--pc" as string]: pitchColor(p.pitch) };
  return (
    <Section idx="01" title={`${p.pitch}${p.subtype ? ` · ${p.subtype}` : ""}`} aside={<Status verdict={p.verdict} />}>
      <div className="pr-grid pr-detail">
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 18, flexWrap: "wrap", alignItems: "flex-end" }}>
            {sh && <div className="mono soft" style={{ fontSize: 12 }}>{num(sh.velo)} mph · {num(sh.ivb)}" IVB · {num(sh.hb_arm)}" HB · {sh.spin} rpm · {p.n} thrown · {p.game_pitches} vs hitters</div>}
            <table style={{ width: "auto" }}>
              <thead><tr><th className="c">Pitch+</th><th className="c">Stuff</th><th className="c">Loc</th><th className="c">Res</th></tr></thead>
              <tbody><tr><GradeCell v={g.pitch_plus} /><GradeCell v={g.stuff_plus} /><GradeCell v={g.location_plus} /><GradeCell v={g.results_plus} /></tr></tbody>
            </table>
          </div>

          {!graded ? (
            <p className="body" style={{ marginTop: 16 }}>{p.so}</p>
          ) : (
            <>
              <div className="pr-callout" style={{ margin: "18px 0 20px", ...pc }}>
                <div style={{ fontWeight: 600 }}>{p.so}</div>
                <div className="body" style={{ fontSize: 13, marginTop: 3 }}>
                  {p.jobs?.length ? <>{p.jobs[0].job}: {p.jobs[0].evidence}.</> : "No clear job yet."}
                  {!!p.great?.length && <> Also: {p.great.join("; ")}.</>}
                </div>
                {!!p.deal_breakers?.length && <div className="out" style={{ fontSize: 13, marginTop: 3 }}>{p.deal_breakers.map((x) => `${x.issue}: ${x.evidence}`).join(" · ")}</div>}
              </div>

              <div className="pr-colhead">Targets</div>
              <TargetTable pitch={p} />

              <div style={{ marginTop: 18 }}>
                {p.targets && <div className="pr-block"><span className="pr-label">Plan</span><span className="body">{p.targets.plan}</span></div>}
                <div className="pr-block">
                  <span className="pr-label">Spin axis</span>
                  <span className="body">
                    {sa?.normal ? (
                      <>
                        Normal <span className="mono" style={{ color: "var(--pr-ink)" }}>{sa.normal}</span>, middle half within {sa.spread_deg}°.{" "}
                        {sa.call && sa.good_word ? `${sa.good_word.charAt(0).toUpperCase()}${sa.good_word.slice(1)} plays better (${sa.why}).` : "No direction clearly plays better for this pitch type."}
                        {sa.good && <> His good version: <span className="mono">{sa.good.clock}</span> ({sa.good.share}% of them).</>}
                        {sa.bad && <> Watch for <span className="mono">{sa.bad.clock}</span> ({sa.bad.share}%).</>}
                      </>
                    ) : "Not enough pitches to read an axis yet."}
                  </span>
                </div>
                <div className="pr-block"><span className="pr-label">Whiff vs contact</span><span className="body">{p.swing_split || "Not enough swings yet."}</span></div>
                {(p.stuff_parts || p.comparable_mlb?.text) && (
                  <div className="pr-block">
                    <span className="pr-label">Why the Stuff grade</span>
                    <span className="body">
                      {p.stuff_parts && <>vs a typical {p.pitch.toLowerCase()}: {Object.entries(p.stuff_parts).map(([k, v]) => `${k} ${v > 0 ? "+" : ""}${v}`).join(", ")}. </>}
                      {p.comparable_mlb?.text}
                    </span>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        <div>
          <div className="pr-colhead">Movement</div>
          <div className="pr-frame"><MovementChart report={r} highlight={p.pitch} size={400} /></div>
          {p.rates && (
            <div style={{ marginTop: 22 }}>
              <div className="pr-colhead">Rates</div>
              <table>
                <thead><tr><th>Rate</th><th className="num">His</th><th className="num">Reference</th></tr></thead>
                <tbody>
                  {RATES.map(([k, label]) => {
                    const v = p.rates?.[k], ref = p.rates_ref?.[k];
                    const cls = typeof v === "number" && typeof ref === "number" ? (v >= ref + 5 ? "in" : v <= ref - 5 ? "out" : "") : "";
                    return <tr key={k}><td>{label}</td><td className={`num ${cls}`}>{pct(v ?? null)}</td><td className="num soft">{pct(ref ?? null)}</td></tr>;
                  })}
                </tbody>
              </table>
              <p className="pr-note" style={{ margin: "8px 0 0" }}>Strike and whiff compare to D3 pitchers; the rest to his own staff.</p>
            </div>
          )}
        </div>
      </div>
    </Section>
  );
}

/* ---------------- Counts ---------------- */

export function CountsTab({ r }: { r: PitcherReport }) {
  const s = r.situations, fps = s.first_pitch_strike, fade = s.fade;
  const splits = Object.entries(s.splits ?? {});
  return (
    <>
      <Section idx="01" title="By count" aside={fps ? <>First-pitch strikes <span className="mono" style={{ color: "var(--pr-ink)" }}>{pct(fps.pct, 1)}</span> of {fps.n} · MLB middle half {fps.pro_range}</> : null}>
        <div className="pr-scroll">
          <table>
            <thead><tr><th>Count</th><th className="num">N</th><th className="num">Strike %</th><th className="num">Whiff %</th><th>Most used</th></tr></thead>
            <tbody>{s.by_count.map((c) => (
              <tr key={c.count}><td>{c.count}</td><td className="num">{c.n}</td><td className="num">{num(c.strike_pct)}</td><td className="num">{num(c.whiff_pct)}</td><td className="body">{c.most_used}</td></tr>
            ))}</tbody>
          </table>
        </div>
      </Section>
      <div className="pr-grid pr-split">
        <Section idx="02" title="Platoon splits">
          {splits.length ? (
            <div className="pr-scroll">
              <table>
                <thead><tr><th>Split</th><th className="num">PA</th><th className="num">AVG</th><th className="num">OBP</th><th className="num">SLG</th><th className="num">K%</th><th className="num">BB%</th></tr></thead>
                <tbody>{splits.map(([k, v]) => (
                  <tr key={k}>
                    <td><b>{k}</b><div className="soft" style={{ fontSize: 12 }}>{Object.entries(v.mix).map(([p, n]) => `${p} ${n}%`).join(", ")}</div></td>
                    <td className="num">{v.pa}</td><td className="num">{num(v.avg, 3)}</td><td className="num">{num(v.obp, 3)}</td><td className="num">{num(v.slg, 3)}</td>
                    <td className="num">{num(v.k_pct)}</td><td className="num">{num(v.bb_pct)}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ) : <p className="soft" style={{ margin: 0 }}>No batter-side data yet.</p>}
        </Section>
        <Section idx="03" title="Early vs late in an outing">
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
        </Section>
      </div>
    </>
  );
}

/* ---------------- Progress ---------------- */

export function ProgressTab({ r }: { r: PitcherReport }) {
  const outings = [...new Set(r.progress.map((x) => x.outing))].sort();
  const pitches = r.pitch_by_pitch.map((p) => p.pitch).filter((p) => r.progress.some((x) => x.pitch === p));
  const series = (key: "velo" | "ivb" | "hb" | "spin" | "stuff_plus") =>
    pitches.map((p) => ({ pitch: p, values: outings.map((o) => r.progress.find((x) => x.outing === o && x.pitch === p)?.[key] ?? null) }));
  if (!outings.length) return <Section idx="01" title="Trends"><p className="soft" style={{ margin: 0 }}>Not enough pitches per outing to chart progress yet.</p></Section>;
  return (
    <>
      <Section idx="01" title="Trends" aside={<span style={{ display: "inline-flex", gap: 14, flexWrap: "wrap" }}>{pitches.map((p) => <span key={p} style={{ display: "inline-flex", alignItems: "center" }}><Swatch pitch={p} />{p}</span>)}</span>}>
        <div className="pr-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 22 }}>
          <TrendChart title="Stuff+" series={series("stuff_plus")} outings={outings} />
          <TrendChart title="Velocity" unit="mph" series={series("velo")} outings={outings} />
          <TrendChart title="Induced vertical break" unit="in" series={series("ivb")} outings={outings} />
          <TrendChart title="Horizontal break" unit="in" series={series("hb")} outings={outings} />
        </div>
      </Section>
      <Section idx="02" title="By outing" aside="Stuff+ shown with 5+ of a pitch type in the outing">
        <div className="pr-scroll">
          <table>
            <thead><tr><th>Outing</th><th>Pitch</th><th className="num">N</th><th className="num">Velo</th><th className="num">IVB</th><th className="num">HB</th><th className="num">Spin</th><th className="c">Stuff+</th></tr></thead>
            <tbody>
              {[...r.progress].sort((a, b) => a.outing.localeCompare(b.outing)).map((x, i, arr) => (
                <tr key={i} style={i > 0 && arr[i - 1].outing !== x.outing ? { borderTop: "1px solid var(--pr-line2)" } : undefined}>
                  <td className="mono">{i === 0 || arr[i - 1].outing !== x.outing ? x.outing : ""}</td>
                  <td style={{ whiteSpace: "nowrap" }}><Swatch pitch={x.pitch} />{x.pitch}</td><td className="num">{x.n}</td>
                  <td className="num">{num(x.velo)}</td><td className="num">{num(x.ivb)}</td><td className="num">{num(x.hb)}</td><td className="num">{x.spin ?? "–"}</td>
                  <GradeCell v={x.stuff_plus} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </>
  );
}
