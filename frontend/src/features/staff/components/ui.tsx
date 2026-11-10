import type { Grade, Pitch, PitcherReport } from "../types";
import { fmtGrade, pitchColor, savant, signed, verdictTone, sentence } from "../format";

export function GradeCell({ v }: { v: Grade | undefined }) {
  const [bg, fg] = savant(v ?? null);
  return (
    <td className="g" style={{ background: bg, color: fg }}>
      {fmtGrade(v)}
    </td>
  );
}

export function Score({ v }: { v: Grade | undefined }) {
  const [bg, fg] = savant(v ?? null);
  return (
    <span className="pr-score" style={{ background: bg, color: fg }}>
      {fmtGrade(v)}
    </span>
  );
}

export function Status({ verdict }: { verdict: string }) {
  return (
    <span className="pr-status">
      <i style={{ background: verdictTone(verdict) }} />
      {sentence(verdict)}
    </span>
  );
}

export function Dot({ pitch }: { pitch: string }) {
  return <span className="pr-dot" style={{ background: pitchColor(pitch) }} aria-hidden="true" />;
}

/** Movement chart: every pitch as a faint dot, each pitch type's average as a labeled marker. */
export function MovementChart({ report, highlight, width = 420, height = 340 }: { report: PitcherReport; highlight?: string; width?: number; height?: number }) {
  const pad = 36;
  const sx = (v: number) => pad + ((v + 25) / 50) * (width - 2 * pad);
  const sy = (v: number) => height - pad - ((v + 25) / 50) * (height - 2 * pad);
  const ticks = [-20, -10, 0, 10, 20];
  const label = `Movement chart for ${report.pitcher}: ${report.movement_profile.map((m) => `${m.pitch} ${m.ivb} in vertical, ${m.hb_arm} in horizontal`).join("; ")}`;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} width="100%" className="pr-chart" role="img" aria-label={label}>
      {ticks.map((v) => (
        <g key={v}>
          <line x1={sx(v)} y1={pad} x2={sx(v)} y2={height - pad} stroke={`rgba(232,235,239,${v === 0 ? 0.16 : 0.05})`} />
          <line x1={pad} y1={sy(v)} x2={width - pad} y2={sy(v)} stroke={`rgba(232,235,239,${v === 0 ? 0.16 : 0.05})`} />
          <text x={sx(v)} y={height - pad + 15} fill="#8C94A1" fontSize="10" textAnchor="middle" fontFamily="IBM Plex Mono, monospace">{v}</text>
          <text x={pad - 8} y={sy(v) + 3} fill="#8C94A1" fontSize="10" textAnchor="end" fontFamily="IBM Plex Mono, monospace">{v}</text>
        </g>
      ))}
      <text x={width / 2} y={height - 3} fill="#8C94A1" fontSize="10" textAnchor="middle">Horizontal break, in (arm side →)</text>
      <text x={10} y={height / 2} fill="#8C94A1" fontSize="10" textAnchor="middle" transform={`rotate(-90 10 ${height / 2})`}>Induced vertical break, in</text>
      {report.points.map((p, i) => (
        <circle key={i} cx={sx(p.hb_arm)} cy={sy(p.InducedVertBreak)} r={2.8} fill={pitchColor(p.shape_family)}
          fillOpacity={highlight && p.shape_family !== highlight ? 0.08 : 0.45} />
      ))}
      {report.movement_profile.map((m) => {
        const dim = !!highlight && m.pitch !== highlight;
        return (
          <g key={m.pitch} opacity={dim ? 0.2 : 1}>
            <circle cx={sx(m.hb_arm)} cy={sy(m.ivb)} r={6} fill={pitchColor(m.pitch)} stroke="#0D1015" strokeWidth={2} />
            {!dim && <text x={sx(m.hb_arm) + 10} y={sy(m.ivb) + 4} fill="#E8EBEF" fontSize="11">{m.pitch}</text>}
          </g>
        );
      })}
    </svg>
  );
}

/** "What the research says" table: now vs target for each trait, plus strike rate. */
export function TargetTable({ pitch }: { pitch: Pitch }) {
  const t = pitch.targets;
  if (!t) return <p className="pr-note">No research targets for this pitch yet.</p>;
  const rows = t.rows.filter((r) => !r.hide);
  const fb = rows.find((r) => r.vs_fb)?.fb_velo;
  const fmt = (trait: string, v: number) =>
    trait === "ivb" || trait === "hb" ? signed(v) : trait === "spin" ? Math.round(v).toLocaleString() : v.toFixed(1);
  const strike = pitch.rates?.strike, ref = pitch.rates_ref?.strike;
  return (
    <div className="pr-scroll">
      <table>
        <thead>
          <tr><th>Trait</th><th className="num">Now</th><th className="num">Target (vs. his FB)</th><th className="num">Stuff+ gain</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.trait}>
              <td>{r.label}</td>
              <td className={`num ${r.in_range ? "in" : "out"}`}>{fmt(r.trait, r.now)} {r.unit}</td>
              <td className="num">{r.in_range && !r.vs_fb ? <span className="soft">on target</span> : r.target_text}</td>
              <td className="num">{r.in_range ? <span className="soft">–</span> : signed(r.gain, 0)}</td>
            </tr>
          ))}
          {typeof strike === "number" && typeof ref === "number" && (
            <tr>
              <td>Strike %</td>
              <td className={`num ${strike >= ref - 3 ? "in" : "out"}`}>{Math.round(strike)}%</td>
              <td className="num">{Math.round(ref)}%+</td>
              <td className="num"><span className="soft">–</span></td>
            </tr>
          )}
        </tbody>
      </table>
      {typeof fb === "number" && (
        <p className="pr-note" style={{ margin: "8px 0 0" }}>
          Velocity target is set off his current {fb.toFixed(1)} mph fastball and moves up with it.
        </p>
      )}
    </div>
  );
}

/** Small multi-line trend chart (one line per pitch type) across outings. */
export function TrendChart({ title, series, outings, unit }: { title: string; series: { pitch: string; values: (number | null)[] }[]; outings: string[]; unit?: string }) {
  const w = 300, h = 150, padL = 34, padR = 20, padT = 10, padB = 24;
  const all = series.flatMap((s) => s.values).filter((v): v is number => typeof v === "number");
  if (!all.length || outings.length < 1) return null;
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 2) { lo -= 1; hi += 1; }
  const padv = (hi - lo) * 0.12; lo -= padv; hi += padv;
  const x = (i: number) => (outings.length === 1 ? (padL + w - padR) / 2 : padL + (i / (outings.length - 1)) * (w - padL - padR));
  const y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * (h - padT - padB);
  const yt = [lo + (hi - lo) * 0.15, (lo + hi) / 2, hi - (hi - lo) * 0.15];
  return (
    <div className="pr-panel">
      <h2>{title}{unit ? <span className="soft" style={{ fontWeight: 400 }}> · {unit}</span> : null}</h2>
      <svg viewBox={`0 0 ${w} ${h}`} width="100%" role="img" aria-label={`${title} by outing`}>
        {yt.map((v, i) => (
          <g key={i}>
            <line x1={padL} x2={w - padR} y1={y(v)} y2={y(v)} stroke="rgba(232,235,239,0.06)" />
            <text x={padL - 6} y={y(v) + 3} fill="#8C94A1" fontSize="9" textAnchor="end" fontFamily="IBM Plex Mono, monospace">{Math.abs(hi - lo) > 30 ? Math.round(v) : v.toFixed(1)}</text>
          </g>
        ))}
        {outings.map((o, i) => (
          <text key={o} x={x(i)} y={h - 6} fill="#8C94A1" fontSize="9" textAnchor="middle" fontFamily="IBM Plex Mono, monospace">{o.slice(5)}</text>
        ))}
        {series.map((s) => {
          const pts = s.values.map((v, i) => (typeof v === "number" ? [x(i), y(v)] : null)).filter((p): p is number[] => !!p);
          return (
            <g key={s.pitch}>
              {pts.length > 1 && <polyline points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke={pitchColor(s.pitch)} strokeWidth={1.6} />}
              {pts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={2.6} fill={pitchColor(s.pitch)} />)}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
