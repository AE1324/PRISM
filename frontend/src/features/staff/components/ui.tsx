import type { ReactNode } from "react";
import type { Grade, Pitch, PitcherReport } from "../types";
import { fmtGrade, gradePos, pitchColor, savant, signed, verdictTone, sentence } from "../format";

export function GradeCell({ v }: { v: Grade | undefined }) {
  const [bg, fg] = savant(v ?? null);
  return (
    <td className="g">
      <span style={{ background: bg, color: fg }}>{fmtGrade(v)}</span>
    </td>
  );
}

export function Status({ verdict }: { verdict: string }) {
  return <span className="pr-status" style={{ color: verdictTone(verdict) }}>{sentence(verdict)}</span>;
}

/** Square pitch-type swatch (squares read sharper than dots in tables). */
export function Swatch({ pitch }: { pitch: string }) {
  return <span className="pr-sw" style={{ background: pitchColor(pitch) }} aria-hidden="true" />;
}

/** A numbered report section with a ruled header. */
export function Section({ idx, title, aside, children }: { idx: string; title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="pr-sec">
      <header>
        <span className="idx">{idx}</span>
        <h2>{title}</h2>
        {aside && <span className="aside">{aside}</span>}
      </header>
      {children}
    </section>
  );
}

/** Thin percentile bar: filled to the value, Savant color, tick at the value. `pos` is 0-1. */
export function PctBar({ pos, color }: { pos: number; color: string }) {
  const p = Math.max(0, Math.min(1, pos)) * 100;
  return (
    <div className="pr-bar" aria-hidden="true">
      <i style={{ width: `${p}%`, background: color }} />
      <b style={{ left: `calc(${p}% - 1px)` }} />
    </div>
  );
}
export const pctColor = (pctile: number) => savant(70 + (pctile / 100) * 60)[0];
export const gradeBar = (v: number) => ({ pos: gradePos(v), color: savant(v)[0] });

/** 1-SD ellipse for a pitch type's movement (principal axes of its covariance). */
function ellipse(xs: number[], ys: number[]) {
  const n = xs.length;
  if (n < 5) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let sxx = 0, syy = 0, sxy = 0;
  for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy; }
  sxx /= n - 1; syy /= n - 1; sxy /= n - 1;
  const tr = sxx + syy, det = sxx * syy - sxy * sxy, disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(0, tr / 2 - disc);
  const ang = Math.atan2(l1 - sxx, sxy || 1e-9);
  return { mx, my, rx: Math.sqrt(l1), ry: Math.sqrt(l2), ang };
}

/**
 * Movement chart: each pitch as a dot in its Savant color, a 1-SD ellipse around each pitch type,
 * and the average as a ringed marker with a label. Pitcher's view, arm side to the right.
 */
export function MovementChart({ report, highlight, size = 460 }: { report: PitcherReport; highlight?: string; size?: number }) {
  const w = size, h = size, pad = 38;
  const lim = 25;
  const sx = (v: number) => pad + ((v + lim) / (2 * lim)) * (w - 2 * pad);
  const sy = (v: number) => h - pad - ((v + lim) / (2 * lim)) * (h - 2 * pad);
  const unit = (w - 2 * pad) / (2 * lim);
  const ticks = [-20, -10, 0, 10, 20];
  const label = `Movement chart for ${report.pitcher}: ${report.movement_profile.map((m) => `${m.pitch} ${m.ivb} in vertical, ${m.hb_arm} in horizontal`).join("; ")}`;
  const byType = new Map<string, { xs: number[]; ys: number[] }>();
  for (const p of report.points) {
    const g = byType.get(p.shape_family) ?? { xs: [], ys: [] };
    g.xs.push(p.hb_arm); g.ys.push(p.InducedVertBreak); byType.set(p.shape_family, g);
  }
  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="pr-chart" role="img" aria-label={label}>
      <rect x={pad} y={pad} width={w - 2 * pad} height={h - 2 * pad} fill="none" stroke="rgba(241,243,246,0.18)" />
      {ticks.map((v) => (
        <g key={v}>
          {v !== 0 && <line x1={sx(v)} y1={pad} x2={sx(v)} y2={h - pad} stroke="rgba(241,243,246,0.06)" />}
          {v !== 0 && <line x1={pad} y1={sy(v)} x2={w - pad} y2={sy(v)} stroke="rgba(241,243,246,0.06)" />}
          <text x={sx(v)} y={h - pad + 15} fill="#8A929E" fontSize="10" textAnchor="middle" fontFamily="IBM Plex Mono, monospace">{v}</text>
          <text x={pad - 7} y={sy(v) + 3} fill="#8A929E" fontSize="10" textAnchor="end" fontFamily="IBM Plex Mono, monospace">{v}</text>
        </g>
      ))}
      <line x1={sx(0)} y1={pad} x2={sx(0)} y2={h - pad} stroke="rgba(241,243,246,0.35)" />
      <line x1={pad} y1={sy(0)} x2={w - pad} y2={sy(0)} stroke="rgba(241,243,246,0.35)" />
      <text x={w - pad} y={h - 5} fill="#8A929E" fontSize="10" textAnchor="end">Horizontal break (in) · arm side →</text>
      <text x={12} y={pad} fill="#8A929E" fontSize="10" textAnchor="end" transform={`rotate(-90 12 ${pad})`}>Induced vertical break (in)</text>
      {[...byType.entries()].map(([fam, g]) => {
        const e = ellipse(g.xs, g.ys);
        if (!e) return null;
        const dim = !!highlight && fam !== highlight;
        return (
          <ellipse key={fam} cx={sx(e.mx)} cy={sy(e.my)} rx={e.rx * unit} ry={e.ry * unit}
            transform={`rotate(${(-e.ang * 180) / Math.PI} ${sx(e.mx)} ${sy(e.my)})`}
            fill={pitchColor(fam)} fillOpacity={dim ? 0.02 : 0.1} stroke={pitchColor(fam)} strokeOpacity={dim ? 0.12 : 0.7} strokeWidth={1.2} />
        );
      })}
      {report.points.map((p, i) => {
        const dim = !!highlight && p.shape_family !== highlight;
        return <circle key={i} cx={sx(p.hb_arm)} cy={sy(p.InducedVertBreak)} r={2.6} fill={pitchColor(p.shape_family)} fillOpacity={dim ? 0.12 : 0.85} />;
      })}
      {report.movement_profile.map((m) => {
        const dim = !!highlight && m.pitch !== highlight;
        const x = sx(m.hb_arm), y = sy(m.ivb), right = m.hb_arm < 12;
        return (
          <g key={m.pitch} opacity={dim ? 0.25 : 1}>
            <circle cx={x} cy={y} r={6.5} fill="#0B0D11" stroke="#F1F3F6" strokeWidth={2} />
            <circle cx={x} cy={y} r={3} fill={pitchColor(m.pitch)} />
            {!dim && (
              <text x={right ? x + 11 : x - 11} y={y - 8} fill="#F1F3F6" fontSize="11.5" fontWeight={600} textAnchor={right ? "start" : "end"}
                stroke="#0B0D11" strokeWidth={3} paintOrder="stroke">{m.pitch}</text>
            )}
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
              <td className="num">{r.beyond ? <span className="soft">{r.target_text}</span> : r.in_range && !r.vs_fb ? <span className="soft">on target</span> : r.target_text}</td>
              <td className="num">{r.in_range ? <span className="soft">–</span> : <b>{signed(r.gain, 0)}</b>}</td>
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
  const w = 300, h = 160, padL = 34, padR = 20, padT = 10, padB = 24;
  const all = series.flatMap((s) => s.values).filter((v): v is number => typeof v === "number");
  if (!all.length || outings.length < 1) return null;
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi - lo < 2) { lo -= 1; hi += 1; }
  const padv = (hi - lo) * 0.12; lo -= padv; hi += padv;
  const x = (i: number) => (outings.length === 1 ? (padL + w - padR) / 2 : padL + (i / (outings.length - 1)) * (w - padL - padR));
  const y = (v: number) => padT + (1 - (v - lo) / (hi - lo)) * (h - padT - padB);
  const yt = [lo + (hi - lo) * 0.15, (lo + hi) / 2, hi - (hi - lo) * 0.15];
  return (
    <div>
      <div className="pr-colhead">{title}{unit ? <span className="soft" style={{ fontWeight: 400 }}> · {unit}</span> : null}</div>
      <div className="pr-frame">
        <svg viewBox={`0 0 ${w} ${h}`} width="100%" role="img" aria-label={`${title} by outing`}>
          {yt.map((v, i) => (
            <g key={i}>
              <line x1={padL} x2={w - padR} y1={y(v)} y2={y(v)} stroke="rgba(241,243,246,0.07)" />
              <text x={padL - 6} y={y(v) + 3} fill="#8A929E" fontSize="9" textAnchor="end" fontFamily="IBM Plex Mono, monospace">{Math.abs(hi - lo) > 30 ? Math.round(v) : v.toFixed(1)}</text>
            </g>
          ))}
          {outings.map((o, i) => (
            <text key={o} x={x(i)} y={h - 6} fill="#8A929E" fontSize="9" textAnchor="middle" fontFamily="IBM Plex Mono, monospace">{o.slice(5)}</text>
          ))}
          {series.map((s) => {
            const pts = s.values.map((v, i) => (typeof v === "number" ? [x(i), y(v)] : null)).filter((p): p is number[] => !!p);
            return (
              <g key={s.pitch}>
                {pts.length > 1 && <polyline points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke={pitchColor(s.pitch)} strokeWidth={1.8} />}
                {pts.map((p, i) => <rect key={i} x={p[0] - 2.6} y={p[1] - 2.6} width={5.2} height={5.2} fill={pitchColor(s.pitch)} />)}
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

export const TIER_TEXT: Record<string, { short: string; long: string }> = {
  measured: { short: "Measured", long: "measured contact: exit velocity and launch angle, judged against this level's batted balls" },
  charted: { short: "Charted", long: "charted contact: batted-ball type and weak/average/hard, as entered" },
  outcome: { short: "Outcomes only", long: "outcomes only: no contact data, so results count the least" },
};

/** How much is known about this pitch's balls in play, and how much its results count toward Pitch+. */
export function TierTag({ tier, weight }: { tier?: string; weight?: number }) {
  if (!tier || !TIER_TEXT[tier]) return null;
  return (
    <span className={`pr-tier ${tier}`} title={`Results+ from ${TIER_TEXT[tier].long}.`}>
      Results: {TIER_TEXT[tier].short}{typeof weight === "number" ? ` · ${weight}% of Pitch+` : ""}
    </span>
  );
}
