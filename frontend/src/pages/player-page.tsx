import { useEffect, useMemo } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { SiteFooter, SiteNav, SitePage } from "@/features/site/site-shell";
import { loadStaff, slug } from "@/features/staff/staff-store";
import { num, ordinal, pct, pitchColor, savant } from "@/features/staff/format";
import { PctBar, Status, Swatch, gradeBar, pctColor } from "@/features/staff/components/ui";
import { CountsTab, OverviewTab, PitchTab, ProgressTab } from "@/features/staff/components/tabs";
import "@/features/staff/staff.css";

const TABS = ["overview", "pitches", "counts", "progress"] as const;
type Tab = (typeof TABS)[number];

/** One pitcher's page: header stats, an arsenal strip, and four tabs. Tab and pitch live in the URL. */
export default function PlayerPage() {
  const { player = "" } = useParams();
  const [params, setParams] = useSearchParams();
  const navigate = useNavigate();
  const staff = useMemo(() => loadStaff(), []);
  const r = staff?.pitchers.find((p) => slug(p.pitcher) === player);

  const tab: Tab = (TABS as readonly string[]).includes(params.get("tab") ?? "") ? (params.get("tab") as Tab) : "overview";
  const pitchIdx = Math.min(Math.max(0, Number(params.get("pitch") ?? 0) || 0), Math.max(0, (r?.pitch_by_pitch.length ?? 1) - 1));

  useEffect(() => { window.scrollTo(0, 0); }, [player]);

  if (!staff || !r) {
    return (
      <SitePage>
        <SiteNav />
        <div className="pr" style={{ flex: 1 }}>
          <main className="pr-main" style={{ maxWidth: 640, paddingTop: 60 }}>
            <h1>{staff ? "Pitcher not found" : "No staff file loaded"}</h1>
            <p className="body">{staff ? "That pitcher isn't in the loaded staff file." : "Load your staff file in the workspace to see pitcher pages."}</p>
            <Link to="/workspace" className="pr-btn primary">Go to workspace</Link>
          </main>
        </div>
      </SitePage>
    );
  }

  const set = (next: { tab?: Tab; pitch?: number }) => {
    const p = new URLSearchParams(params);
    if (next.tab) p.set("tab", next.tab);
    if (next.pitch !== undefined) p.set("pitch", String(next.pitch));
    if ((next.tab ?? tab) !== "pitches") p.delete("pitch");
    setParams(p, { replace: true });
  };
  const fb = r.fastball_traits, d = r.development, fps = r.situations.first_pitch_strike;
  const cols = Math.min(Math.max(r.pitch_by_pitch.length, 1), 5);

  return (
    <SitePage>
      <SiteNav />
      <div className="pr" style={{ flex: 1 }}>
        <main className="pr-main">
          <div className="pr-crumbs">
            <div>
              <Link to="/workspace">Workspace</Link> / {staff.team} staff
              <select className="pr-select pr-noprint" aria-label="Choose pitcher" value={slug(r.pitcher)}
                onChange={(e) => navigate(`/workspace/${e.target.value}${tab !== "overview" ? `?tab=${tab}` : ""}`)}>
                {staff.pitchers.map((p) => <option key={p.pitcher} value={slug(p.pitcher)}>{p.pitcher}</option>)}
              </select>
            </div>
            <button type="button" className="pr-btn pr-noprint" onClick={() => window.print()}>Print / save PDF</button>
          </div>

          <div className="pr-head">
            <div>
              <div className="pr-label" style={{ marginBottom: 8 }}>{r.throws === "L" ? "LHP" : "RHP"} · {staff.team} · {staff.label}</div>
              <h1>{r.pitcher}</h1>
            </div>
            <div className="pr-meta mono">{r.pitches} pitches · {r.game_pitches} vs hitters · {r.outings} outings</div>
          </div>
          <p className="pr-summary">{r.summary}</p>

          <div className="pr-kpis">
            <div>
              <div className="pr-label">Fastball velo</div><div className="v">{num(fb?.velo)}</div>
              <div className="s">mph · {ordinal(fb?.velo_pct_d3)} pct D3</div>
              {typeof fb?.velo_pct_d3 === "number" && <PctBar pos={fb.velo_pct_d3 / 100} color={pctColor(fb.velo_pct_d3)} />}
            </div>
            <div>
              <div className="pr-label">Perceived velo</div><div className="v">{num(fb?.perceived_velo)}</div>
              <div className="s">mph · {num(fb?.extension_ft)} ft extension</div>
              {typeof fb?.perceived_pct_d3 === "number" && <PctBar pos={fb.perceived_pct_d3 / 100} color={pctColor(fb.perceived_pct_d3)} />}
            </div>
            <div>
              <div className="pr-label">Stuff+</div><div className="v">{d.stuff.stuff_plus ?? "–"}</div>
              <div className="s">{d.stuff.label}</div>
              {typeof d.stuff.stuff_plus === "number" && <PctBar {...gradeBar(d.stuff.stuff_plus)} />}
            </div>
            <div>
              <div className="pr-label">Location+</div><div className="v">{d.command.location_plus ?? "–"}</div>
              <div className="s">{pct(d.command.strike_pct, 1)} strikes</div>
              {typeof d.command.location_plus === "number" && <PctBar {...gradeBar(d.command.location_plus)} />}
            </div>
            <div><div className="pr-label">BB%</div><div className="v">{num(d.command.bb_pct)}</div><div className="s">{d.command.pa} PA</div></div>
            <div><div className="pr-label">First-pitch strike</div><div className="v">{fps ? pct(fps.pct, 1) : "–"}</div><div className="s">{fps ? `MLB ${fps.pro_range}` : ""}</div></div>
          </div>

          <div className="pr-tabs pr-noprint" role="tablist" aria-label="Sections">
            {TABS.map((t) => (
              <button key={t} type="button" role="tab" aria-selected={t === tab} onClick={() => set({ tab: t })}>{t}</button>
            ))}
          </div>

          <div className="pr-arsenal" style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}>
            {r.pitch_by_pitch.map((p, i) => {
              const [bg, fg] = savant(p.grades.pitch_plus);
              return (
                <button key={p.pitch} type="button" aria-pressed={tab === "pitches" && i === pitchIdx} onClick={() => set({ tab: "pitches", pitch: i })}
                  style={{ ["--pc" as string]: pitchColor(p.pitch) }}>
                  <div className="info">
                    <div className="name"><Swatch pitch={p.pitch} />{p.pitch} <span className="soft" style={{ fontWeight: 400 }}>{p.subtype}</span></div>
                    <div className="soft mono" style={{ fontSize: 11.5, margin: "4px 0 6px" }}>{p.usage_pct}% · {num(p.shape?.velo)} mph</div>
                    <Status verdict={p.verdict} />
                  </div>
                  <div className="pp" style={{ background: bg, color: fg }}>
                    <span className="k">PITCH+</span>
                    <span className="n">{typeof p.grades.pitch_plus === "number" ? Math.round(p.grades.pitch_plus) : "–"}</span>
                  </div>
                </button>
              );
            })}
          </div>

          {tab === "overview" && <OverviewTab r={r} />}
          {tab === "pitches" && <PitchTab r={r} p={r.pitch_by_pitch[pitchIdx]} />}
          {tab === "counts" && <CountsTab r={r} />}
          {tab === "progress" && <ProgressTab r={r} />}
        </main>
      </div>
      <SiteFooter />
    </SitePage>
  );
}
