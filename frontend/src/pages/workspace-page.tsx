import { useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { SiteFooter, SiteNav, SitePage } from "@/features/site/site-shell";
import { bestPitch, clearStaff, loadStaff, parseStaffFile, saveStaff, slug } from "@/features/staff/staff-store";
import type { StaffFile } from "@/features/staff/types";
import { num } from "@/features/staff/format";
import { Dot, Score } from "@/features/staff/components/ui";
import "@/features/staff/staff.css";

/**
 * Workspace home. With a staff file loaded it lists the staff; each row opens that pitcher's page.
 * Without one it asks for the file. The file stays in this browser only.
 */
export default function WorkspacePage() {
  const [staff, setStaff] = useState<StaffFile | null>(() => loadStaff());
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const accept = async (f: File | undefined) => {
    if (!f) return;
    setError("");
    try {
      const s = await parseStaffFile(f);
      setStaff(s);
      setNote(saveStaff(s) ? "" : "Loaded for this visit only: this browser wouldn't let PRISM keep the file.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Couldn't read that file.");
    }
  };

  const picker = (
    <input ref={input} type="file" accept=".json,application/json" hidden
      onChange={(e) => { accept(e.target.files?.[0]); e.target.value = ""; }} />
  );

  return (
    <SitePage>
      <SiteNav />
      <div className="pr" style={{ flex: 1 }}>
        <main className="pr-main">
          {!staff ? (
            <div style={{ maxWidth: 640, margin: "40px auto 0" }}>
              <h1>Workspace</h1>
              <p className="body" style={{ margin: "8px 0 22px" }}>
                Load your staff file to open every pitcher's page. The file stays in this browser; nothing is uploaded.
              </p>
              <div
                className={`pr-drop ${dragging ? "on" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files?.[0]); }}
              >
                <p style={{ margin: "0 0 12px" }}>Drop a staff file here</p>
                <button type="button" className="pr-btn primary" onClick={() => input.current?.click()}>Choose file</button>
                {picker}
                <p className="pr-note" style={{ margin: "14px 0 0" }}>
                  Make one on your computer with <code>python -m prism_model.export_staff</code>
                </p>
              </div>
              {error && <p className="pr-err" role="alert">{error}</p>}
              <p className="pr-note" style={{ marginTop: 28 }}>
                Have a single Rapsodo session instead? <Link to="/workspace/session" style={{ color: "var(--pr-ink)" }}>Open the session dashboards</Link>.
              </p>
            </div>
          ) : (
            <>
              <div className="pr-crumbs">
                <span>Workspace</span>
                <span style={{ display: "flex", gap: 8 }} className="pr-noprint">
                  <Link to="/workspace/session" className="pr-btn">Session dashboards</Link>
                  <button type="button" className="pr-btn" onClick={() => input.current?.click()}>Load a different file</button>
                  <button type="button" className="pr-btn" onClick={() => { clearStaff(); setStaff(null); }}>Remove from this browser</button>
                  {picker}
                </span>
              </div>
              <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
                <h1>{staff.team} staff</h1>
                <span className="soft">{staff.label} · {staff.pitchers.length} pitchers · updated {new Date(staff.generated).toLocaleDateString()}</span>
              </div>
              {error && <p className="pr-err" role="alert">{error}</p>}
              {note && <p className="pr-note">{note}</p>}
              <div className="pr-panel" style={{ marginTop: 18, padding: "6px 8px" }}>
                <div className="pr-scroll">
                  <table className="pr-roster">
                    <thead>
                      <tr><th>Pitcher</th><th>Throws</th><th className="num">Pitches</th><th className="num">Outings</th><th className="num">FB velo</th><th className="num">Stuff+</th><th className="num">Location+</th><th>Arsenal</th><th className="c">Best Pitch+</th></tr>
                    </thead>
                    <tbody>
                      {staff.pitchers.map((p) => {
                        const best = bestPitch(p);
                        const to = `/workspace/${slug(p.pitcher)}`;
                        return (
                          <tr key={p.pitcher} onClick={() => navigate(to)}>
                            <td><Link to={to} style={{ fontWeight: 500 }}>{p.pitcher}</Link></td>
                            <td className="soft">{p.throws === "L" ? "LHP" : "RHP"}</td>
                            <td className="num">{p.pitches}</td>
                            <td className="num">{p.outings}</td>
                            <td className="num">{num(p.fastball_traits?.velo)}</td>
                            <td className="num">{p.development.stuff.stuff_plus ?? "–"}</td>
                            <td className="num">{p.development.command.location_plus ?? "–"}</td>
                            <td><span style={{ display: "inline-flex", alignItems: "center" }}>{p.pitch_by_pitch.map((x) => <Dot key={x.pitch} pitch={x.pitch} />)}</span></td>
                            <td className="c">{best ? <span title={best.pitch}><Score v={best.value} /></span> : <span className="soft">–</span>}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
              <p className="pr-note" style={{ marginTop: 10 }}>This staff file is stored only in this browser.</p>
            </>
          )}
        </main>
      </div>
      <SiteFooter />
    </SitePage>
  );
}
