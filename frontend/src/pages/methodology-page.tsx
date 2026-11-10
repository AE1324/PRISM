import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { FACTORS, SITE, SiteFooter, SiteNav, SitePage } from "@/features/site/site-shell";

const DISPLAY = "font-['Space_Grotesk',system-ui,sans-serif]";
const MONO = "font-['IBM_Plex_Mono',ui-monospace,monospace]";
const LINE = "border-[rgba(237,240,244,0.08)]";

const STEPS = [
  { id: "data", n: "01", title: "Collect", sub: "TrackMan, Rapsodo, Statcast, Synergy" },
  { id: "classify", n: "02", title: "Classify", sub: "Pitch type from shape, not tags" },
  { id: "stuff", n: "03", title: "Model", sub: "Stuff+ and Shape+ from 2.6M pitches" },
  { id: "d3", n: "04", title: "Calibrate", sub: "Put it on a Division III footing" },
  { id: "pitchplus", n: "05", title: "Grade", sub: "Location, results, Pitch+" },
  { id: "targets", n: "06", title: "Explain", sub: "Jobs, traits and targets" },
];

const SECTIONS = [
  { id: "data", title: "Data" },
  { id: "classify", title: "Pitch classification" },
  { id: "stuff", title: "Stuff+ and Shape+" },
  { id: "d3", title: "Calibrating to Division III" },
  { id: "location", title: "Location+" },
  { id: "results", title: "Results+" },
  { id: "pitchplus", title: "Pitch+" },
  { id: "jobs", title: "Jobs and verdicts" },
  { id: "fastball", title: "Fastball traits" },
  { id: "targets", title: "Research targets" },
  { id: "axis", title: "Spin axis" },
  { id: "samples", title: "Sample sizes" },
  { id: "limits", title: "Limits and what's next" },
];

function Section({ id, title, short, children, specs }: { id: string; title: string; short: ReactNode; children: ReactNode; specs?: [string, ReactNode][] }) {
  const n = String(SECTIONS.findIndex((s) => s.id === id) + 1).padStart(2, "0");
  return (
    <section id={id} className={`scroll-mt-6 border-t ${LINE} py-12`}>
      <p className={`m-0 mb-2 text-[13px] text-[#7D8694] ${MONO}`}>{n}</p>
      <h2 className={`m-0 mb-4 text-[28px] font-semibold leading-tight tracking-[-0.01em] ${DISPLAY}`}>{title}</h2>
      <p className="m-0 mb-5 text-[19px] leading-[1.55] text-[#EDF0F4]">{short}</p>
      <div className="flex flex-col gap-4 text-[16px] leading-[1.75] text-[#C2C8D1]">{children}</div>
      {specs && (
        <div className={`mt-6 overflow-hidden rounded-xl border ${LINE}`} style={{ background: SITE.surface }}>
          <p className="m-0 border-b border-[rgba(237,240,244,0.08)] px-5 py-2.5 text-[11px] font-semibold uppercase tracking-[0.14em] text-[#9AA3B1]">Technical detail</p>
          <dl className="m-0">
            {specs.map(([k, v]) => (
              <div key={k} className={`flex flex-wrap gap-x-6 gap-y-1 border-t ${LINE} px-5 py-3 first:border-t-0`}>
                <dt className="w-[190px] flex-none text-[13px] text-[#9AA3B1]">{k}</dt>
                <dd className={`m-0 min-w-0 flex-[1_1_300px] text-[13.5px] leading-relaxed text-[#D5DAE1] ${MONO}`}>{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}
    </section>
  );
}

const Formula = ({ children }: { children: ReactNode }) => (
  <div className={`overflow-x-auto rounded-xl border ${LINE} px-5 py-4 text-[14px] leading-relaxed text-[#EDF0F4] ${MONO}`} style={{ background: SITE.surface }}>{children}</div>
);

function MiniTable({ head, rows }: { head: string[]; rows: (string | number)[][] }) {
  return (
    <div className={`overflow-x-auto rounded-xl border ${LINE}`} style={{ background: SITE.surface }}>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr>{head.map((h, i) => <th key={h} className={`px-4 py-2.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-[#9AA3B1] ${i ? "text-right" : "text-left"}`}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={`border-t ${LINE}`}>
              {r.map((c, j) => <td key={j} className={`px-4 py-2.5 ${j ? `text-right ${MONO}` : ""}`}>{c}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function MethodologyPage() {
  const jump = (id: string) => () => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });

  return (
    <SitePage>
      <SiteNav />

      {/* ---------- hero + pipeline ---------- */}
      <header className="mx-auto w-full max-w-[1200px] px-6 pt-20 pb-10">
        <p className="m-0 mb-3 text-xs font-semibold uppercase tracking-[0.14em] text-[#9AA3B1]">Methodology</p>
        <h1 className={`m-0 mb-5 max-w-[860px] text-5xl font-semibold leading-[1.06] tracking-[-0.015em] ${DISPLAY}`}>
          How PRISM turns pitch data into grades and a plan.
        </h1>
        <p className="m-0 max-w-[760px] text-[17px] leading-[1.7] text-[#B3BAC5]">
          PRISM learns which pitch shapes work from millions of MLB pitches, puts that knowledge on a Division III footing,
          and grades every pitch an Oberlin pitcher throws. Each step below starts with the short version, and ends with the
          technical detail for anyone who wants to check the work.
        </p>

        <ol className="m-0 mt-12 grid list-none grid-cols-[repeat(auto-fit,minmax(170px,1fr))] gap-3 p-0">
          {STEPS.map((s, i) => (
            <li key={s.id}>
              <button type="button" onClick={jump(s.id)}
                className={`group h-full w-full cursor-pointer rounded-2xl border ${LINE} p-5 text-left transition-colors hover:border-[rgba(237,240,244,0.25)]`}
                style={{ background: SITE.surface }}>
                <span className="mb-4 block h-1 w-10 rounded-full" style={{ background: FACTORS[i % FACTORS.length].color }} />
                <span className={`block text-[12px] text-[#7D8694] ${MONO}`}>{s.n}</span>
                <span className={`block text-lg font-semibold text-[#EDF0F4] ${DISPLAY}`}>{s.title}</span>
                <span className="mt-1 block text-[13px] leading-snug text-[#9AA3B1]">{s.sub}</span>
              </button>
            </li>
          ))}
        </ol>
      </header>

      <div className="mx-auto flex w-full max-w-[1200px] flex-wrap gap-x-16 gap-y-10 px-6 pb-24">
        <aside className="min-w-0 flex-[0_1_220px]">
          <nav aria-label="Sections" className="sticky top-6 flex flex-col items-start gap-2 text-sm">
            <p className="m-0 mb-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[#7D8694]">On this page</p>
            {SECTIONS.map((s, i) => (
              <button key={s.id} type="button" onClick={jump(s.id)}
                className="cursor-pointer border-0 bg-transparent p-0 text-left text-[#B3BAC5] transition-colors hover:text-[#EDF0F4]">
                <span className={`mr-2 text-[#5F6876] ${MONO}`}>{String(i + 1).padStart(2, "0")}</span>{s.title}
              </button>
            ))}
          </nav>
        </aside>

        <article className="min-w-0 max-w-[760px] flex-[1_1_560px]">
          <Section id="data" title="Data"
            short="Every source is converted to one format, TrackMan's, before anything is graded, so a pitch means the same thing whether it came from a game, a bullpen or MLB."
            specs={[
              ["Internal format", "TrackMan column names, units and sign conventions; one adapter per source"],
              ["Model training", "MLB Statcast 2022–2025: 2.6M pitches, 1,098 pitchers, 7,517 pitcher-season pitch types"],
              ["Oberlin data", "Rapsodo (fall intrasquad, live hitters); results matched from AWRE where available"],
              ["Division III reference", "Synergy team data from 100 D3 programs, spread across team strength"],
              ["Privacy", "Player data never goes to the server or the public repo; staff files load in the browser only"],
            ]}>
            <p className="m-0">
              The models need volume and outcomes, so they learn from four seasons of MLB pitch tracking, where every pitch
              has a result attached. Oberlin's own pitches come from Rapsodo for now, with TrackMan as the native format so
              the system is ready the day the program has one. Division III data from Synergy is used for something
              narrower and just as important: knowing what normal looks like at this level.
            </p>
            <p className="m-0">
              Before grading, sessions are cleaned automatically. Misreads, pitches tagged to the wrong pitcher and
              impossible values are flagged for review instead of silently dropped.
            </p>
          </Section>

          <Section id="classify" title="Pitch classification"
            short="Device tags are ignored. Every pitch is classified from its own movement and speed compared to that pitcher's fastball."
            specs={[
              ["Model", "Gradient-boosted classifier (scikit-learn HistGradientBoosting) trained on MLB's own pitch tags"],
              ["Features", "velo, IVB and HB off the primary fastball; IVB; arm-side HB; spin direction (sin, cos; mirrored for LHP); spin ÷ fastball spin"],
              ["Agreement with MLB tags", "90% of individual pitches; 93% of pitcher pitch types (25+ pitches)"],
              ["Special cases", "under 400 rpm → knuckleball; rare stray tags (under 10 pitches) folded into his matching regular pitch"],
            ]}>
            <p className="m-0">
              PRISM first finds each pitcher's primary fastball from his hardest pitches. Every other pitch is then described
              relative to it: how much slower, how much more drop, how much more sweep. Raw velocity is deliberately left
              out, so an 82 mph D3 fastball still reads as a fastball. That gives a pitch type (four-seam, sinker, cutter,
              slider, curveball, changeup, splitter) and a bucket (fastball, breaking, offspeed).
            </p>
            <p className="m-0">On top of that, each pitch gets a descriptive subtype from the pitcher's average shape:</p>
            <MiniTable head={["Pitch", "Subtypes", "Rule of thumb"]} rows={[
              ["Four-seam", "Ride · Flat", "14+ in of IVB is ride"],
              ["Slider", "Sweeper · Gyro · Traditional", "10+ in glove side is a sweeper; under 5 is gyro"],
              ["Curveball", "Slurve · 12-6 · Standard", "slurve: sweep about equal to drop"],
              ["Changeup", "Fade · Drop", "4+ in more arm-side run than the fastball is fade"],
            ]} />
            <p className="m-0">
              Subtypes never change a grade. They decide which pitches a pitch is compared with when PRISM looks up research
              and targets, so a gyro slider is held to what works for gyro sliders, not sweepers.
            </p>
          </Section>

          <Section id="stuff" title="Stuff+ and Shape+"
            short="Stuff+ predicts how many runs a pitch will give up based only on how it's built. It's learned from MLB, where there's enough data to tell good shapes from lucky outcomes."
            specs={[
              ["Target", "expected run value per 100 pitches (xRV/100)"],
              ["Balls in play", "valued by the average run value of balls in play with similar expected wOBA, in the same count"],
              ["Training rows", "pitcher-season pitch types with 100+ pitches, weighted by pitch count"],
              ["Models", "3 gradient-boosted regressors (fastball, breaking, offspeed); settings chosen by cross-validation"],
              ["Fastball features", "perceived velo, IVB, HB, spin, release height, release side"],
              ["Secondary features", "perceived velo, IVB, HB, spin, plus velo / IVB / HB off his fastball"],
              ["Constraint", "monotonic in velocity: a harder pitch never grades worse than the same pitch thrown softer"],
              ["Validation", "5-fold cross-validation grouped by pitcher, scored on out-of-fold predictions only"],
            ]}>
            <p className="m-0">
              The target is <i>expected</i> run value. Balls, strikes and whiffs keep their real value, but a ball in play is
              valued by how well it was hit, not whether it found a hole. That strips out defense and luck, which would
              otherwise teach the model noise.
            </p>
            <Formula>
              perceived velo = release speed + 1.6 × (extension − 6.3 ft)<br />
              Stuff+ = 100 − 10 × (predicted xRV − average) ÷ SD &nbsp;<span className="text-[#7D8694]">(per pitch type)</span>
            </Formula>
            <p className="m-0">
              Secondary pitches are judged with their separation from the pitcher's own fastball, because a changeup's value
              is the gap it creates. <b className="text-[#EDF0F4]">Shape+</b> is the same model with velocity removed: it shows how well a pitch is
              designed, separately from how hard it's thrown.
            </p>
            <p className="m-0">Out-of-sample accuracy, on pitchers the model never saw:</p>
            <MiniTable head={["Model", "Rows", "Rank correlation", "Top vs bottom fifth"]} rows={[
              ["Fastball", "3,756", "0.25", "0.84 runs / 100"],
              ["Breaking", "2,603", "0.20", "0.72 runs / 100"],
              ["Offspeed", "1,156", "0.26", "0.95 runs / 100"],
            ]} />
            <p className="m-0">
              Those correlations are against a single season of results, which are themselves noisy, so they understate how
              well the model ranks pitches. The useful number is the last column: the top fifth of pitches by Stuff+ give up
              roughly 0.7 to 1 fewer run per 100 pitches than the bottom fifth.
            </p>
          </Section>

          <Section id="d3" title="Calibrating to Division III"
            short="MLB teaches PRISM which shapes work. Division III sets the bar. An 85 mph fastball is ordinary in MLB but in the top 15% in D3, and it's graded that way."
            specs={[
              ["Velocity scale", "pitcher-level D3 fastball percentiles from 100 Synergy teams, shrunk toward team-strength predictions and reweighted to match D3's strength spread"],
              ["D3 percentiles", "10th 78.1 · 25th 80.1 · 50th 82.3 · 75th 84.5 · 90th 86.5 · 99th 89.8 mph"],
              ["Translation", "his fastball's D3 percentile → the MLB speed at that percentile; every pitch shifts by the same amount"],
              ["Velocity weight", "2.0× on fastball velocity: whiffs climb faster with velocity in D3 than in MLB (capped at 2)"],
              ["Rate references", "strike% and whiff% by pitch type from Synergy D3, reweighted by team strength"],
            ]}>
            <p className="m-0">
              Feeding a D3 pitcher's raw velocity into an MLB model would grade almost every fastball as poor. Instead, PRISM
              asks where his fastball ranks among Division III pitchers, finds the MLB velocity at that same rank, and grades
              his whole arsenal as if it were thrown at that speed. Shapes and separation stay exactly as he throws them.
            </p>
            <p className="m-0">
              The D3 data also showed that velocity matters more at this level than in MLB: the jump in swing-and-miss rate
              from a slower fastball to a faster one is about twice as steep. PRISM weights fastball velocity accordingly.
            </p>
          </Section>

          <Section id="location" title="Location+"
            short="Location+ measures how often a pitch lands where hitters can't do much with it. Because location takes a lot of pitches to read, a small sample is pulled toward the pitcher's overall command."
            specs={[
              ["Competitive zone", "within 1.11 ft of the middle of the plate, 1.17–3.83 ft high"],
              ["Middle", "within 0.56 ft of center, 1.83–3.17 ft high"],
              ["Score", "% in the competitive zone − % in the middle"],
              ["Regression", "pitcher command → staff at n ÷ (n + 300); each pitch type → his command at n ÷ (n + 400)"],
            ]}>
            <p className="m-0">
              On Oberlin's fall data, the spread in location from one pitch type to another was 8.6 points, and 8.2 of that is
              what chance alone produces in 20 to 40 pitches. So a 17-pitch slider that happened to land well can't be graded
              like an elite command pitch. PRISM regresses in two steps: first each pitcher's overall command toward the
              staff, then each pitch type toward his own command. As pitches pile up, each pitch's own location takes over.
            </p>
          </Section>

          <Section id="results" title="Results+"
            short="Results+ is what actually happened against hitters, valued in runs."
            specs={[
              ["Non-contact pitches", "run value by count and outcome, from MLB"],
              ["Balls in play", "run value by count, exit velocity (5 mph bins) and launch angle (10° bins); play result if not tracked"],
              ["Scale", "runs per 100 game pitches, flipped and scaled so 100 = average and 10 = 1 SD"],
            ]}>
            <p className="m-0">
              Every pitch thrown to a live hitter is worth something: a 2-0 ball costs more than an 0-2 ball, and a 100 mph
              line drive costs more than a weak ground ball. Results+ adds those up. It's the most honest grade over a full
              season and the least reliable over a few outings, which is why Pitch+ trusts it gradually.
            </p>
          </Section>

          <Section id="pitchplus" title="Pitch+"
            short="Pitch+ blends what a pitch is with what it has done, and lets the results take over only once there are enough of them to mean something."
            specs={[
              ["Projection", "mean of Stuff+ and Location+"],
              ["Results weight", "n ÷ (n + 770), where n is game pitches"],
              ["Why 770", "on MLB pitcher seasons, a pitch type's results are half signal at about 770 game pitches"],
            ]}>
            <Formula>Pitch+ = projection + n ÷ (n + 770) × (Results+ − projection)</Formula>
            <MiniTable head={["Game pitches", "Results weight", "What it means"]} rows={[
              [17, "2%", "a few outings: graded on what the pitch is"],
              [50, "6%", "a strong fall"],
              [250, "25%", "about a full season"],
              [770, "50%", "results and traits count equally"],
            ]} />
            <p className="m-0">
              On Oberlin's data, the whole spread in Results from one pitch to another was what chance would produce, so
              trusting results early would mostly reward luck.
            </p>
          </Section>

          <Section id="jobs" title="Jobs and verdicts"
            short="A pitch earns a job when it's clearly better than average at something specific, and a verdict says whether it's working."
            specs={[
              ["Jobs", "strike, whiff, called strike, chase, weak contact, ground ball, putaway"],
              ["Clearly better", "at least 3 points better and z ≥ 1.28 (about 90% one-sided), with a minimum sample per rate"],
              ["References", "strike% and whiff% vs Division III; the other rates vs the Oberlin staff"],
              ["Deal-breakers", "clearly below average on strikes, or clearly above average on hard contact"],
              ["Verdict", "Working · Struggling · No clear job yet · Too few game pitches (under 15)"],
            ]}>
            <p className="m-0">
              Each pitch gets one job, the one where it beats average by the most, so a card says "whiff pitch," not a list of
              everything it does a little well. A working pitch is never told to change; it gets add-ons to try. A struggling
              pitch gets the fix that matters most first.
            </p>
          </Section>

          <Section id="fastball" title="Fastball traits"
            short="Three traits get called out on their own because they explain how a fastball plays and can be trained: perceived velocity, extension and approach angle."
            specs={[
              ["Extension", "compared to MLB and to what's typical for his height (about 1.015 × height)"],
              ["Approach angle", "a physics flight model with MLB-average release, extension and ride gives the expected angle for his speed and plate height"],
              ["Calibration", "per tracking system; Rapsodo reads about 0.4° steeper than Statcast"],
              ["Scaling", "residuals scaled by velocity ÷ 94 mph so angles compare across speeds"],
              ["Highlighted", "only when unusual: 85th percentile or higher, or 15th or lower"],
            ]}>
            <p className="m-0">
              A fastball's raw approach angle mostly reflects where it was thrown and how tall the pitcher is. PRISM compares
              it to what physics expects for his release, speed and location, so what's left is the part that comes from the
              pitch itself. A flat four-seam is called out as a strength; a flat or steep sinker is noted either way.
            </p>
          </Section>

          <Section id="targets" title="Research targets"
            short="Targets come from the Stuff model itself: PRISM tries realistic versions of the pitch and shows which changes would grade better."
            specs={[
              ["Range searched", "each trait moved on its own across the 10th–90th percentile of MLB pitches of the same subtype"],
              ["Velocity", "fastball: up only, to +4 mph. Secondaries: this pitch's own speed with the fastball held fixed"],
              ["Spin", "at most ±300 rpm"],
              ["Becomes a target", "worth 5+ Stuff+; the target is every value within 2 points of the best"],
              ["Secondary velo window", "2 mph, set off his current fastball (moves up as the fastball does)"],
              ["Slower", "only changeups and splitters that sit too close to the fastball, by grip, at fastball arm speed"],
            ]}>
            <p className="m-0">
              For each trait (velocity, ride, run, spin), PRISM holds everything else where the pitcher is and asks the model
              what the pitch would grade at each realistic value. If moving a trait is worth at least 5 Stuff+, it becomes a
              target, and the target range is the window that gets nearly all of the gain. Everything else shows as on target.
            </p>
            <p className="m-0">
              Because targets are judged within the pitch's subtype, a gyro slider is never told to become a sweeper. For
              secondary pitches, velocity is shown as a speed range off his current fastball, so it's clear the range rises
              as the fastball gets harder.
            </p>
          </Section>

          <Section id="axis" title="Spin axis"
            short="Spin direction is shown as a clock face, and for four pitch types the research shows which direction plays better."
            specs={[
              ["Direction", "movement direction from IVB and arm-side HB, as a clock face; mirrored for left-handers"],
              ["A different version", "12°+ off his normal direction, with at least 5 pitches"],
              ["Research", "2024–25 Statcast: each pitcher's pitch compared with itself at similar total break"],
            ]}>
            <MiniTable head={["Pitch", "Plays better", "Whiffs per swing"]} rows={[
              ["Four-seam", "more upright", "26% vs 22% at his normal · 17% drifting arm side"],
              ["Cutter", "more cut", "25% vs 22%"],
              ["Changeup", "more drop", "35–37% vs 32% · 25–30% toward the fastball"],
              ["Curveball", "more 12–6", "37–38% vs 32% at his normal"],
            ]} />
            <p className="m-0">
              Sliders, sinkers and splitters showed no clear direction, so their cards show the axis for reference only.
            </p>
          </Section>

          <Section id="samples" title="Sample sizes"
            short="PRISM would rather say 'not yet' than show a confident number built on a handful of pitches."
            specs={[
              ["Graded pitch type", "10+ pitches"],
              ["Location grade", "10+ located pitches"],
              ["Verdict", "15+ pitches against hitters"],
              ["Outing trends", "5+ of a pitch type in that outing"],
            ]}>
            <p className="m-0">
              Below these, the page shows N/A or says what's missing. Above them, location and results are still weighed by
              how much data is behind them, so grades firm up as the season goes on instead of jumping around.
            </p>
          </Section>

          <Section id="limits" title="Limits and what's next"
            short="PRISM is honest about what it can't see yet.">
            <ul className="m-0 flex list-disc flex-col gap-2 pl-5 marker:text-[#5F6876]">
              <li>Shape and location grades are centered on MLB pitch types until enough Division III TrackMan data is available to recenter them.</li>
              <li>Rapsodo and TrackMan measure slightly differently; known offsets (like approach angle) are corrected, and others will be as data allows.</li>
              <li>Fall samples are small. The regression built into Location+ and Pitch+ exists for exactly that reason.</li>
              <li>Next: D3 TrackMan data to recenter shape and location, more team-strength bands for the velocity weight, and secure logins so pages update automatically.</li>
            </ul>
          </Section>
        </article>
      </div>

      <SiteFooter>
        <Link to="/measures" className="text-[#9AA3B1] hover:text-[#EDF0F4]">← What it measures</Link>
      </SiteFooter>
    </SitePage>
  );
}
