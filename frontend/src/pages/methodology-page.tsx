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
  { id: "devices", title: "Devices" },
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
              ["Oberlin data", "Rapsodo (fall 2026 live sessions); results matched from AWRE where available"],
              ["College TrackMan", "NECBL summer games (both teams): the measured college reference for exit velocity, location and device checks"],
              ["Division III reference", "Synergy team data from 100 D3 programs, spread across team strength"],
              ["Session tags", "device (TrackMan, Rapsodo) and setting (game, live, bullpen); only game and live are graded"],
              ["Misread checks", "impossible values; wrong side of the rubber; release 1+ ft from his usual spot; unlike his other pitches of that type; speed 4+ mph off his usual and 2+ mph from every other one"],
              ["Game checks (TrackMan)", "a game's movement is set aside if over half its pitches have none, or every pitcher's fastball reads 3+ in off his other games"],
              ["Privacy", "Player data never goes to the server or the public repo; staff files load in the browser only"],
            ]}>
            <p className="m-0">
              The models need volume and outcomes, so they learn from four seasons of MLB pitch tracking, where every pitch
              has a result attached. Oberlin's own pitches come from Rapsodo for now, with TrackMan as the native format so
              the system is ready the day the program has one. Division III data from Synergy is used for something
              narrower and just as important: knowing what normal looks like at this level.
            </p>
            <p className="m-0">
              Every session is tagged with its device and setting: game, live or bullpen. Only game and live sessions are
              graded; bullpens are kept but never counted. The tag alone decides, never a guess from velocity.
            </p>
            <p className="m-0">
              Before grading, sessions are cleaned automatically, and every flag goes to a review sheet instead of being
              silently dropped. PRISM flags impossible values, pitches released from the wrong side or far from the
              pitcher's usual release point (naming the likely pitcher when it matches someone else in the same game),
              pitches unlike his others of the same type, and a pitch whose speed sits alone, 4+ mph off his usual for
              that pitch and 2+ mph from every other one he threw. For TrackMan, each game's movement is also checked as a
              whole. Calibration is set per game, so a bad day shifts every pitcher the same way. If most pitches have no
              reliable movement, or every pitcher's fastball reads 3+ inches off his other games, that game's movement is
              set aside. Its pitches still count for velocity, location and results.
            </p>
          </Section>

          <Section id="devices" title="Devices"
            short="Rapsodo and TrackMan measure the same pitch in very different ways, so each device has a profile that says how far to trust each kind of number."
            specs={[
              ["TrackMan", "radar: measures release, flight, plate location and batted balls. Results+ can be measured; untagged sessions are games; movement checked per game"],
              ["Rapsodo", "camera behind the plate: spin read at release, movement from the flight it sees, plate location projected from early flight (about 6 in of error, built into Location+). Results+ at best charted; untagged sessions are bullpens"],
              ["Unknown device", "treated like the least certain one"],
              ["Known offsets", "recorded with their evidence; only applied once enough pitchers confirm them (none yet)"],
            ]}>
            <p className="m-0">
              TrackMan is a radar in the stands that follows the ball from release to the plate and off the bat. Rapsodo is a
              camera unit behind the plate that reads spin at release and projects the rest. Both are good at what they're
              built for, but they don't produce the same kind of number for everything, so PRISM keeps one profile per device
              and the grades read from it. Adding a device means adding a profile.
            </p>
            <p className="m-0">
              Plate location is the biggest difference: TrackMan measures where the ball crosses the plate, while Rapsodo
              projects it from early flight, with about 6 inches of error per pitch by our comparison. Location+ accounts
              for that blur rather than ignoring it (see Location+).
            </p>
            <p className="m-0">
              Two pitchers have now been on both devices, each a few months apart: a lefty on one indoor Rapsodo session
              against TrackMan games, and a righty on fall Rapsodo lives against three TrackMan bullpens. The clearest
              pattern: Rapsodo read less horizontal break on every pitch compared, by 1 to 3.5 inches, and less fastball
              ride for both (1 and 2 inches). Spin axis and approach angle agreed. Velocity, spin rate, extension and
              release point differed in opposite directions for the two, which points to the pitchers, not the devices.
              The righty also changed his own fastball run by almost 4 inches between two TrackMan sessions, so the
              offsets stay recorded, not applied, until a same-day comparison confirms them.
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
              ["Four-seam", "Ride · Cut-ride · Ride-run · Cut · Run · Deadzone", "14+ in ride: cut-ride under 6 in run, ride 6–12, ride-run over 12. Under 14: cut under 6 in run, run over 14. Deadzone: 10–14 in of both"],
              ["Slider", "Sweeper · Gyro · Traditional", "10+ in glove side is a sweeper; under 5 is gyro"],
              ["Curveball", "Slurve · 12-6 · Standard", "slurve: sweep about equal to drop"],
              ["Changeup", "Fade-drop · Fade · Drop · Firm · Standard", "vs his fastball: fade 6+ in more run; drop 9+ in less ride; firm under 6 in less ride"],
            ]} />
            <p className="m-0">
              Four-seams use the standard fastball-shape zones. A shape between zones takes the nearest one (ties go to
              deadzone). The deadzone, 10 to 14 inches of both ride and run, is where four-seams have nowhere to hide: not
              enough carry to ride and not enough movement to cut or run. A changeup is defined by how it separates from his own fastball, because that separation is the pitch: 9+
              inches less ride works whatever the run, 6+ inches more run works and works best with depth, and a changeup
              with under 6 inches of either sits too close to the fastball. The same 18 inches of run is a fade changeup off
              a cutting four-seam and an ordinary one off a high-run sinker.
            </p>
            <p className="m-0">
              Subtypes never change a grade. They keep a pitch in its own lane when PRISM looks for targets: a cut fastball
              is compared with cut fastballs and its targets stay inside the cut zone, because a cutting fastball usually
              comes from how a pitcher releases the ball, and no target will turn it into a ride-run one. A sweeper is held
              to sweepers the same way. The weak shapes are the exception: a deadzone four-seam or a firm changeup is
              compared with every pitch of its kind, so its targets can push it out toward whatever grades better.
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
              ["Translation", "his fastball's D3 percentile → the MLB speed at that percentile; the ratio between the two is his arm-speed factor"],
              ["Scaled by that factor", "velocity of every pitch, spin rate (spin per mph unchanged), separation off the fastball (same % off)"],
              ["Left as thrown", "movement (with spin scaled to speed, break stays about the same), release point, extension"],
              ["Velocity weight", "2.0× on fastball velocity: whiffs climb faster with velocity in D3 than in MLB (capped at 2)"],
              ["Rate references", "strike% and whiff% by pitch type from Synergy D3, reweighted by team strength"],
            ]}>
            <p className="m-0">
              Feeding a D3 pitcher's raw velocity into an MLB model would grade almost every fastball as poor. Instead, PRISM
              asks where his fastball ranks among Division III pitchers, finds the MLB velocity at that same rank, and grades
              his whole arsenal as if he threw it with that arm speed.
            </p>
            <p className="m-0">
              Arm speed doesn't only change velocity. For the same grip and release, a pitch thrown harder also spins
              harder, so spin is scaled by the same factor. Oberlin's pitchers spin the ball at the same rate per mph as
              MLB pitchers, about 24 to 25 rpm per mph on four-seams, so leaving spin alone would grade a translated
              fastball as low-spin when it isn't. Separation is kept as the same percentage off the fastball, which keeps
              the hitter's relative timing gap the same. Movement stays as thrown: when spin rises with speed, the extra
              Magnus force and the shorter flight time cancel out, so the break is about the same. Every number on a
              pitcher's page is still what he actually threw; only the model's view of the pitch is translated.
            </p>
            <p className="m-0">
              The D3 data also showed that velocity matters more at this level than in MLB: the jump in swing-and-miss rate
              from a slower fastball to a faster one is about twice as steep. PRISM weights fastball velocity accordingly.
            </p>
          </Section>

          <Section id="location" title="Location+"
            short="Location+ is what a pitcher's spots are worth in runs, for that pitch type in that count, judged against the Division III average."
            specs={[
              ["Location value", "an MLB model (2024–25 Statcast) of the run value at each plate location, by pitch type, count and same- vs opposite-handed hitter; batted balls valued by exit velocity and launch angle"],
              ["Per pitch", "the model's value at his spot minus the average for that pitch type, count and matchup; unknown count or hitter side averaged over how often each happens in MLB"],
              ["Center (100)", "Division III average: Synergy D3 take rates through the MLB take-rate fit, corrected by measured college data (NECBL TrackMan, both teams)"],
              ["Device", "Rapsodo pitches are held to the D3 average as read through 6 in of plate-location blur (about 0.008–0.010 runs per pitch); TrackMan as measured"],
              ["Regression", "his command counts n ÷ (n + 100) from D3 average (measured on 38 college pitchers); each pitch type n ÷ (n + 430) from his command (MLB). n = located pitches"],
              ["Scale", "10 points = one standard deviation of MLB pitcher-season pitch types (0.0044 runs per pitch)"],
              ["Validation", "on MLB it predicts runs on a pitcher's other pitches (r = 0.10) where the old zone-box score didn't (r = −0.03)"],
            ]}>
            <p className="m-0">
              Where a pitch should go depends on what it is and the count. A slider buried low and away with two strikes is
              worth a lot, while the same spot on a 3-1 fastball is a walk risk. So instead of two fixed boxes, every
              location is valued by what actually happens there in MLB for that pitch type, in that count, against a
              same- or opposite-handed hitter. A pitch earns credit for how much its spot helped compared to an average
              spot for the same situation, so a pitcher isn't rewarded or punished for the counts he happened to be in.
            </p>
            <p className="m-0">
              100 is the Division III average, a fixed standard, not the rest of the staff, so a staff that struggles to
              command the ball doesn't grade itself up. Synergy has D3 take rates but no locations. In MLB, a pitch type's
              taken-ball rate tracks its location value well, but college hitters chase more, which makes take rates
              flatter college command. The NECBL TrackMan games show how much: both teams' measured locations were worth
              about 0.007 runs per pitch more than their take rates implied. The D3 average uses that measured
              correction. As a check, Location+ follows real strike-throwing: on both staffs it correlates 0.63 to 0.72
              with each pitcher's strike rate, against 0.35 for the old version.
            </p>
            <p className="m-0">
              Rapsodo projects plate location from the first part of the flight, so each pitch carries some error. Compared
              with NECBL TrackMan at the same strike rate, Oberlin's Rapsodo locations graded about 0.010 runs per pitch
              worse and scattered more, and adding 6 inches of blur to the TrackMan locations reproduces both. Blur moves
              pitches off good spots on average, so a Rapsodo pitch is held to the D3 average as it would read through the
              same 6 inches of blur (measured by blurring college TrackMan locations). Before this, almost no Rapsodo pitch
              could grade above average no matter where it went.
            </p>
            <p className="m-0">
              Location takes pitches to read, so a small sample leans on the pitcher's overall command, and that command
              leans on D3 average until he's thrown enough. How fast it's trusted depends on how different pitchers really
              are. College command varies about twice as much as MLB command, so a college sample says more: his command
              is half trusted at 100 located pitches (MLB's spread would need 270). Measured on 38 college pitchers, NECBL
              (both teams) and Oberlin. Within a pitcher, pitch-type differences in command weren't detectable beyond
              chance in college data, so each pitch type still leans on his overall command (half at 430 pitches, from MLB).
            </p>
          </Section>

          <Section id="results" title="Results+"
            short="Results+ is what actually happened against hitters, valued in runs, with a tag for how much is known about the contact."
            specs={[
              ["Non-contact pitches", "run value by count and outcome, from MLB"],
              ["Balls in play", "12 categories: ground ball, line drive, fly ball or popup × weak, average or hard contact, each worth its MLB average run value by count"],
              ["Measured", "TrackMan: type from launch angle (Statcast bands: under 10°, 10–25°, 25–50°, 50°+); contact from exit velocity ranked among that level's batted balls"],
              ["Charted", "batted-ball type and weak/average/hard as entered by whoever charts"],
              ["Outcomes only", "the play result (single, out…) valued by its MLB average"],
              ["Scale", "runs per 100 game pitches, flipped and scaled so 100 = average and 10 = 1 SD"],
            ]}>
            <p className="m-0">
              Every pitch thrown to a live hitter is worth something: a 2-0 ball costs more than an 0-2 ball, and a hard line
              drive costs more than a weak ground ball. Results+ adds those up. It's the most honest grade over a full season
              and the least reliable over a few outings, which is why Pitch+ trusts it gradually.
            </p>
            <p className="m-0">
              How much is known about a ball in play depends on how it was recorded, so every Results+ carries a tag.
              Measured means TrackMan exit velocity and launch angle. Charted means someone recorded the batted-ball type and
              whether it was hit weak, average or hard. Outcomes only means just the result. All three use the same 12
              categories; what changes is how a ball gets placed in one.
            </p>
            <p className="m-0">
              Hard contact means hard for that level. Exit velocities run about 14 mph lower in summer college ball than in
              MLB at the median, so a measured exit velocity is ranked among every batted ball in that level's games, both
              teams, and judged by that rank. A charter's "hard" is already relative to what they're watching, so charted
              calls are taken as entered.
            </p>
            <p className="m-0">
              Even measured balls are valued by category rather than by exact exit velocity and launch angle. On MLB pitcher
              seasons, the 12 categories predicted a pitcher's runs on the rest of his pitches better than exact 5 mph × 10°
              cells, and coarser beat finer every time. For a pitcher, the exact numbers mostly add luck: a 101 mph liner at
              the shortstop and one that finds the gap count very differently.
            </p>
          </Section>

          <Section id="pitchplus" title="Pitch+"
            short="Pitch+ blends what a pitch is with what it has done, and lets the results take over only once there are enough of them to mean something."
            specs={[
              ["Projection", "mean of Stuff+ and Location+"],
              ["Results weight", "n ÷ (n + k), where n is game pitches and k depends on how the balls in play were recorded"],
              ["k", "measured 400 · charted 440 · outcomes only 1,000"],
              ["Shown when", "15+ pitches vs hitters (game or live sessions); below that the card shows Stuff+, Shape+ and Location+ without a Pitch+"],
              ["How k was measured", "the game pitches at which a pitch type's results predict runs on the rest of his pitches as much as chance moves them (MLB 2024–25 pitcher seasons, split in random halves; charted with 1 in 5 calls one category off)"],
            ]}>
            <Formula>Pitch+ = projection + n ÷ (n + k) × (Results+ − projection)</Formula>
            <MiniTable head={["Game pitches", "Measured", "Outcomes only", "What it means"]} rows={[
              [17, "4%", "2%", "a few outings: graded on what the pitch is"],
              [50, "11%", "5%", "a strong fall"],
              [250, "38%", "20%", "about a full season"],
              [1000, "71%", "50%", "results lead"],
            ]} />
            <p className="m-0">
              Pitch+ is only shown once a pitch has some results behind it: 15 pitches against live hitters. Before that
              there's nothing for the results side to say, and a Pitch+ would just be Stuff and Location under another name.
            </p>
            <p className="m-0">
              Charted contact sits just under measured, at 440. One in five charted calls being a category off (the rate
              TrackMan operators' hand-tagged hit types disagreed with launch angle) costs surprisingly little. Outcomes alone
              carry far more luck, so they need more than twice the sample to mean the same thing.
            </p>
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
            short="Targets come from the Stuff model, checked against what actually happened: PRISM tries realistic versions of the pitch and keeps a change only when MLB pitches like his back it up."
            specs={[
              ["Compared with", "the 15% of MLB pitcher-season pitch types of the same kind and subtype most like his (60–150): fastballs by ride, run, spin and release height; other pitches by speed gap, ride and run off the fastball, plus the fastball they come off. Deadzone four-seams and firm changeups: all of their kind"],
              ["Four-seam zones", "ride and run targets stay inside his fastball's zone, except deadzone"],
              ["Range searched", "each trait moved on its own, both ways, across the 10th–90th percentile of those pitches"],
              ["Rare shapes", "ride or run beyond 97% of MLB at either end isn't targeted: the model has too little like it to learn from"],
              ["Evidence check", "a ride, run or spin target stands only if, among those pitches, the half with more (or less) of it actually allowed fewer runs"],
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
              "Realistic" means what the MLB pitches most like his actually do, not every pitch with the same label. A
              changeup with 18 inches of run off a high-run sinker is compared with changeups off high-run fastballs, and
              searched in both directions. Where a pitch is more extreme than nearly anything in MLB, the model is guessing,
              so that trait gets no target.
            </p>
            <p className="m-0">
              Every movement and spin target is then checked against real results: among those same MLB pitches, did the
              ones with more ride (or run, or spin) actually allow fewer runs? If the model and the results disagree, there's
              no target. The card quotes the result when there is one. For secondary pitches, velocity is shown as a speed
              range off his current fastball, so it's clear the range rises as the fastball gets harder.
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
              ["Pitch+ and verdict", "15+ pitches against hitters"],
              ["Outing trends", "5+ of a pitch type in that outing"],
              ["Location weight", "his command n ÷ (n + 100) from D3 average; each pitch type n ÷ (n + 430) from his command"],
              ["Results weight in Pitch+", "n ÷ (n + 400) measured · 440 charted · 1,000 outcomes only"],
            ]}>
            <p className="m-0">
              Below these, the page shows N/A or says what's missing. Above them, location and results are still weighed by
              how much data is behind them, so grades firm up as the season goes on instead of jumping around. Stuff+ and
              Shape+ don't need this: a pitch's shape is measured precisely on every throw, so a handful of pitches already
              describes it well. Location+ is shown already weighed by its sample. Results+ is shown as it happened and
              weighed only inside Pitch+.
            </p>
          </Section>

          <Section id="limits" title="Limits and what's next"
            short="PRISM is honest about what it can't see yet.">
            <ul className="m-0 flex list-disc flex-col gap-2 pl-5 marker:text-[#5F6876]">
              <li>The Division III location average is an estimate: Synergy has D3 take rates but no locations, so it's corrected with NECBL TrackMan data. D3 TrackMan games would measure it directly.</li>
              <li>Rapsodo's 6 inches of plate-location error comes from one comparison (Oberlin vs NECBL at the same strike rate). A same-day session on both devices would pin it down, and would confirm or rule out the movement offsets (less horizontal break, less fastball ride on Rapsodo) that are recorded but not applied.</li>
              <li>Oberlin's Results+ is outcomes only until batted balls are charted; it counts for less until then.</li>
              <li>Fall samples are small. Within a pitcher, differences in command between pitch types weren't detectable yet, so each pitch type leans on his overall command.</li>
              <li>Job rates other than strikes and whiffs are compared with his own staff, because Synergy doesn't have them for D3.</li>
              <li>Next: a charting tool for live sessions and games, a same-day Rapsodo and TrackMan comparison, D3 TrackMan data, and secure logins so pages update automatically.</li>
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
