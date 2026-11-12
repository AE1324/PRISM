import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { FACTORS, SITE, SiteFooter, SiteNav, SitePage } from "@/features/site/site-shell";

const DISPLAY = "font-['Space_Grotesk',system-ui,sans-serif]";
const MONO = "font-['IBM_Plex_Mono',ui-monospace,monospace]";
const LINE = "border-[rgba(237,240,244,0.08)]";

/** Savant-style grade shading used across PRISM (100 = average; red above, blue below). */
function gradeBg(v: number) {
  const t = Math.max(-1, Math.min(1, (v - 100) / 25));
  const mid = [46, 51, 59], end = t > 0 ? [210, 45, 73] : [50, 90, 165];
  const k = Math.pow(Math.abs(t), 0.8);
  return `rgb(${mid.map((m, i) => Math.round(m + k * (end[i] - m))).join(",")})`;
}

function Grade({ v, size = "md" }: { v: number; size?: "md" | "lg" }) {
  return (
    <span
      className={`inline-flex items-center justify-center rounded-lg font-semibold text-white ${DISPLAY} ${size === "lg" ? "h-14 min-w-[84px] px-3 text-[30px]" : "h-8 min-w-[52px] px-2 text-base"}`}
      style={{ background: gradeBg(v) }}
    >
      {v}
    </span>
  );
}

function Eyebrow({ children }: { children: ReactNode }) {
  return <p className="m-0 mb-3 text-xs font-semibold uppercase tracking-[0.14em] text-[#9AA3B1]">{children}</p>;
}

function Block({ id, eyebrow, title, intro, children }: { id: string; eyebrow: string; title: string; intro: ReactNode; children: ReactNode }) {
  return (
    <section id={id} className={`scroll-mt-6 border-t ${LINE} py-16`}>
      <div className="flex flex-wrap gap-x-16 gap-y-8">
        <div className="min-w-0 flex-[0_1_330px]">
          <Eyebrow>{eyebrow}</Eyebrow>
          <h2 className={`m-0 mb-4 text-[32px] font-semibold leading-[1.12] tracking-[-0.01em] ${DISPLAY}`}>{title}</h2>
          <div className="text-[16px] leading-[1.7] text-[#B3BAC5]">{intro}</div>
        </div>
        <div className="min-w-0 flex-[1_1_520px]">{children}</div>
      </div>
    </section>
  );
}

/* ---------- the grades ---------- */
const GRADES = [
  { name: "Stuff+", q: "How good is the pitch itself?", body: "Velocity, movement, spin and release, graded on how many runs pitches built like this one give up. Nothing about where it was thrown or what happened.", from: "Pitch tracking only" },
  { name: "Shape+", q: "How good is the design, ignoring speed?", body: "Stuff+ with velocity taken out. It separates a well-shaped pitch from one that's carried by arm speed, and shows what a pitch could become as velocity comes.", from: "Pitch tracking only" },
  { name: "Location+", q: "Does he put it where it needs to go?", body: "What his spots are worth in runs for that pitch type and count, from where MLB pitches do damage and where they get hit, judged against the D3 average. It's weighed by how much data is behind it.", from: "Plate location" },
  { name: "Results+", q: "What has it actually done in games?", body: "The run value of every pitch thrown against hitters: balls, strikes, whiffs, and the type and quality of contact when the ball was put in play, judged against that level.", from: "Live at-bats" },
];

/* ---------- jobs ---------- */
const JOBS = [
  ["Strike pitch", "Lands strikes at a clearly higher rate"],
  ["Whiff pitch", "Gets swings and misses"],
  ["Called-strike pitch", "Steals strikes without a swing"],
  ["Chase pitch", "Gets hitters to swing out of the zone"],
  ["Weak-contact pitch", "Limits hard contact"],
  ["Ground-ball pitch", "Keeps the ball on the ground"],
  ["Putaway pitch", "Finishes hitters with two strikes"],
];

const VERDICTS = [
  { label: "Working", color: "#3CCF7A", text: "It has a clear job and nothing is going wrong. It's never told to change: it gets add-ons to try, not fixes." },
  { label: "Struggling", color: "#F0566A", text: "It has a deal-breaker: it isn't landing strikes or it's getting hit hard. The card says which and by how much." },
  { label: "No clear job yet", color: "#F2B33D", text: "Nothing it does is clearly better than average yet. The card shows the shape changes that would give it one." },
  { label: "Too few game pitches", color: "#F2B33D", text: "Fewer than 15 pitches against hitters. It's graded on its traits, but PRISM won't call it working or struggling yet." },
];

const DETAILS: Record<string, { desc: string; inputs: string }> = {
  velocity: { desc: "How fast the pitch gets to the hitter. PRISM uses perceived velocity: a pitcher who releases the ball a foot closer to the plate gets about 1.6 mph of extra effective speed.", inputs: "release speed, extension" },
  movement: { desc: "How the ball breaks, and for every pitch that isn't a fastball, how far it separates from his own fastball in speed and movement. A changeup is judged on the gap it creates, not on raw break alone.", inputs: "induced vertical break, horizontal break, velo / IVB / HB off the fastball" },
  spin: { desc: "How hard the ball spins and which way. Spin direction is shown as a clock face, and for some pitch types the research shows which direction plays better.", inputs: "spin rate, spin axis" },
  release: { desc: "Where the ball comes out of the hand. For fastballs, release height and width change how the pitch arrives at the plate and how it plays.", inputs: "release height, release side" },
  command: { desc: "Whether the pitch lands where it needs to. Command is its own grade (Location+), kept separate from the physical traits above so a nasty pitch that isn't commanded yet still shows as nasty.", inputs: "plate location, count, pitch type, hitter side" },
};

export default function MeasuresPage() {
  const example = { stuff: 121, loc: 95, res: 122, n: 27 };
  const resW = example.n / (example.n + 400); // measured contact (TrackMan)
  const pp = Math.round((example.stuff + example.loc) / 2 + resW * (example.res - (example.stuff + example.loc) / 2));

  return (
    <SitePage>
      <SiteNav />

      {/* ---------- hero ---------- */}
      <section className="mx-auto flex w-full max-w-[1200px] flex-wrap items-center gap-14 px-6 pt-20 pb-16">
        <div className="min-w-0 flex-[1_1_440px]">
          <Eyebrow>What it measures</Eyebrow>
          <h1 className={`m-0 mb-5 text-5xl font-semibold leading-[1.06] tracking-[-0.015em] ${DISPLAY}`}>
            Every pitch, split into the parts that make it work.
          </h1>
          <p className="m-0 mb-4 text-[17px] leading-[1.7] text-[#B3BAC5]">
            A prism splits white light into its colors. PRISM does the same with a pitch: it pulls apart how the pitch is
            built, where it's thrown and what it does in games, grades each one, and tells the pitcher what to work on next.
          </p>
          <p className="m-0 text-[17px] leading-[1.7] text-[#B3BAC5]">
            Every grade is on the same scale: <span className="text-[#EDF0F4]">100 is average</span> for the level he's
            pitching at, and every 10 points is one standard deviation.
          </p>
        </div>

        <div className="min-w-0 flex-[1_1_440px] rounded-2xl border border-[rgba(237,240,244,0.1)] p-7" style={{ background: SITE.surface }}>
          <div className="mb-6 flex items-start justify-between gap-4">
            <div>
              <p className={`m-0 mb-1 text-[22px] font-semibold ${DISPLAY}`}>Slider · Sweeper</p>
              <p className="m-0 text-sm text-[#9AA3B1]">Example pitch · 77 mph · {example.n} pitches vs hitters</p>
            </div>
            <div className="text-right">
              <p className="m-0 mb-1 text-[11px] font-semibold uppercase tracking-[0.14em] text-[#9AA3B1]">Pitch+</p>
              <Grade v={pp} size="lg" />
            </div>
          </div>
          <div className="flex flex-col gap-3">
            {[
              ["Stuff+", example.stuff, "How it's built"],
              ["Location+", example.loc, "Where it's thrown"],
              ["Results+", example.res, `What it's done (${Math.round(resW * 100)}% weight so far)`],
            ].map(([k, v, d]) => (
              <div key={k as string} className={`flex items-center justify-between gap-3 rounded-xl border ${LINE} px-4 py-3`}>
                <div>
                  <p className="m-0 text-[15px] font-medium">{k}</p>
                  <p className="m-0 text-[13px] text-[#9AA3B1]">{d}</p>
                </div>
                <Grade v={v as number} />
              </div>
            ))}
          </div>
          <p className="m-0 mt-5 text-[13px] leading-relaxed text-[#9AA3B1]">
            Pitch+ starts from the average of Stuff+ and Location+. Results only take over once there's enough of a game
            sample to trust them, and Pitch+ isn't shown until a pitch has 15 pitches against hitters.
          </p>
        </div>
      </section>

      <div className="mx-auto w-full max-w-[1200px] px-6 pb-12">
        {/* ---------- the grades ---------- */}
        <Block id="grades" eyebrow="01 · The grades" title="Four questions about every pitch"
          intro={<>Each grade answers one question, from one kind of data. Keeping them apart is what lets PRISM say <i>why</i> a pitch is working or not, instead of handing back a single number.</>}>
          <div className="grid grid-cols-[repeat(auto-fit,minmax(240px,1fr))] gap-3">
            {GRADES.map((g) => (
              <div key={g.name} className={`rounded-2xl border ${LINE} p-5`} style={{ background: SITE.surface }}>
                <p className={`m-0 mb-1 text-xl font-semibold ${DISPLAY}`}>{g.name}</p>
                <p className="m-0 mb-3 text-[15px] text-[#EDF0F4]">{g.q}</p>
                <p className="m-0 mb-4 text-sm leading-relaxed text-[#B3BAC5]">{g.body}</p>
                <p className="m-0 text-xs uppercase tracking-[0.12em] text-[#7D8694]">{g.from}</p>
              </div>
            ))}
          </div>
          <div className={`mt-3 rounded-2xl border ${LINE} p-5`} style={{ background: SITE.surface }}>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className={`m-0 mb-1 text-xl font-semibold ${DISPLAY}`}>Pitch+</p>
                <p className="m-0 text-[15px] text-[#EDF0F4]">The one-number summary.</p>
              </div>
              <code className={`rounded-lg px-3 py-2 text-[13px] text-[#C2C8D1] ${MONO}`} style={{ background: SITE.bg }}>
                Pitch+ = avg(Stuff+, Location+) → Results+ by n ÷ (n + k)
              </code>
            </div>
            <p className="m-0 mt-3 text-sm leading-relaxed text-[#B3BAC5]">
              Pitch+ shows up once a pitch has 15 pitches against live hitters; before that the card shows the other grades
              and says how many it has. With 50 game pitches, results are about 11% of the grade with TrackMan contact data
              and 5% with outcomes only; over a full season of about 250, 38% and 20%. Early on, a pitch is judged mostly on
              what it is, not on a handful of outcomes. Every Results+ is tagged measured, charted or outcomes only, so you
              know how much is behind it.
            </p>
          </div>
        </Block>

        {/* ---------- what goes into stuff ---------- */}
        <Block id="components" eyebrow="02 · Inside the grade" title="What goes into a pitch"
          intro={<>Each component keeps its color across PRISM. On a pitcher's page, the Stuff grade is broken down into these parts so you can see which ones are adding points and which are costing them.</>}>
          <div className={`border-t ${LINE}`}>
            {FACTORS.map((f) => (
              <div key={f.key} className={`flex flex-wrap gap-x-10 gap-y-3 border-b ${LINE} py-6`}>
                <div className="flex min-w-0 flex-[0_1_170px] items-center gap-3">
                  <span className="h-8 w-1 rounded-sm" style={{ background: f.color }} />
                  <h3 className={`m-0 text-[22px] font-semibold ${DISPLAY}`}>{f.key === "command" ? "Command" : f.name}</h3>
                </div>
                <div className="min-w-0 flex-[1_1_320px]">
                  <p className="m-0 mb-2 text-[15px] leading-[1.65] text-[#C2C8D1]">{DETAILS[f.key].desc}</p>
                  <p className="m-0 text-[13px] text-[#8A93A1]">Inputs: <span className={`text-[#B3BAC5] ${MONO}`}>{DETAILS[f.key].inputs}</span></p>
                </div>
              </div>
            ))}
          </div>
        </Block>

        {/* ---------- jobs and verdicts ---------- */}
        <Block id="jobs" eyebrow="03 · Jobs and verdicts" title="Every pitch needs a job"
          intro={<>Grades say how good a pitch is. Jobs say what it's <i>for</i>. A pitch earns a job when it's clearly better than average at one of seven things, measured against Division III pitchers for strikes and whiffs and against his own staff for the rest.</>}>
          <div className="grid grid-cols-[repeat(auto-fit,minmax(230px,1fr))] gap-2.5">
            {JOBS.map(([j, d]) => (
              <div key={j} className={`rounded-xl border ${LINE} px-4 py-3.5`} style={{ background: SITE.surface }}>
                <p className="m-0 text-[15px] font-medium">{j}</p>
                <p className="m-0 text-[13px] text-[#9AA3B1]">{d}</p>
              </div>
            ))}
          </div>
          <p className="m-0 mt-8 mb-3 text-[15px] text-[#EDF0F4]">Each pitch also gets a verdict:</p>
          <div className="flex flex-col gap-2.5">
            {VERDICTS.map((v) => (
              <div key={v.label} className={`flex flex-wrap items-baseline gap-x-4 gap-y-1 rounded-xl border ${LINE} px-4 py-3.5`} style={{ background: SITE.surface }}>
                <span className="w-[190px] flex-none text-xs font-semibold uppercase tracking-[0.1em]" style={{ color: v.color }}>{v.label}</span>
                <span className="min-w-0 flex-[1_1_300px] text-sm leading-relaxed text-[#B3BAC5]">{v.text}</span>
              </div>
            ))}
          </div>
        </Block>

        {/* ---------- fastball ---------- */}
        <Block id="fastball" eyebrow="04 · The fastball" title="Traits that change how a fastball plays"
          intro={<>A fastball's grade comes from the model, but three traits get called out on their own because they explain a lot and can be trained. PRISM only highlights them when they're unusual, good or bad.</>}>
          <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-3">
            {[
              ["Perceived velocity", "Radar speed adjusted for extension. 88 mph released a foot closer to the plate plays like about 89.6.", "Velocity is graded against Division III: his fastball's D3 percentile sets how it grades."],
              ["Extension", "How far down the mound he releases the ball. PRISM compares it to what's typical for his height, so short extension from a tall pitcher reads as mechanical, not size.", "Each half foot gained plays like about 0.8 mph."],
              ["Approach angle", "How flat or steep the fastball arrives, compared to what's expected for his release, speed and where the pitch crossed the plate.", "A flat four-seam up in the zone misses bats; a steep or flat sinker is flagged too."],
            ].map(([t, a, b]) => (
              <div key={t} className={`rounded-2xl border ${LINE} p-5`} style={{ background: SITE.surface }}>
                <p className={`m-0 mb-2 text-lg font-semibold ${DISPLAY}`}>{t}</p>
                <p className="m-0 mb-3 text-sm leading-relaxed text-[#B3BAC5]">{a}</p>
                <p className="m-0 text-[13px] leading-relaxed text-[#9AA3B1]">{b}</p>
              </div>
            ))}
          </div>
        </Block>

        {/* ---------- labels ---------- */}
        <Block id="labels" eyebrow="05 · Pitch labels" title="What kind of pitch it is"
          intro={<>Every pitch gets a label from its average shape. Labels never change a grade. They keep a pitch in its own lane: a cut fastball is compared with cut fastballs and is never asked to become a ride fastball, because a pitcher's natural shape comes from how he releases the ball.</>}>
          <div className="grid grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-3">
            {[
              ["Four-seam", [["Ride", "14+ in ride, 6–12 in run"], ["Cut-ride", "14+ in ride, under 6 in run"], ["Ride-run", "14+ in ride, over 12 in run"], ["Cut", "under 14 in ride, under 6 in run"], ["Run", "under 14 in ride, over 14 in run"], ["Deadzone", "10–14 in of both: the weak spot"]]],
              ["Changeup (off his own fastball)", [["Fade-drop", "6+ in more run and 9+ in less ride"], ["Fade", "6+ in more run"], ["Drop", "9+ in less ride"], ["Firm", "under 6 in less ride: too close to the fastball"], ["Standard", "everything else"]]],
              ["Slider", [["Sweeper", "10+ in glove side"], ["Gyro", "under 5 in glove side"], ["Traditional", "in between"]]],
              ["Curveball", [["Slurve", "sweep about equal to drop"], ["12-6", "under 5 in glove side"], ["Standard", "in between"]]],
            ].map(([pitch, labels]) => (
              <div key={pitch as string} className={`rounded-2xl border ${LINE} p-5`} style={{ background: SITE.surface }}>
                <p className={`m-0 mb-3 text-lg font-semibold ${DISPLAY}`}>{pitch as string}</p>
                <div className="flex flex-col gap-1.5">
                  {(labels as string[][]).map(([l, d]) => (
                    <p key={l} className="m-0 text-sm text-[#B3BAC5]"><span className="text-[#EDF0F4]">{l}</span> · {d}</p>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="m-0 mt-5 text-[15px] leading-relaxed text-[#B3BAC5]">
            The weak shapes are the exception. A deadzone four-seam or a firm changeup is compared with every pitch of its
            kind, so its targets can push it toward whatever grades better. A changeup is labeled by how it separates from
            his own fastball, because that separation is the pitch: the same 18 inches of run is a fade changeup off a
            cutting four-seam and an ordinary one off a high-run sinker.
          </p>
        </Block>

        {/* ---------- reading a card ---------- */}
        <Block id="reading" eyebrow="06 · Reading a pitch card" title="Targets, the spin clock and the colors"
          intro={<>Every pitch card ends in something to do. Targets come from the same model that grades the pitch, are searched only among the MLB pitches most like his, and are kept only when real MLB results back them up.</>}>
          <div className={`overflow-x-auto rounded-2xl border ${LINE}`} style={{ background: SITE.surface }}>
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="text-left text-[11px] uppercase tracking-[0.12em] text-[#9AA3B1]">
                  <th className="px-5 py-3 font-semibold">Trait</th><th className="px-3 py-3 text-right font-semibold">Now</th>
                  <th className="px-3 py-3 text-right font-semibold">Target (vs. his FB)</th><th className="px-5 py-3 text-right font-semibold">Stuff+ gain</th>
                </tr>
              </thead>
              <tbody className={MONO}>
                {[
                  ["Velocity", "73.7 mph", "76.2–78.2 mph", "+16", false],
                  ["IVB (ride +)", "+1.6 in", "on target", "–", true],
                  ["Spin", "2,375 rpm", "2,575 rpm or more", "+7", false],
                ].map(([t, n, g, s, ok]) => (
                  <tr key={t as string} className={`border-t ${LINE}`}>
                    <td className="px-5 py-3 font-['Inter',system-ui,sans-serif]">{t}</td>
                    <td className="px-3 py-3 text-right" style={{ color: ok ? "#3CCF7A" : "#F0566A" }}>{n}</td>
                    <td className="px-3 py-3 text-right text-[#C2C8D1]">{g}</td>
                    <td className="px-5 py-3 text-right">{s}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <ul className="m-0 mt-5 flex list-none flex-col gap-3 p-0 text-[15px] leading-relaxed text-[#B3BAC5]">
            <li><span className="text-[#EDF0F4]">Green means on target, red means there's a gain to be had.</span> A change only becomes a target if it's worth at least 5 Stuff+, and a ride, run or spin target only stands if, among the MLB pitches most like his, the ones with that change actually allowed fewer runs. The card quotes that result.</li>
            <li><span className="text-[#EDF0F4]">"No target (rare)"</span> means the trait is beyond 97% of MLB pitches of that type, so there isn't enough data to say whether more or less would help.</li>
            <li><span className="text-[#EDF0F4]">Velocity targets for secondary pitches are set off his current fastball.</span> As the fastball gets faster, the range moves up with it. PRISM never asks a pitcher to throw slower, except a changeup that sits too close to the fastball, and then only through the grip.</li>
            <li><span className="text-[#EDF0F4]">Spin axis is shown as a clock face</span> (12:00 is pure backspin, mirrored for lefties). When the research shows a direction plays better for that pitch type, the card says which way to go.</li>
            <li><span className="text-[#EDF0F4]">Under the grades,</span> a tag says how the results were recorded (measured, charted or outcomes only) and how much they count toward Pitch+, or how many more pitches against hitters it needs before Pitch+ shows. The card's top line gives the pitch's average and top velocity.</li>
          </ul>
          <div className="mt-8 flex flex-wrap items-center gap-2">
            <span className="mr-2 text-sm text-[#9AA3B1]">Grade colors:</span>
            {[75, 85, 95, 100, 105, 115, 125].map((v) => <Grade key={v} v={v} />)}
            <span className="ml-2 text-sm text-[#9AA3B1]">blue below average · red above</span>
          </div>
        </Block>
      </div>

      <div className="mx-auto w-full max-w-[1200px] px-6 pb-12">
        <Block id="data" eyebrow="07 · The data behind it" title="Where the numbers come from"
          intro={<>PRISM grades Rapsodo and TrackMan data, and knows the difference between them.</>}>
          <div className="grid grid-cols-[repeat(auto-fit,minmax(240px,1fr))] gap-3">
            {[
              ["Sessions", "Every session is tagged game, live or bullpen. Only game and live sessions count toward the grades; bullpens are kept, not graded."],
              ["Devices", "TrackMan measures release, flight, plate location and batted balls. Rapsodo reads spin at release and projects the rest, so its plate locations carry about 6 inches of error, which Location+ accounts for."],
              ["Cleaning", "Misreads are flagged for review, not silently dropped: impossible values, a pitch from the wrong side or the wrong release point, or a speed far off everything else he threw. A TrackMan game whose movement reads off for everyone has its movement set aside."],
            ].map(([t, d]) => (
              <div key={t} className={`rounded-2xl border ${LINE} p-5`} style={{ background: SITE.surface }}>
                <p className={`m-0 mb-2 text-lg font-semibold ${DISPLAY}`}>{t}</p>
                <p className="m-0 text-sm leading-relaxed text-[#B3BAC5]">{d}</p>
              </div>
            ))}
          </div>
        </Block>
      </div>

      <SiteFooter>
        <Link to="/methodology" className="text-[#9AA3B1] hover:text-[#EDF0F4]">Next: how it's built →</Link>
      </SiteFooter>
    </SitePage>
  );
}
