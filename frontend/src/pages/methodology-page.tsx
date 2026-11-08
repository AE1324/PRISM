import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { SiteFooter, SiteNav, SitePage } from "@/features/site/site-shell";

const SECTIONS = [
  { id: "shape", title: "Pitch classification" },
  { id: "stuff", title: "Stuff+ and Shape+" },
  { id: "scales", title: "Grading scales" },
  { id: "results", title: "Results and Pitch+" },
  { id: "samples", title: "Sample sizes" },
  { id: "data", title: "Data" },
];

function Section({ id, title, last, children }: { id: string; title: string; last?: boolean; children: ReactNode }) {
  return (
    <section
      id={id}
      className={`scroll-mt-6 ${last ? "" : "mb-11 border-b border-[rgba(237,240,244,0.08)] pb-11"}`}
    >
      <h2 className="m-0 mb-3.5 font-['Space_Grotesk',system-ui,sans-serif] text-2xl font-semibold">{title}</h2>
      <div className="flex flex-col gap-4 text-base leading-[1.7] text-[#C2C8D1]">{children}</div>
    </section>
  );
}

export default function MethodologyPage() {
  const jump = (id: string) => () => document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });

  return (
    <SitePage>
      <SiteNav />

      <div className="mx-auto flex w-full max-w-[1200px] flex-wrap gap-x-20 gap-y-12 px-6 pt-20 pb-24">
        <aside className="min-w-0 flex-[0_1_240px]">
          <h1 className="m-0 mb-4 font-['Space_Grotesk',system-ui,sans-serif] text-[40px] font-semibold leading-[1.1]">Methodology</h1>
          <p className="m-0 mb-7 text-[15px] leading-relaxed text-[#9AA3B1]">How PRISM turns tracking data into grades.</p>
          <nav aria-label="Sections" className="flex flex-col items-start gap-2.5 text-sm">
            {SECTIONS.map((s) => (
              <button
                key={s.id}
                type="button"
                onClick={jump(s.id)}
                className="cursor-pointer border-0 bg-transparent p-0 text-left text-[#B3BAC5] transition-colors hover:text-[#EDF0F4]"
              >
                {s.title}
              </button>
            ))}
          </nav>
        </aside>

        <article className="min-w-0 max-w-[720px] flex-[1_1_560px]">
          <Section id="shape" title="Pitch classification">
            <p className="m-0">
              Device pitch tags are not used. Every pitch is sorted into a shape family, such as riding fastball,
              sinker, gyro slider, sweeper, curveball, or changeup, based on its velocity and movement relative to the
              pitcher&apos;s own fastball. Grades are always calculated within a shape family, so a sweeper is compared to
              other sweepers.
            </p>
          </Section>

          <Section id="stuff" title="Stuff+ and Shape+">
            <p className="m-0">
              Stuff+ predicts the run value a pitch is expected to produce, based only on its physical traits. Balls,
              strikes, and whiffs keep their actual value; balls in play are valued by expected outcome rather than by
              whether they happened to fall for hits.
            </p>
            <p className="m-0">
              Separate models handle fastballs, breaking balls, and offspeed pitches. Velocity is constrained so that a
              harder pitch never grades worse than the same pitch thrown softer.
            </p>
            <p className="m-0">
              Shape+ is the same grade with velocity removed. It separates how well a pitch is designed from how hard
              it is thrown.
            </p>
          </Section>

          <Section id="scales" title="Grading scales">
            <p className="m-0">
              Every grade is centered at 100, and every 10 points is one standard deviation. Grades can be viewed on
              two scales.
            </p>
            <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-3">
              <div className="rounded-xl border border-[rgba(237,240,244,0.08)] bg-[#151A21] p-5">
                <p className="m-0 mb-1.5 text-[15px] font-medium text-[#EDF0F4]">Staff scale</p>
                <p className="m-0 text-sm leading-relaxed text-[#9AA3B1]">100 is the average for your own pitching staff.</p>
              </div>
              <div className="rounded-xl border border-[rgba(237,240,244,0.08)] bg-[#151A21] p-5">
                <p className="m-0 mb-1.5 text-[15px] font-medium text-[#EDF0F4]">Division III scale</p>
                <p className="m-0 text-sm leading-relaxed text-[#9AA3B1]">100 is the average across Division III pitchers.</p>
              </div>
            </div>
          </Section>

          <Section id="results" title="Results and Pitch+">
            <p className="m-0">
              Pitch+ combines three parts: Stuff (how the pitch is built), Location (how well it is commanded), and
              Results (what it does in games). Results carry more weight as the game sample grows.
            </p>
            <p className="m-0">A pitch that is working in games is never told to change. It can get add-ons to try, not fixes.</p>
          </Section>

          <Section id="samples" title="Sample sizes">
            <p className="m-0">
              Below a minimum pitch count, PRISM shows N/A and the reason instead of a grade. Between the minimum and a
              full sample, grades are marked as an early read.
            </p>
          </Section>

          <Section id="data" title="Data" last>
            <p className="m-0">
              PRISM is built on TrackMan&apos;s data format. Rapsodo sessions are converted to it, so every source is
              graded the same way. The models learn from large pitch-tracking datasets, then are calibrated to Division
              III.
            </p>
          </Section>
        </article>
      </div>

      <SiteFooter>
        <Link to="/" className="text-[#9AA3B1] hover:text-[#EDF0F4]">
          Back to home
        </Link>
      </SiteFooter>
    </SitePage>
  );
}
