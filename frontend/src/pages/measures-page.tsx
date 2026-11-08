import { Link } from "react-router-dom";
import { FACTORS, SiteFooter, SiteNav, SitePage } from "@/features/site/site-shell";

const DETAILS: Record<string, { desc: string; inputs: string; example: number }> = {
  velocity: {
    desc: "How fast the pitch gets to the hitter. PRISM uses perceived velocity, which adjusts for extension, since a pitch released closer to the plate plays faster than the radar reading.",
    inputs: "release speed, extension",
    example: -2,
  },
  movement: {
    desc: "How the pitch breaks and how far it separates from the fastball. Secondary pitches are judged on their difference from the pitcher’s own fastball, not on raw break alone.",
    inputs: "induced vertical break, horizontal break, velo and break differentials",
    example: 7,
  },
  spin: {
    desc: "How much the ball spins and how efficiently that spin turns into movement.",
    inputs: "spin rate, spin efficiency, spin axis",
    example: 3,
  },
  release: {
    desc: "Where the ball comes out and the angle it arrives at the plate. A flatter approach angle helps a fastball play up, and a secondary released far from the fastball is easier to pick up.",
    inputs: "release height, release side, vertical approach angle",
    example: 1,
  },
  command: {
    desc: "Whether the pitch lands where it needs to. Command is graded from game and live-at-bat data, separately from the physical traits above.",
    inputs: "strike rate, competitive rate, middle-middle rate",
    example: -5,
  },
};

const signed = (v: number) => (v > 0 ? `+${v}` : v < 0 ? `−${Math.abs(v)}` : "0");

/** Diverging bar: negative values grow left of center, positive values right. */
function ContributionBar({ value, color }: { value: number; color: string }) {
  const width = `${Math.min(100, Math.abs(value) * 10)}%`;
  return (
    <div className="flex h-2.5 rounded-[5px] bg-[rgba(237,240,244,0.06)]">
      <div className="flex flex-1 justify-end border-r border-[rgba(237,240,244,0.35)]">
        {value < 0 && <div className="h-2.5 rounded-[5px]" style={{ width, background: color }} />}
      </div>
      <div className="flex flex-1 justify-start">
        {value > 0 && <div className="h-2.5 rounded-[5px]" style={{ width, background: color }} />}
      </div>
    </div>
  );
}

export default function MeasuresPage() {
  const total = FACTORS.reduce((sum, f) => sum + DETAILS[f.key].example, 0);

  return (
    <SitePage>
      <SiteNav />

      <section className="mx-auto flex max-w-[1200px] flex-wrap items-center gap-14 px-6 pt-20 pb-16">
        <div className="min-w-0 flex-[1_1_420px]">
          <h1 className="m-0 mb-5 font-['Space_Grotesk',system-ui,sans-serif] text-5xl font-semibold leading-[1.08] tracking-[-0.01em]">
            One grade, five components.
          </h1>
          <p className="m-0 mb-4 text-[17px] leading-[1.65] text-[#B3BAC5]">
            A prism splits white light into the colors that make it up. PRISM does the same with a pitch grade: it
            shows how many points each part of the pitch adds or takes away.
          </p>
          <p className="m-0 text-[17px] leading-[1.65] text-[#B3BAC5]">
            Each component keeps its color on every page, for every pitcher.
          </p>
        </div>

        <div className="min-w-0 flex-[1_1_440px] rounded-2xl border border-[rgba(237,240,244,0.1)] bg-[#151A21] p-7">
          <div className="mb-[26px] flex flex-wrap items-end justify-between gap-3">
            <div>
              <p className="m-0 mb-1 font-['Space_Grotesk',system-ui,sans-serif] text-[22px] font-semibold">Slider</p>
              <p className="m-0 text-sm text-[#9AA3B1]">Example pitch · 74.5 mph</p>
            </div>
            <div className="text-right">
              <p className="m-0 text-[13px] text-[#9AA3B1]">Stuff+</p>
              <p className="m-0 font-['JetBrains_Mono',ui-monospace,monospace] text-[40px] font-medium leading-none">{100 + total}</p>
            </div>
          </div>
          <div className="flex flex-col gap-3.5">
            {FACTORS.map((f) => {
              const v = DETAILS[f.key].example;
              return (
                <div key={f.key} className="grid grid-cols-[92px_minmax(0,1fr)_36px] items-center gap-3.5">
                  <span className="text-sm">{f.name}</span>
                  <ContributionBar value={v} color={f.color} />
                  <span className="text-right font-['JetBrains_Mono',ui-monospace,monospace] text-sm" style={{ color: f.color }}>
                    {signed(v)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      <section className="mx-auto w-full max-w-[1200px] px-6 pb-24">
        <div className="border-t border-[rgba(237,240,244,0.08)]">
          {FACTORS.map((f) => (
            <div
              key={f.key}
              className="flex flex-wrap gap-x-14 gap-y-6 border-b border-[rgba(237,240,244,0.08)] py-9"
            >
              <div className="flex min-w-0 flex-[0_1_260px] items-center gap-3.5">
                <span className="h-9 w-1 rounded-sm" style={{ background: f.color }} />
                <h2 className="m-0 font-['Space_Grotesk',system-ui,sans-serif] text-[26px] font-semibold">{f.name}</h2>
              </div>
              <div className="min-w-0 flex-[1_1_360px]">
                <p className="m-0 mb-3.5 text-base leading-[1.65] text-[#C2C8D1]">{DETAILS[f.key].desc}</p>
                <p className="m-0 text-sm text-[#8A93A1]">
                  Inputs: <span className="font-['JetBrains_Mono',ui-monospace,monospace] text-[#B3BAC5]">{DETAILS[f.key].inputs}</span>
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <SiteFooter>
        <Link to="/methodology" className="text-[#9AA3B1] hover:text-[#EDF0F4]">
          Next: Methodology
        </Link>
      </SiteFooter>
    </SitePage>
  );
}
