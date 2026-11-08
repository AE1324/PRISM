import { Link } from "react-router-dom";
import { PrismLogo, SiteNav, SitePage } from "@/features/site/site-shell";

/** Landing page: logo, name, one short description, and the way in. */
export default function HomePage() {
  return (
    <SitePage>
      <SiteNav />

      <main className="flex flex-1 items-center justify-center px-6 pt-10 pb-24">
        <div className="flex w-full max-w-[640px] flex-col items-center text-center">
          <PrismLogo className="mb-9 block h-auto w-[300px] max-w-full" />

          <h1 className="m-0 mb-2.5 pl-[0.14em] font-['Space_Grotesk',system-ui,sans-serif] text-[64px] font-semibold leading-none tracking-[0.14em]">
            PRISM
          </h1>
          <p className="m-0 mb-9 text-sm tracking-[0.06em] text-[#9AA3B1]">
            Pitch Rating Integrated Stuff Model
          </p>
          <p className="m-0 mb-9 max-w-[520px] text-lg leading-relaxed text-[#C2C8D1]">
            A pitch grading system built for coaches and pitchers. PRISM turns
            pitch-tracking data into clear grades for every pitch, and shows
            what&apos;s helping or holding each one back.
          </p>

          <div className="flex flex-wrap justify-center gap-3">
            <Link
              to="/workspace"
              className="rounded-lg bg-[#EDF0F4] px-6 py-[13px] text-[15px] font-semibold text-[#0D1015] no-underline transition-colors hover:bg-white"
            >
              Open workspace
            </Link>
            <Link
              to="/measures"
              className="rounded-lg border border-[rgba(237,240,244,0.22)] px-6 py-3 text-[15px] font-medium text-[#EDF0F4] no-underline transition-colors hover:border-[rgba(237,240,244,0.45)]"
            >
              How it works
            </Link>
          </div>
        </div>
      </main>

      <footer className="p-[22px] text-center text-[13px] text-[#6F7886]">Built for Oberlin Baseball</footer>
    </SitePage>
  );
}
