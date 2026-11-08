import type { ReactNode } from "react";
import { Link, NavLink } from "react-router-dom";

/**
 * Shared pieces for PRISM's public pages (home, what it measures,
 * methodology). These pages use their own dark "spectrum" look, separate
 * from the workspace's theme tokens.
 */

/** The five grade components. Each keeps its color everywhere in PRISM. */
export const FACTORS = [
  { key: "velocity", name: "Velocity", color: "#E8786A" },
  { key: "movement", name: "Movement", color: "#E9A94F" },
  { key: "spin", name: "Spin", color: "#6CC59A" },
  { key: "release", name: "Release", color: "#58A9D8" },
  { key: "command", name: "Command", color: "#9A8CE8" },
] as const;

export const SITE = {
  bg: "#0D1015",
  surface: "#151A21",
  ink: "#EDF0F4",
  soft: "#9AA3B1",
  body: "#C2C8D1",
  line: "rgba(237,240,244,0.08)",
};

/** Large logo: a beam enters the prism and splits into the five colors. */
export function PrismLogo({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 320 150"
      role="img"
      aria-label="PRISM logo: a beam of light enters a prism and splits into five colors"
      className={className}
    >
      <line x1="0" y1="86" x2="122" y2="86" stroke={SITE.ink} strokeWidth="2" strokeLinecap="round" />
      <polygon points="96,136 150,14 204,136" fill="rgba(237,240,244,0.03)" stroke={SITE.ink} strokeWidth="2" strokeLinejoin="round" />
      <line x1="122" y1="86" x2="178" y2="86" stroke="rgba(237,240,244,0.28)" strokeWidth="1.5" />
      {FACTORS.map((f, i) => (
        <line key={f.key} x1="178" y1="86" x2="316" y2={34 + i * 26} stroke={f.color} strokeWidth="1.6" strokeLinecap="round" />
      ))}
    </svg>
  );
}

/** Small logo for the nav bar. */
export function PrismMark() {
  return (
    <svg width="34" height="20" viewBox="0 0 34 20" aria-hidden="true">
      <polygon points="2,18 9,2 16,18" fill="none" stroke={SITE.ink} strokeWidth="1.5" strokeLinejoin="round" />
      {FACTORS.map((f, i) => (
        <line key={f.key} x1="13" y1="11" x2="32" y2={3 + i * 4} stroke={f.color} strokeWidth="1.4" />
      ))}
    </svg>
  );
}

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  `no-underline transition-colors hover:text-[#EDF0F4] ${isActive ? "text-[#EDF0F4] font-medium" : "text-[#9AA3B1]"}`;

const outlineButton =
  "rounded-lg border border-[rgba(237,240,244,0.22)] px-4 py-[9px] font-medium text-[#EDF0F4] no-underline transition-colors hover:border-[rgba(237,240,244,0.45)]";

/** Nav bar. `showLogo` is off on the home page, where the big logo leads. */
export function SiteNav({ showLogo = true }: { showLogo?: boolean }) {
  return (
    <header style={{ borderBottom: showLogo ? `1px solid ${SITE.line}` : "none" }}>
      <nav
        className={`mx-auto flex max-w-[1200px] flex-wrap items-center gap-4 px-6 ${
          showLogo ? "justify-between py-[18px]" : "justify-end py-[22px]"
        }`}
      >
        {showLogo && (
          <Link to="/" className="flex items-center gap-2.5 text-[#EDF0F4] no-underline">
            <PrismMark />
            <span className="font-['Space_Grotesk',system-ui,sans-serif] text-lg font-semibold tracking-[0.12em]">PRISM</span>
          </Link>
        )}
        <div className="flex flex-wrap items-center gap-7 text-sm">
          <NavLink to="/measures" className={navLinkClass}>
            What it measures
          </NavLink>
          <NavLink to="/methodology" className={navLinkClass}>
            Methodology
          </NavLink>
          <Link to="/workspace" className={outlineButton}>
            Open workspace
          </Link>
        </div>
      </nav>
    </header>
  );
}

export function SiteFooter({ children }: { children?: ReactNode }) {
  return (
    <footer style={{ borderTop: `1px solid ${SITE.line}` }}>
      <div className="mx-auto flex max-w-[1200px] flex-wrap justify-between gap-3 p-6 text-[13px] text-[#6F7886]">
        <span>PRISM · Built for Oberlin Baseball</span>
        {children}
      </div>
    </footer>
  );
}

/** Full-height dark page wrapper shared by the public pages. */
export function SitePage({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col font-['Inter',system-ui,sans-serif]" style={{ background: SITE.bg, color: SITE.ink }}>
      {children}
    </div>
  );
}
