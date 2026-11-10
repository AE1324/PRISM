import type { Grade } from "./types";

/**
 * Pitch colors: Savant's familiar hues (four-seam red, slider yellow, curveball cyan, changeup green...),
 * softened a step so they sit comfortably on the dark page instead of glaring.
 */
const PITCH_COLOR: Record<string, string> = {
  "Four-seam": "#E0596B",
  Sinker: "#EDA24A",
  Cutter: "#C97452",
  Slider: "#D9C75B",
  Curveball: "#4FBFD6",
  Changeup: "#5DBE76",
  Splitter: "#57AFAB",
  Knuckleball: "#A39A5A",
};
export const pitchColor = (p: string) => PITCH_COLOR[p] ?? "#C3C9D2";

/**
 * Baseball Savant grade shading for a 100-centered grade: Savant blue below average, Savant red above,
 * a neutral cell at 100, full strength at 100 +- 25. Returns [background, text].
 */
export function savant(v: Grade | number | undefined): [string, string] {
  if (typeof v !== "number") return ["transparent", "var(--pr-soft)"];
  const t = Math.max(-1, Math.min(1, (v - 100) / 25));
  const mid = [46, 51, 59];
  const end = t > 0 ? [210, 45, 73] : [50, 90, 165];
  const k = Math.pow(Math.abs(t), 0.8); // reach strong color a little sooner, so 110 / 90 already read clearly
  const c = mid.map((m, i) => Math.round(m + k * (end[i] - m)));
  return [`rgb(${c.join(",")})`, "#FFFFFF"];
}

/** Position (0-1) of a 100-centered grade on a 70-130 bar. */
export const gradePos = (v: number) => Math.max(0, Math.min(1, (v - 70) / 60));

export const fmtGrade = (v: Grade | undefined) => (typeof v === "number" ? String(Math.round(v)) : "–");
export const pct = (v: number | null | undefined, digits = 0) => (typeof v === "number" ? `${v.toFixed(digits)}%` : "–");
export const num = (v: number | null | undefined, digits = 1) => (typeof v === "number" ? v.toFixed(digits) : "–");
export const signed = (v: number, digits = 1) => `${v > 0 ? "+" : ""}${v.toFixed(digits)}`;
export const ordinal = (n: number | null | undefined) => {
  if (typeof n !== "number") return "–";
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return `${n}${s[(v - 20) % 10] || s[v] || s[0]}`;
};
export const sentence = (s: string) => (s ? s.charAt(0).toUpperCase() + s.slice(1).toLowerCase() : s);

/** Status color for a pitch verdict ("Working", "Struggling", "No clear job yet", ...). */
export function verdictTone(v: string): string {
  if (/^working/i.test(v)) return "var(--pr-good)";
  if (/strugg|fail|hit hard/i.test(v)) return "var(--pr-bad)";
  return "var(--pr-warn)";
}
