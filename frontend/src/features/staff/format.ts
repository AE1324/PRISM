import type { Grade } from "./types";

/** Muted pitch-type colors: distinguishable on the dark ground without being loud. */
const PITCH_COLOR: Record<string, string> = {
  "Four-seam": "#D27F72",
  Sinker: "#CFA164",
  Cutter: "#B48BC4",
  Slider: "#6E9FC9",
  Curveball: "#8E86C9",
  Changeup: "#7DB596",
  Splitter: "#6DB3AE",
  Knuckleball: "#A7A08F",
};
export const pitchColor = (p: string) => PITCH_COLOR[p] ?? "#B9C0CA";

/**
 * Baseball Savant style shading for a 100-centered grade: blue below average, red above, full strength
 * at 100 +- 25. Muted so it reads on the dark UI. Returns [background, text].
 */
export function savant(v: Grade | number | undefined): [string, string] {
  if (typeof v !== "number") return ["transparent", "var(--pr-soft)"];
  const t = Math.max(-1, Math.min(1, (v - 100) / 25));
  const mid = [24, 29, 36];
  const end = t > 0 ? [150, 52, 64] : [44, 74, 128];
  const c = mid.map((m, i) => Math.round(m + Math.abs(t) * (end[i] - m)));
  return [`rgb(${c.join(",")})`, Math.abs(t) > 0.15 ? "#F2F4F7" : "var(--pr-body)"];
}

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
