import type { Grade, PitcherReport, StaffFile } from "./types";

/**
 * The staff file lives only in this browser (localStorage), never on the server: the site and repo are
 * public, so player data must not be. It survives refreshes and restarts on this device until cleared.
 */
const KEY = "prism:staff";

export function saveStaff(staff: StaffFile): boolean {
  try {
    localStorage.setItem(KEY, JSON.stringify(staff));
    return true;
  } catch {
    return false; // private mode or storage full: the page still works for this visit
  }
}

export function loadStaff(): StaffFile | null {
  try {
    const raw = localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as StaffFile) : null;
  } catch {
    return null;
  }
}

export function clearStaff(): void {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* ignore */
  }
}

/** Read and check a staff file the user picked. Throws a readable message when it isn't one. */
export async function parseStaffFile(file: File): Promise<StaffFile> {
  let data: unknown;
  try {
    data = JSON.parse(await file.text());
  } catch {
    throw new Error("That file isn't valid JSON. Export it with: python -m prism_model.export_staff");
  }
  const s = data as Partial<StaffFile>;
  if (s?.format !== "prism-staff" || !Array.isArray(s.pitchers)) {
    throw new Error("That isn't a PRISM staff file. Export one with: python -m prism_model.export_staff");
  }
  if (!s.pitchers.length) throw new Error("The staff file has no pitchers in it.");
  return s as StaffFile;
}

export const slug = (name: string) =>
  name.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

export const isNum = (v: Grade | number | undefined | null): v is number => typeof v === "number" && Number.isFinite(v);

/** Best graded Pitch+ in a pitcher's arsenal (for the roster). */
export function bestPitch(p: PitcherReport): { pitch: string; value: number } | null {
  let best: { pitch: string; value: number } | null = null;
  for (const x of p.pitch_by_pitch) {
    const v = x.grades.pitch_plus;
    if (isNum(v) && (!best || v > best.value)) best = { pitch: x.pitch, value: v };
  }
  return best;
}
