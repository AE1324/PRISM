/**
 * Shape of the staff file written by `python -m prism_model.export_staff`.
 * One PitcherReport per pitcher; fields mirror prism_model/arsenal.py's report dict.
 * Optional fields are optional because small samples leave them empty.
 */

export type Grade = number | "N/A" | null;

export interface Grades {
  pitch_plus: Grade;
  stuff_plus: Grade;
  shape_plus: Grade;
  location_plus: Grade;
  results_plus: Grade;
  compared_to?: string;
}

export interface TargetRow {
  trait: string;
  label: string;
  unit: string;
  now: number;
  lo: number;
  hi: number;
  gain: number;
  best_delta: number;
  in_range: boolean;
  target_text?: string;
  hide?: boolean;
  vs_fb?: boolean;
  fb_velo?: number;
}

export interface Targets {
  rows: TargetRow[];
  now_plus: number;
  pool: string;
  plan: string;
}

export interface SpinAxis {
  normal?: string;
  spread_deg?: number;
  n?: number;
  call?: boolean;
  good_word?: string;
  bad_word?: string;
  why?: string;
  good?: { clock: string; share: number; ivb: number; hb: number; velo: number; total: number } | null;
  bad?: { clock: string; share: number; ivb: number; hb: number; velo: number; total: number } | null;
}

export type Rates = Partial<Record<"strike" | "called_strike" | "zone" | "whiff" | "chase" | "hard" | "gb" | "putaway", number | null>>;

export interface Pitch {
  pitch: string;
  subtype?: string | null;
  bucket?: string;
  n: number;
  game_pitches: number;
  usage_pct: number;
  grades: Grades;
  results_weight?: number;
  results_tier?: "measured" | "charted" | "outcome";
  shape?: { velo: number; ivb: number; hb_arm: number; spin: number; velo_diff?: number | null };
  stuff_parts?: Record<string, number>;
  comparable_mlb?: { text?: string } | null;
  verdict: string;
  so: string;
  jobs?: { job: string; does: string; evidence: string }[];
  deal_breakers?: { issue: string; evidence: string }[];
  great?: string[];
  not_strength?: string[];
  rates?: Rates;
  rates_ref?: Rates;
  targets?: Targets | null;
  spin_axis?: SpinAxis | null;
  swing_split?: string | null;
}

export interface CountRow {
  count: string;
  n: number;
  strike_pct: number | null;
  whiff_pct: number | null;
  hard_pct?: number | null;
  most_used: string;
}

export interface Split {
  pa: number;
  avg: number | null;
  obp: number | null;
  slg: number | null;
  k_pct: number | null;
  bb_pct: number | null;
  pitches: number;
  mix: Record<string, number>;
}

export interface ProgressRow {
  outing: string;
  pitch: string;
  n: number;
  velo: number | null;
  ivb: number | null;
  hb: number | null;
  spin: number | null;
  stuff_plus: number | null;
}

export interface PitcherReport {
  pitcher: string;
  pitcher_id: string;
  throws: string;
  team?: string | null;
  pitches: number;
  game_pitches: number;
  outings: number;
  summary: string;
  development: {
    stuff: { stuff_plus: number | null; label: string };
    pitches_together: string[];
    command: { strike_pct: number | null; staff_strike_pct?: number | null; bb_pct: number | null; pa: number; location_plus: number | null; label: string };
    what_makes_him: string[];
    holding_back: string[];
    priority_order: string[];
  };
  pitch_by_pitch: Pitch[];
  situations: {
    game_pitches: number;
    first_pitch_strike?: { pct: number | null; n: number; pro_range: string } | null;
    by_count: CountRow[];
    splits?: Record<string, Split>;
    fade?: { early_n: number; late_n: number; strike_early: number | null; strike_late: number | null; fb_velo_early: number | null; fb_velo_late: number | null } | null;
  };
  fastball_role?: { pitch: string; subtype?: string; velo: number; ivb: number; hb_arm: number; role: string } | null;
  fastball_traits?: {
    pitch: string;
    velo: number;
    extension_ft: number | null;
    perceived_velo: number | null;
    extension_pct_mlb: number | null;
    vaa: number | null;
    vaa_above_expected: number | null;
    vaa_ae_pct_mlb: number | null;
    velo_pct_d3: number | null;
    perceived_pct_d3: number | null;
    show_vaa?: boolean;
  } | null;
  movement_profile: { pitch: string; subtype?: string; n: number; usage_pct: number; velo: number; max_velo: number; ivb: number; hb_arm: number; spin: number; tilt?: string; vaa?: number | null; rel_height?: number; rel_side_arm?: number; extension?: number | null }[];
  progress: ProgressRow[];
  arsenal_plan?: string;
  points: { shape_family: string; InducedVertBreak: number; hb_arm: number; RelSpeed: number }[];
}

export interface StaffFile {
  format: "prism-staff";
  version: number;
  team: string;
  label: string;
  generated: string;
  pitchers: PitcherReport[];
}
