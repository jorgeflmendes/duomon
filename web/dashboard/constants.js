export const OPPONENTS = [
  ["all", "All"],
  ["random", "Random"],
  ["maxpower", "MaxPower"],
  ["simpleheuristics", "Simple"],
  ["abyssal", "Abyssal"],
];

export const OPPONENT_LABELS = Object.fromEntries(OPPONENTS);

export const OPPONENT_COLORS = {
  random: "var(--type-random)",
  maxpower: "var(--type-maxpower)",
  simpleheuristics: "var(--type-simple)",
  abyssal: "var(--type-abyssal)",
};

export const OPPONENT_DESCRIPTIONS = {
  random: "Selects a legal move or switch at random; useful as the baseline sanity check.",
  maxpower: "Prioritizes high base-power attacks, with small bonuses for accuracy and priority.",
  simpleheuristics:
    "Uses poke-env SimpleHeuristics-style scoring with type-aware targeting and double-battle target multipliers.",
  abyssal:
    "Uses expected damage, KO pressure, STAB, accuracy, priority, selected status moves and safer switches.",
};
