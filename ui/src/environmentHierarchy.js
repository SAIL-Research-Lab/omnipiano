const HAND_NUMBERS = {
  One: 1,
  Two: 2,
  Three: 3,
  Four: 4,
  Five: 5,
};

const GENERAL_LAYOUTS = {
  default: "Default",
  prototype: "Prototype",
  static_partition: "Static partition",
};

const ROBUST_TARGETS = {
  clean: "Clean",
  action: "Action",
  observation: "Observation",
  reward: "Reward",
  gravity: "Gravity",
  contact_friction: "Contact friction",
  hand_position: "Hand position",
  combined: "Combined / special",
};

const ROBUST_DISTRIBUTIONS = {
  none: "None",
  gaussian: "Gaussian",
  uniform: "Uniform",
  shift: "Shift",
  registered: "Registered variant",
};

const SAFETY_CONSTRAINTS = {
  joint_range: "Joint range",
  actuator_power: "Actuator power",
  injured_finger: "Injured finger",
  hand_collision: "Hand collision",
};

const SAFETY_SETTINGS = {
  event: "Event",
  excess: "Excess",
  fraction: "Fraction",
};

const SAFETY_BUDGETS = [1.44, 4.32, 14.4, 43.2, 144.0];

const HIERARCHIES = {
  standard: [
    { key: "piece", label: "Piece" },
    { key: "hands", label: "Number of hands", order: [1, 2, 3, 4, 5] },
    { key: "layout", label: "Layout", order: ["default", "prototype", "static_partition"] },
  ],
  robust: [
    {
      key: "target",
      label: "Perturbation target",
      order: ["clean", "action", "observation", "reward", "gravity", "contact_friction", "hand_position", "combined"],
    },
    { key: "distribution", label: "Distribution", order: ["none", "gaussian", "uniform", "shift", "registered"] },
    { key: "variant", label: "Level / variant" },
  ],
  safety: [
    {
      key: "constraint",
      label: "Safety constraint",
      order: ["joint_range", "actuator_power", "injured_finger", "hand_collision"],
    },
    { key: "setting", label: "Cost setting", order: ["event", "excess", "fraction"] },
    { key: "scenario", label: "Scenario" },
  ],
};

function humanize(value) {
  return String(value ?? "")
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replaceAll("_", " ")
    .replace(/\s+/g, " ")
    .trim();
}

function parseGeneral(item) {
  const match = item.id.match(
    /^OmniPiano-(.+?)-(One|Two|Three|Four|Five)Hand(?:(Prototype)|-(StaticPartition))?-GeneralRL-v0$/,
  );
  if (!match) return null;
  const layout = match[3] ? "prototype" : match[4] ? "static_partition" : "default";
  const hands = HAND_NUMBERS[match[2]];
  return {
    item,
    values: { piece: match[1], hands, layout },
    labels: {
      piece: humanize(match[1]),
      hands: `${hands} hand${hands === 1 ? "" : "s"}`,
      layout: GENERAL_LAYOUTS[layout],
    },
  };
}

function robustParts(id) {
  const body = id.replace(/^OmniPiano-/, "").replace(/-v\d+$/, "");
  const match = body.match(/^(ClairDeLune|ForElise|FantaisieImpromptu)(?:-(One|Two|Three|Four|Five)Hand)?-(.+)$/);
  if (!match) return { piece: "Registered", hands: null, detail: body };
  return {
    piece: match[1],
    hands: match[2] ? HAND_NUMBERS[match[2]] : 2,
    detail: match[3],
  };
}

function robustTarget(detail) {
  if (detail.startsWith("Clean")) return "clean";
  if (detail === "ActionRobust" || /^A-/.test(detail)) return "action";
  if (detail === "ObservationRobust" || /^O-/.test(detail)) return "observation";
  if (/^R-/.test(detail)) return "reward";
  if (/^G-/.test(detail)) return "gravity";
  if (/^CF-/.test(detail)) return "contact_friction";
  if (/^HP-/.test(detail)) return "hand_position";
  return "combined";
}

function robustDistribution(detail) {
  if (detail.startsWith("Clean")) return "none";
  if (detail.includes("Gauss") || ["ActionRobust", "ObservationRobust"].includes(detail)) return "gaussian";
  if (detail.includes("Uniform")) return "uniform";
  if (detail.includes("Shift")) return "shift";
  return "registered";
}

function severityFor(target, detail) {
  const code = detail.match(/(?:^|-)(P\d+)(?:-|$)/)?.[1];
  const levels = {
    action: { P05: "Low", P10: "Medium", P15: "High" },
    observation: { P10: "Low", P20: "Medium", P30: "High" },
    reward: { P10: "Low", P30: "Medium", P50: "High" },
    gravity: { P200: "Low", P300: "Medium", P400: "High" },
    contact_friction: { P20: "Low", P30: "Medium", P40: "High" },
  };
  if (target === "hand_position") {
    if (/Y10-Z0?5/.test(detail)) return "Low";
    if (/Y25-Z10/.test(detail)) return "Medium";
    if (/Y50-Z20/.test(detail)) return "High";
  }
  return levels[target]?.[code] ?? null;
}

function robustVariantLabel(target, detail, piece, hands) {
  if (detail.startsWith("Clean")) {
    const clean = ["Clean baseline"];
    if (detail.includes("NoOAR")) clean.push("No OAR");
    if (piece !== "ClairDeLune" || hands !== 2) clean.push(`${humanize(piece)}${hands ? ` · ${hands} hands` : ""}`);
    return clean.join(" · ");
  }
  if (detail === "ActionRobust" || detail === "ObservationRobust") {
    return `Legacy task · ${humanize(piece)}`;
  }

  const severity = severityFor(target, detail);
  const parts = [];
  if (severity) parts.push(severity);
  if (["gravity", "contact_friction"].includes(target)) {
    parts.push(detail.includes("-Step-") ? "Step" : "Episode");
  }
  if (/(?:^|-)N\d+/.test(detail) || /YN\d+|ZN\d+/.test(detail)) parts.push("Negative");
  if (detail.includes("NoOAR")) parts.push("No OAR");
  if (!parts.length || target === "combined") {
    const compact = detail
      .replace(/-(Gauss|Uniform|Shift)-?/, " · ")
      .replaceAll("-", " · ");
    parts.push(humanize(compact));
  }
  if (piece !== "ClairDeLune" || hands !== 2) {
    parts.push(`${humanize(piece)}${hands ? ` · ${hands} hands` : ""}`);
  }
  return parts.join(" · ");
}

function parseRobust(item) {
  const { piece, hands, detail } = robustParts(item.id);
  const target = robustTarget(detail);
  const distribution = robustDistribution(detail);
  return {
    item,
    values: { target, distribution, variant: item.id },
    labels: {
      target: ROBUST_TARGETS[target],
      distribution: ROBUST_DISTRIBUTIONS[distribution],
      variant: robustVariantLabel(target, detail, piece, hands),
    },
  };
}

function safetyMetadata(item) {
  if (item.cost && item.setting && item.song && item.hands) return item;
  const match = item.id.match(
    /^OmniPiano-Safety-(.+)-([2-5])H-([a-z_]+)-(event|excess|fraction)-[a-f0-9]+-v\d+$/,
  );
  if (!match) return item;
  return { ...item, song: match[1], hands: Number(match[2]), cost: match[3], setting: match[4] };
}

function safetyScenarioLabel(item) {
  const base = `${humanize(item.song)} · ${item.hands} hands`;
  if (item.group === "hands") return `${base} · Hand scalability`;
  if (item.group === "budget") {
    const budget = SAFETY_BUDGETS[item.task_index];
    return `${base}${budget == null ? " · Budget sweep" : ` · Budget ${budget}`}`;
  }
  return base;
}

function parseSafety(source) {
  const item = safetyMetadata(source);
  if (!item.cost || !item.setting || !item.song || !item.hands) return null;
  return {
    item,
    values: { constraint: item.cost, setting: item.setting, scenario: item.id },
    labels: {
      constraint: SAFETY_CONSTRAINTS[item.cost] ?? humanize(item.cost),
      setting: SAFETY_SETTINGS[item.setting] ?? humanize(item.setting),
      scenario: safetyScenarioLabel(item),
    },
  };
}

function fallbackRecord(item) {
  return {
    item,
    values: { task: item.id },
    labels: { task: humanize(item.label ?? item.id) },
  };
}

export function buildTaskHierarchy(track, environments, currentId) {
  const unique = new Map(environments.map((item) => [item.id, item]));
  if (currentId && !unique.has(currentId)) unique.set(currentId, { id: currentId, label: currentId });
  const parser = { standard: parseGeneral, robust: parseRobust, safety: parseSafety }[track];
  const parsed = [...unique.values()].map((item) => parser?.(item)).filter(Boolean);
  if (parsed.length) return { levels: HIERARCHIES[track], records: parsed };
  const records = [...unique.values()].map(fallbackRecord);
  return { levels: [{ key: "task", label: "Task" }], records };
}

function compareOptions(level, left, right) {
  const leftIndex = level.order?.indexOf(left.value) ?? -1;
  const rightIndex = level.order?.indexOf(right.value) ?? -1;
  if (leftIndex >= 0 || rightIndex >= 0) {
    if (leftIndex < 0) return 1;
    if (rightIndex < 0) return -1;
    return leftIndex - rightIndex;
  }
  const tierOrder = ["Clean baseline", "Low", "Medium", "High"];
  const leftTier = tierOrder.findIndex((tier) => left.label.startsWith(tier));
  const rightTier = tierOrder.findIndex((tier) => right.label.startsWith(tier));
  if (leftTier >= 0 || rightTier >= 0) {
    if (leftTier < 0) return 1;
    if (rightTier < 0) return -1;
    if (leftTier !== rightTier) return leftTier - rightTier;
  }
  return left.label.localeCompare(right.label, undefined, { numeric: true });
}

export function optionsForHierarchyLevel(records, levels, selected, levelIndex) {
  const eligible = records.filter((record) => levels.slice(0, levelIndex).every(
    (level) => record.values[level.key] === selected?.values[level.key],
  ));
  const level = levels[levelIndex];
  const options = new Map();
  eligible.forEach((record) => {
    const value = record.values[level.key];
    options.set(value, { value, label: record.labels[level.key] ?? humanize(value) });
  });
  return [...options.values()].sort((left, right) => compareOptions(level, left, right));
}

export function chooseHierarchyRecord(records, levels, currentId, levelIndex, value) {
  const current = records.find((record) => record.item.id === currentId) ?? records[0];
  const required = Object.fromEntries([
    ...levels.slice(0, levelIndex).map((level) => [level.key, current?.values[level.key]]),
    [levels[levelIndex].key, value],
  ]);
  const eligible = records.filter((record) => Object.entries(required).every(
    ([key, expected]) => record.values[key] === expected,
  ));
  return eligible.sort((left, right) => {
    const leftScore = levels.slice(levelIndex + 1).filter((level) => left.values[level.key] === current?.values[level.key]).length;
    const rightScore = levels.slice(levelIndex + 1).filter((level) => right.values[level.key] === current?.values[level.key]).length;
    return rightScore - leftScore || left.item.id.localeCompare(right.item.id, undefined, { numeric: true });
  })[0] ?? current;
}

