export const TRACKS = [
  {
    id: "standard",
    short: "S",
    title: "Standard RL",
    description: "Nominal control",
  },
  {
    id: "robust",
    short: "R",
    title: "Robust RL",
    description: "Perturbed tasks",
  },
  {
    id: "safety",
    short: "C",
    title: "Safe RL",
    description: "Constrained learning",
  },
  {
    id: "marl",
    short: "M",
    title: "Multi-Agent RL",
    description: "Multi-agent coordination",
  },
];

export const STEPS = [
  { id: "experiment", number: "01", title: "Experiment" },
  { id: "task", number: "02", title: "Piano task" },
  { id: "design", number: "03", title: "Track design" },
  { id: "algorithm", number: "04", title: "Algorithm" },
  { id: "run", number: "05", title: "Train & evaluate" },
  { id: "review", number: "06", title: "Review & export" },
];

export const REPERTOIRE = [
  { value: "TwinkleTwinkleLittleStar", label: "Twinkle Twinkle Little Star" },
  { value: "ForElise", label: "Für Elise" },
  { value: "ClairDeLune", label: "Clair de Lune" },
  { value: "WinterWind", label: "Winter Wind" },
  { value: "PicturesGreatKiev", label: "Pictures at an Exhibition · Great Gate of Kiev" },
  { value: "NocturneOp9No2", label: "Nocturne Op. 9 No. 2" },
  { value: "PianoSonataNo281StMov", label: "Piano Sonata No. 28 · I" },
  { value: "PianoSonataNo301StMov", label: "Piano Sonata No. 30 · I" },
  { value: "PolonaiseOp40No1", label: "Polonaise Op. 40 No. 1" },
];

export const ROBUST_CHANNELS = {
  action: {
    label: "Action noise",
    group: "signal",
    levels: { low: 0.05, medium: 0.1, high: 0.15 },
    frequency: "step",
  },
  observation: {
    label: "Observation noise",
    group: "signal",
    levels: { low: 0.05, medium: 0.1, high: 0.15 },
    frequency: "step",
  },
  reward: {
    label: "Reward noise",
    group: "signal",
    levels: { low: 0.1, medium: 0.3, high: 0.5 },
    frequency: "step",
  },
  gravity: {
    label: "Gravity",
    group: "physical",
    levels: { low: 0.5, medium: 1.0, high: 1.5 },
    frequency: "episode",
  },
  friction: {
    label: "Key friction",
    group: "physical",
    levels: { low: 0.05, medium: 0.1, high: 0.15 },
    frequency: "episode",
  },
  initial_hand_pose: {
    label: "Initial hand pose",
    group: "physical",
    levels: {
      low: { y_mm: 10, z_mm: 5 },
      medium: { y_mm: 25, z_mm: 10 },
      high: { y_mm: 50, z_mm: 20 },
    },
    frequency: "episode",
  },
};

const SAFETY_SONG_HORIZONS = {
  TwinkleTwinkleLittleStar: 161,
  WinterWind: 314,
  ForElise: 399,
  ClairDeLune: 588,
  PicturesGreatKiev: 720,
};

const HAND_LAYOUTS = {
  1: {
    name: "OneHand",
    hands: [["rh", "right", [1, 88]]],
  },
  2: {
    name: "TwoHand",
    hands: [
      ["lh", "left", [1, 44]],
      ["rh", "right", [45, 88]],
    ],
  },
  3: {
    name: "ThreeHand",
    hands: [
      ["lh", "left", [1, 29]],
      ["rh_c", "right", [30, 59]],
      ["rh", "right", [60, 88]],
    ],
  },
  4: {
    name: "FourHand",
    hands: [
      ["lh_b", "left", [1, 22]],
      ["rh_b", "right", [23, 44]],
      ["lh_t", "left", [45, 66]],
      ["rh_t", "right", [67, 88]],
    ],
  },
  5: {
    name: "FiveHand",
    hands: [
      ["lh_b", "left", [1, 18]],
      ["rh_b", "right", [19, 36]],
      ["rh_c", "right", [37, 53]],
      ["lh_t", "left", [54, 71]],
      ["rh_t", "right", [72, 88]],
    ],
  },
};

export function cloneConfig(config) {
  return structuredClone(config);
}

export function upgradeImportedConfig(source) {
  if (source?.schema_version !== 1) return source;

  const next = cloneConfig(source);
  next.schema_version = 2;
  if (next.design?.type === "marl") {
    delete next.design.config.scho_axis;
    const agents = next.design.config.agents;
    if (agents.length === 2 && agents[0].name === "secondo" && agents[1].name === "primo") {
      const names = { secondo: "agent_1", primo: "agent_2" };
      agents.forEach((agent) => { agent.name = names[agent.name]; });
      next.design.config.sustain_owner = names[next.design.config.sustain_owner]
        ?? next.design.config.sustain_owner;
    }
  }
  return next;
}

export function setAtPath(source, path, value) {
  const next = cloneConfig(source);
  const parts = Array.isArray(path) ? path : path.split(".");
  let cursor = next;
  for (let index = 0; index < parts.length - 1; index += 1) {
    cursor = cursor[parts[index]];
  }
  cursor[parts.at(-1)] = value;
  return next;
}

export function getSafetyPaperBudget(config, costType = config.design.config.primary_cost.type) {
  if (config.task.music.source !== "repertoire") return null;
  const horizon = SAFETY_SONG_HORIZONS[config.task.music.piece];
  if (!horizon) return null;
  const rate = ["range_excess", "power_excess"].includes(costType) ? 0.02 : 0.05;
  return { horizon, rate, value: Number((horizon * rate).toFixed(2)) };
}

export function applyMusicSelection(source, changes) {
  const next = cloneConfig(source);
  Object.assign(next.task.music, changes);
  if (
    next.design.type === "safety"
    && next.design.config.primary_cost.budget_mode === "paper_default"
  ) {
    const budget = getSafetyPaperBudget(next);
    if (budget) next.design.config.primary_cost.episodic_budget = budget.value;
    else next.design.config.primary_cost.budget_mode = "custom";
  }
  return next;
}

export function applyTrackingProvider(source, provider) {
  const next = cloneConfig(source);
  next.tracking.provider = provider;
  if (provider === "none") next.tracking.mode = "disabled";
  if (provider === "wandb" && next.tracking.mode === "disabled") {
    next.tracking.mode = "offline";
  }
  return next;
}

function distributeHands(handIds, numAgents) {
  const assignments = Array.from({ length: numAgents }, () => []);
  handIds.forEach((handId, index) => {
    const agentIndex = Math.min(
      numAgents - 1,
      Math.floor((index * numAgents) / handIds.length),
    );
    assignments[agentIndex].push(handId);
  });
  return assignments;
}

function rebalanceMarl(config) {
  if (config.design.type !== "marl") return;

  const marl = config.design.config;
  const handIds = config.task.embodiment.hands.map((hand) => hand.id);
  const numAgents = Math.min(Math.max(1, marl.num_agents), handIds.length);
  const assignments = distributeHands(handIds, numAgents);
  const previous = marl.agents;

  marl.num_agents = numAgents;
  marl.assignment = "balanced";
  marl.agents = assignments.map((ownedHands, index) => {
    const existing = previous[index];
    const ranges = ownedHands.map(
      (handId) => config.task.embodiment.hands.find((hand) => hand.id === handId).initial_key_range,
    );
    const keyRange = [
      Math.min(...ranges.map((range) => range[0])),
      Math.max(...ranges.map((range) => range[1])),
    ];
    return {
      name: existing?.name ?? `agent_${index + 1}`,
      hand_ids: ownedHands,
      action_key_range: keyRange,
      observation_key_range: keyRange,
      visible_teammate_hands: "boundary",
    };
  });

  if (!marl.agents.some((agent) => agent.name === marl.sustain_owner)) {
    marl.sustain_owner = marl.agents.at(-1).name;
  }
}

export function applyHandLayout(source, numHands, mobility) {
  if (numHands === source.task.embodiment.num_hands && mobility === source.task.embodiment.mobility) {
    return cloneConfig(source);
  }
  const next = cloneConfig(source);
  const layout = HAND_LAYOUTS[numHands];
  const isStatic = mobility === "static_partition";
  const previousHands = source.task.embodiment.hands;

  next.task.embodiment.num_hands = numHands;
  next.task.embodiment.mobility = mobility;
  next.task.embodiment.layout_preset = isStatic
    ? `${layout.name}StaticPartition`
    : layout.name;
  next.task.embodiment.hands = layout.hands.map(([name, side, range], id) => ({
    id,
    name,
    side,
    mobile: isStatic ? false : mobility === "custom"
      ? (previousHands.find((hand) => hand.name === name)?.mobile ?? id % 2 === 0)
      : true,
    initial_key_range: [...range],
  }));
  if (numHands !== 2) {
    next.task.music.fingering_mode = "optimal_transport";
  }
  if (next.design.type === "safety") {
    const validIds = new Set(next.task.embodiment.hands.map((hand) => hand.id));
    const cost = next.design.config.primary_cost;
    cost.target_hand_ids = cost.target_hand_ids.filter((id) => validIds.has(id));
    if (!cost.target_hand_ids.length) cost.target_hand_ids = [...validIds];
  }
  rebalanceMarl(next);
  return next;
}

export function applyStandardVariant(source, variant) {
  if (source.design.type !== "standard") return cloneConfig(source);
  if (source.design.config.benchmark_variant === variant) return cloneConfig(source);
  const mobility = {
    general: "unrestricted",
    mobility: "custom",
    restricted_range: "static_partition",
  }[variant];
  const next = applyHandLayout(
    source,
    source.task.embodiment.num_hands,
    mobility,
  );
  if (variant === "mobility" && source.design.config.benchmark_variant !== "mobility") {
    next.task.embodiment.hands.forEach((hand, index) => { hand.mobile = index % 2 === 0; });
  }
  next.design.config.benchmark_variant = variant;
  return next;
}

export function setStandardHandMobile(source, handId, mobile) {
  const next = cloneConfig(source);
  if (next.design.type !== "standard" || next.design.config.benchmark_variant !== "mobility") return next;
  next.task.embodiment.hands[handId].mobile = mobile;
  next.task.embodiment.mobility = "custom";
  return next;
}

export function applyMarlAgentCount(source, numAgents) {
  const next = cloneConfig(source);
  if (next.design.type !== "marl") return next;
  if (numAgents === next.design.config.num_agents) return next;
  next.design.config.num_agents = Math.min(
    Math.max(1, numAgents),
    next.task.embodiment.num_hands,
  );
  rebalanceMarl(next);
  return next;
}

export function assignMarlHandOwner(source, handId, targetAgentIndex) {
  const next = cloneConfig(source);
  if (next.design.type !== "marl") return next;
  const agents = next.design.config.agents;
  const sourceIndex = agents.findIndex((agent) => agent.hand_ids.includes(handId));
  if (sourceIndex < 0 || sourceIndex === targetAgentIndex) return next;

  const sourceAgent = agents[sourceIndex];
  const targetAgent = agents[targetAgentIndex];
  if (sourceAgent.hand_ids.length === 1) {
    const swapHand = targetAgent.hand_ids[0];
    targetAgent.hand_ids = targetAgent.hand_ids.filter((id) => id !== swapHand);
    targetAgent.hand_ids.push(handId);
    sourceAgent.hand_ids = [swapHand];
  } else {
    sourceAgent.hand_ids = sourceAgent.hand_ids.filter((id) => id !== handId);
    targetAgent.hand_ids.push(handId);
  }
  agents.forEach((agent) => agent.hand_ids.sort((left, right) => left - right));
  agents.forEach((agent) => {
    if (Array.isArray(agent.visible_teammate_hands)) {
      agent.visible_teammate_hands = agent.visible_teammate_hands.filter(
        (id) => !agent.hand_ids.includes(id),
      );
    }
  });
  next.design.config.assignment = "explicit";
  return next;
}

function robustMagnitude(target, preset, distribution) {
  const channel = ROBUST_CHANNELS[target];
  const level = channel.levels[preset === "custom" ? "medium" : preset];
  if (target === "initial_hand_pose") return { ...level };
  if (distribution === "bounded_uniform") return { low: -level, high: level };
  return { value: level };
}

export function addRobustChannel(source, target) {
  const next = cloneConfig(source);
  if (next.design.type !== "robust") return next;
  if (next.design.config.channels.some((channel) => channel.target === target)) return next;
  const metadata = ROBUST_CHANNELS[target];
  next.design.config.channels.push({
    target,
    distribution: "gaussian",
    preset: "medium",
    frequency: metadata.frequency,
    magnitude: robustMagnitude(target, "medium", "gaussian"),
  });
  return next;
}

export function removeRobustChannel(source, target) {
  const next = cloneConfig(source);
  if (next.design.type !== "robust") return next;
  if (next.design.config.channels.length === 1) return next;
  next.design.config.channels = next.design.config.channels.filter(
    (channel) => channel.target !== target,
  );
  return next;
}

export function updateRobustChannel(source, index, changes) {
  const next = cloneConfig(source);
  if (next.design.type !== "robust") return next;
  const channel = next.design.config.channels[index];
  if (Object.entries(changes).every(([key, value]) => channel[key] === value)) return next;
  const previousMagnitude = channel.magnitude;
  Object.assign(channel, changes);
  if ("distribution" in changes && channel.preset === "custom" && channel.target === "initial_hand_pose") {
    return next;
  }
  if ("distribution" in changes && channel.preset === "custom") {
    const level = previousMagnitude.value ?? Math.max(
      Math.abs(previousMagnitude.low),
      Math.abs(previousMagnitude.high),
    );
    channel.magnitude = channel.distribution === "bounded_uniform"
      ? { low: -level, high: level }
      : { value: level };
  } else if ("distribution" in changes || ("preset" in changes && channel.preset !== "custom")) {
    channel.magnitude = robustMagnitude(
      channel.target,
      channel.preset,
      channel.distribution,
    );
  }
  return next;
}

export function setRobustMagnitude(source, index, key, value) {
  const next = cloneConfig(source);
  if (next.design.type !== "robust") return next;
  const channel = next.design.config.channels[index];
  channel.preset = "custom";
  channel.magnitude[key] = value;
  return next;
}

export function parseIntegerList(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map(Number)
    .filter(Number.isInteger);
}

export function parseStringList(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}
