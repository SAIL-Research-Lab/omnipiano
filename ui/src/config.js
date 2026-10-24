export const TRACKS = [
  { id: "standard", short: "S", title: "Standard RL", description: "Nominal control" },
  { id: "robust", short: "R", title: "Robust RL", description: "Perturbed tasks" },
  { id: "safety", short: "C", title: "Safe RL", description: "Constrained learning" },
  { id: "marl", short: "M", title: "Multi-Agent RL", description: "Multi-agent coordination" },
];

export const STEPS = [
  { id: "experiment", number: "01", title: "Experiment" },
  { id: "task", number: "02", title: "Piano task" },
  { id: "design", number: "03", title: "Track design" },
  { id: "algorithm", number: "04", title: "Algorithm" },
  { id: "run", number: "05", title: "Train & evaluate" },
  { id: "review", number: "06", title: "Review & launch" },
];

export const REPERTOIRE = [
  "WinterWind",
  "PicturesGreatKiev",
  "PolonaiseOp40No1",
  "PianoSonataNo281StMov",
  "PianoSonataNo301StMov",
].map((value) => ({ value, label: value.replace(/([a-z])([A-Z])/g, "$1 $2") }));

const HAND_LAYOUTS = {
  3: {
    preset: "ThreeHand",
    hands: [["lh", "left", [1, 29]], ["rh_c", "right", [30, 59]], ["rh", "right", [60, 88]]],
  },
  4: {
    preset: "FourHand",
    hands: [["lh_b", "left", [1, 22]], ["rh_b", "right", [23, 44]], ["lh_t", "left", [45, 66]], ["rh_t", "right", [67, 88]]],
  },
  5: {
    preset: "FiveHand",
    hands: [["lh_b", "left", [1, 18]], ["rh_b", "right", [19, 36]], ["rh_c", "right", [37, 53]], ["lh_t", "left", [54, 71]], ["rh_t", "right", [72, 88]]],
  },
};

export function cloneConfig(config) { return structuredClone(config); }

export function withUiDefaults(config) {
  const next = cloneConfig(config);
  next.experiment.description ??= "";
  next.design.config ??= {};
  next.algorithm.preset ??= "default";
  next.algorithm.parameters ??= {};
  next.run.evaluation ??= { enabled: true, frequency: 50000, episodes: 1, seed_offset: 10000 };
  next.run.evaluation.seed_offset ??= 10000;
  next.run.video ??= { enabled: false, frequency: 250000, camera: "piano/back", width: 640, height: 480 };
  next.run.compute ??= { device: "auto", num_workers: 1, num_envs: 1 };
  next.run.tracking ??= { provider: "none", mode: "disabled", project: null, group: null, name: null, tags: [] };
  next.run.tracking.project ??= null;
  next.run.tracking.group ??= null;
  next.run.tracking.name ??= null;
  next.run.tracking.tags ??= [];
  next.run.output ??= { root: "runs" };
  return next;
}

export function setAtPath(source, path, value) {
  const next = cloneConfig(source);
  const parts = Array.isArray(path) ? path : path.split(".");
  let cursor = next;
  for (let index = 0; index < parts.length - 1; index += 1) cursor = cursor[parts[index]];
  cursor[parts.at(-1)] = value;
  return next;
}

export function parseIntegerList(value) {
  return [...new Set(value.split(",").map((item) => item.trim()).filter(Boolean).map(Number)
    .filter((item) => Number.isInteger(item) && item >= 0))];
}

function distributeHands(handIds, numAgents) {
  const assignments = Array.from({ length: numAgents }, () => []);
  handIds.forEach((handId, index) => {
    assignments[Math.min(numAgents - 1, Math.floor((index * numAgents) / handIds.length))].push(handId);
  });
  return assignments;
}

function rangeForHands(hands, handIds) {
  const ranges = handIds.map((id) => hands.find((hand) => hand.id === id).initial_key_range);
  return [Math.min(...ranges.map((range) => range[0])), Math.max(...ranges.map((range) => range[1]))];
}

function rebalanceMarl(next, numAgents) {
  const hands = next.task.embodiment.hands;
  const count = Math.min(Math.max(1, Number(numAgents)), hands.length);
  const previous = next.design.config.agents ?? [];
  const assignments = distributeHands(hands.map((hand) => hand.id), count);
  next.design.config.num_agents = count;
  next.design.config.assignment = "explicit";
  next.design.config.agents = assignments.map((handIds, index) => {
    const old = previous[index];
    const range = rangeForHands(hands, handIds);
    return {
      name: old?.name ?? `agent_${index + 1}`,
      hand_ids: handIds,
      action_key_range: old?.action_key_range ?? range,
      observation_key_range: old?.observation_key_range ?? range,
      visible_teammate_hands: old?.visible_teammate_hands ?? "boundary",
    };
  });
  if (!next.design.config.agents.some((agent) => agent.name === next.design.config.sustain_owner)) {
    next.design.config.sustain_owner = next.design.config.agents[0].name;
  }
}

export function makeCustomMarl(source) {
  const next = cloneConfig(source);
  const layout = HAND_LAYOUTS[4];
  next.task = {
    source: "custom",
    music: { source: "repertoire", piece: "WinterWind", midi_path: null, trim_silence: true, stretch_factor: 1, shift_factor: 0, fingering_mode: "optimal_transport" },
    embodiment: {
      num_hands: 4,
      layout_preset: layout.preset,
      mobility: "custom",
      hands: layout.hands.map(([name, side, initial_key_range], id) => ({ id, name, side, mobile: true, initial_key_range })),
    },
  };
  next.design.config = {
    num_agents: 2,
    assignment: "explicit",
    sustain_owner: "agent_1",
    agents: [],
    coordination: { inter_agent_collision_penalty_coefficient: 0.1 },
  };
  rebalanceMarl(next, 2);
  return next;
}

export function makeRegisteredTask(source, environment, track = source.design.type) {
  const next = cloneConfig(source);
  next.task = { source: "registered", environment_id: environment.id };
  if (track === "safety") next.design.config = { group: environment.group, task_index: environment.task_index };
  else if (track === "robust") next.design.config = { evaluation_protocol: "matched" };
  else if (track === "marl") next.design.config = { assignment: "territorial" };
  else next.design = { type: "standard" };
  return next;
}

export function applyHandLayout(source, numHands) {
  const next = cloneConfig(source);
  if (next.task.source !== "custom" || !HAND_LAYOUTS[numHands]) return next;
  const layout = HAND_LAYOUTS[numHands];
  next.task.embodiment = {
    num_hands: numHands,
    layout_preset: layout.preset,
    mobility: "custom",
    hands: layout.hands.map(([name, side, initial_key_range], id) => ({ id, name, side, mobile: true, initial_key_range })),
  };
  rebalanceMarl(next, Math.min(next.design.config.num_agents, numHands));
  return next;
}

export function applyMarlAgentCount(source, numAgents) {
  const next = cloneConfig(source);
  if (next.design.type === "marl" && next.task.source === "custom") rebalanceMarl(next, numAgents);
  return next;
}

export function assignMarlHandOwner(source, handId, targetAgentIndex) {
  const next = cloneConfig(source);
  const agents = next.design.config.agents;
  const sourceIndex = agents.findIndex((agent) => agent.hand_ids.includes(handId));
  if (sourceIndex < 0 || sourceIndex === targetAgentIndex) return next;
  if (agents[sourceIndex].hand_ids.length === 1) {
    const swap = agents[targetAgentIndex].hand_ids[0];
    agents[targetAgentIndex].hand_ids = agents[targetAgentIndex].hand_ids.filter((id) => id !== swap);
    agents[sourceIndex].hand_ids.push(swap);
  }
  agents[sourceIndex].hand_ids = agents[sourceIndex].hand_ids.filter((id) => id !== handId);
  agents[targetAgentIndex].hand_ids.push(handId);
  agents.forEach((agent) => agent.hand_ids.sort((left, right) => left - right));
  return next;
}

export function applyTrackingProvider(source, provider) {
  const next = cloneConfig(source);
  next.run.tracking.provider = provider;
  next.run.tracking.mode = provider === "none" ? "disabled" :
    next.run.tracking.mode === "disabled" ? "offline" : next.run.tracking.mode;
  return next;
}

export function selectAlgorithm(source, name) {
  const next = cloneConfig(source);
  next.algorithm = { name, preset: "default", parameters: {} };
  return next;
}
