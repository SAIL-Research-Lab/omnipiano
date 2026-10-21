import Ajv2020 from "ajv/dist/2020";
import schema from "../contract/experiment.schema.json";
import { findAlgorithm } from "./algorithmCatalog";
import { getSafetyPaperBudget } from "./config";

const schemaValidator = new Ajv2020({ allErrors: true, strict: false }).compile(schema);

function labelSchemaError(error) {
  const path = error.instancePath.replaceAll("/", ".").replace(/^\./, "") || "configuration";
  return `${path}: ${error.message}`;
}

function addRangeError(errors, range, label) {
  if (range && range[0] > range[1]) {
    errors.push(`${label}: range start must not exceed range end`);
  }
}

function validateMarl(config, hands, errors) {
  const marl = config.design.config;
  if (marl.num_agents !== marl.agents.length) {
    errors.push("MARL: agent count must match the number of agent cards");
  }

  const names = marl.agents.map((agent) => agent.name);
  if (new Set(names).size !== names.length) errors.push("MARL: agent names must be unique");
  if (!names.includes(marl.sustain_owner)) errors.push("MARL: sustain owner must be a configured agent");

  const expected = new Set(hands.map((hand) => hand.id));
  const owned = marl.agents.flatMap((agent) => agent.hand_ids);
  if (new Set(owned).size !== owned.length) errors.push("MARL: one hand cannot belong to multiple agents");
  if (owned.length !== expected.size || owned.some((id) => !expected.has(id))) {
    errors.push("MARL: every configured hand must belong to exactly one agent");
  }

  marl.agents.forEach((agent) => {
    addRangeError(errors, agent.action_key_range, `MARL ${agent.name} action range`);
    addRangeError(errors, agent.observation_key_range, `MARL ${agent.name} observation range`);
    if (Array.isArray(agent.visible_teammate_hands)) {
      if (agent.visible_teammate_hands.some((id) => !expected.has(id))) {
        errors.push(`MARL ${agent.name}: visible teammate hand does not exist`);
      }
      if (agent.visible_teammate_hands.some((id) => agent.hand_ids.includes(id))) {
        errors.push(`MARL ${agent.name}: an owned hand cannot be listed as a teammate`);
      }
    }
  });
}

function validateSafety(config, hands, errors) {
  const cost = config.design.config.primary_cost;
  const handIds = new Set(hands.map((hand) => hand.id));
  if (cost.target_hand_ids.some((id) => !handIds.has(id))) {
    errors.push("Safety: target hands must exist in the selected embodiment");
  }

  if (cost.type.startsWith("range_")) {
    if (!cost.target_joints.length) errors.push("Safety: range costs require target joints");
    if (cost.safe_interval_fraction == null) errors.push("Safety: range costs require a safe interval");
  } else {
    if (cost.power_threshold == null) errors.push("Safety: power costs require a power threshold");
    if (config.task.reward.energy_regularization.enabled) {
      errors.push("Safety: disable energy reward when actuator power is the primary cost");
    }
  }

  if (cost.type === "power_spike") {
    errors.push("Safety: Power Spike has no finalized definition; choose another cost");
  }
  if (cost.budget_mode === "paper_default" && !getSafetyPaperBudget(config)) {
    errors.push("Safety: this piece has no paper-default budget; choose Custom");
  }
}

function validateRobust(config, errors) {
  const channels = config.design.config.channels;
  const targets = channels.map((channel) => channel.target);
  if (new Set(targets).size !== targets.length) {
    errors.push("Robust: each perturbation target can appear only once");
  }
  channels.forEach((channel) => {
    if (channel.target === "initial_hand_pose") {
      if (!("y_mm" in channel.magnitude) || !("z_mm" in channel.magnitude)) {
        errors.push("Robust initial hand pose requires y_mm and z_mm magnitudes");
      }
    } else if (channel.distribution === "bounded_uniform") {
      if (!("low" in channel.magnitude) || !("high" in channel.magnitude)) {
        errors.push(`Robust ${channel.target}: bounded uniform requires low and high`);
      } else if (channel.magnitude.low > channel.magnitude.high) {
        errors.push(`Robust ${channel.target}: lower bound must not exceed upper bound`);
      }
    } else if (!("value" in channel.magnitude)) {
      errors.push(`Robust ${channel.target}: this distribution requires a value magnitude`);
    }
  });
}

export function validateExperiment(config) {
  const errors = [];
  const warnings = [];

  if (!schemaValidator(config)) {
    errors.push(...schemaValidator.errors.map(labelSchemaError));
    return { errors: [...new Set(errors)], warnings };
  }

  const { music, embodiment } = config.task;
  const hands = embodiment.hands;
  if (hands.length !== embodiment.num_hands) {
    errors.push("Piano task: number of hands must match the generated hand layout");
  }
  const ids = hands.map((hand) => hand.id);
  if (new Set(ids).size !== ids.length) errors.push("Piano task: hand IDs must be unique");
  hands.forEach((hand) => addRangeError(errors, hand.initial_key_range, `Hand ${hand.name}`));

  if (music.fingering_mode === "annotated" && embodiment.num_hands !== 2) {
    errors.push("Piano task: annotated fingering is only defined for two-hand tasks");
  }
  if (music.fingering_mode === "annotated" && music.source === "custom_midi") {
    warnings.push("Confirm that the uploaded MIDI contains fingering annotations");
  }

  if (config.design.type === "standard") {
    const variant = config.design.config.benchmark_variant;
    const expectedMobility = {
      general: "unrestricted",
      mobility: "custom",
      restricted_range: "static_partition",
    }[variant];
    if (embodiment.mobility !== expectedMobility) {
      errors.push("Standard: mobility metadata must match the selected variant");
    }
  }

  const video = config.evaluation.video;
  if (video.enabled && video.frequency === 0) {
    errors.push("Evaluation video frequency must be positive when recording is enabled");
  }
  if (video.enabled && video.frequency % config.evaluation.frequency !== 0) {
    warnings.push("Video frequency is not a multiple of evaluation frequency");
  }
  if (config.tracking.provider === "none" && config.tracking.mode !== "disabled") {
    errors.push("Tracking mode must be disabled when no provider is selected");
  }
  if (config.tracking.provider === "wandb" && config.tracking.mode === "disabled") {
    errors.push("W&B tracking mode must be Offline or Online");
  }

  const algorithm = findAlgorithm(config.design.type, config.algorithm.name);
  if (!algorithm) {
    errors.push("Algorithm: choose one of the available options for this track");
  } else {
    if (config.algorithm.framework !== algorithm.framework || config.algorithm.family !== algorithm.family) {
      errors.push("Algorithm: framework and family must match the selected algorithm");
    }
    if (config.design.type === "marl" && config.design.config.critic_mode !== algorithm.critic) {
      errors.push("MARL: critic information must match the selected algorithm");
    }
  }

  if (config.design.type === "marl") validateMarl(config, hands, errors);
  if (config.design.type === "safety") validateSafety(config, hands, errors);
  if (config.design.type === "robust") validateRobust(config, errors);

  return {
    errors: [...new Set(errors)],
    warnings: [...new Set(warnings)],
  };
}
