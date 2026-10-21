import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import Ajv2020 from "ajv/dist/2020.js";
import { algorithmsForTrack, selectAlgorithm } from "../src/algorithmCatalog.js";

import {
  addRobustChannel,
  applyHandLayout,
  applyMarlAgentCount,
  applyMusicSelection,
  applyStandardVariant,
  applyTrackingProvider,
  assignMarlHandOwner,
  getSafetyPaperBudget,
  removeRobustChannel,
  ROBUST_CHANNELS,
  setRobustMagnitude,
  setStandardHandMobile,
  updateRobustChannel,
  parseIntegerList,
  parseStringList,
  upgradeImportedConfig,
} from "../src/config.js";

const ROOT = new URL("../", import.meta.url);

async function readJson(path) {
  return JSON.parse(await readFile(new URL(path, ROOT), "utf8"));
}

test("all four canonical examples validate and round-trip", async () => {
  const schema = await readJson("contract/experiment.schema.json");
  const validate = new Ajv2020({ allErrors: true, strict: false }).compile(schema);

  for (const track of ["standard", "robust", "safety", "marl"]) {
    const document = await readJson(`contract/examples/${track}.json`);
    assert.equal(validate(document), true, JSON.stringify(validate.errors));
    assert.deepEqual(JSON.parse(JSON.stringify(document)), document);
  }
});

test("paper safety budget follows repertoire changes but preserves custom budgets", async () => {
  const original = await readJson("contract/examples/safety.json");
  assert.deepEqual(getSafetyPaperBudget(original), {
    horizon: 399,
    rate: 0.05,
    value: 19.95,
  });

  const winter = applyMusicSelection(original, { piece: "WinterWind" });
  assert.equal(winter.design.config.primary_cost.episodic_budget, 15.7);

  const custom = structuredClone(original);
  custom.design.config.primary_cost.budget_mode = "custom";
  custom.design.config.primary_cost.episodic_budget = 7.25;
  assert.equal(
    applyMusicSelection(custom, { piece: "WinterWind" }).design.config.primary_cost.episodic_budget,
    7.25,
  );
});

test("hand and agent changes preserve exact MARL ownership", async () => {
  let config = await readJson("contract/examples/marl.json");
  assert.deepEqual(config.design.config.agents.map((agent) => agent.name), ["agent_1", "agent_2"]);
  assert.equal(config.design.config.sustain_owner, "agent_1");
  assert.equal("scho_axis" in config.design.config, false);
  config = applyHandLayout(config, 5, "custom");
  config = applyMarlAgentCount(config, 3);
  config = assignMarlHandOwner(config, 0, 1);

  assert.equal(config.task.embodiment.hands.length, 5);
  assert.equal(config.design.config.agents.length, 3);
  assert.equal(config.design.config.assignment, "explicit");
  const owned = config.design.config.agents.flatMap((agent) => agent.hand_ids);
  assert.deepEqual([...owned].sort((left, right) => left - right), [0, 1, 2, 3, 4]);
  assert.equal(new Set(owned).size, 5);
  assert.ok(config.design.config.agents.every((agent) => agent.hand_ids.length > 0));
  assert.deepEqual(config.design.config.agents.map((agent) => agent.name), ["agent_1", "agent_2", "agent_3"]);
  config.design.config.agents[0].action_key_range = [1, 44];
  assert.deepEqual(applyMarlAgentCount(config, 3).design.config.agents[0].action_key_range, [1, 44]);
  assert.deepEqual(applyHandLayout(config, 5, "custom").design.config.agents[0].action_key_range, [1, 44]);
});

test("version-1 MARL imports upgrade without losing ownership or ranges", async () => {
  const current = await readJson("contract/examples/marl.json");
  const legacy = structuredClone(current);
  legacy.schema_version = 1;
  legacy.design.config.scho_axis = "coupling";
  legacy.design.config.agents[0].name = "secondo";
  legacy.design.config.agents[1].name = "primo";
  legacy.design.config.sustain_owner = "secondo";

  const upgraded = upgradeImportedConfig(legacy);
  assert.equal(upgraded.schema_version, 2);
  assert.equal("scho_axis" in upgraded.design.config, false);
  assert.deepEqual(upgraded.design.config.agents.map((agent) => agent.name), ["agent_1", "agent_2"]);
  assert.equal(upgraded.design.config.sustain_owner, "agent_1");
  assert.deepEqual(upgraded.design.config.agents[0].action_key_range, [1, 52]);
  assert.equal(legacy.schema_version, 1);
  assert.equal(legacy.design.config.scho_axis, "coupling");

  const schema = await readJson("contract/experiment.schema.json");
  const validate = new Ajv2020({ strict: false }).compile(schema);
  assert.equal(validate(upgraded), true, JSON.stringify(validate.errors));
});

test("robust channel builder supports every channel and distribution shape", async () => {
  let config = await readJson("contract/examples/robust.json");
  for (const target of Object.keys(ROBUST_CHANNELS)) {
    config = addRobustChannel(config, target);
  }
  assert.equal(config.design.config.channels.length, Object.keys(ROBUST_CHANNELS).length);

  const gravityIndex = config.design.config.channels.findIndex((channel) => channel.target === "gravity");
  config = updateRobustChannel(config, gravityIndex, {
    distribution: "bounded_uniform",
    preset: "high",
  });
  assert.deepEqual(config.design.config.channels[gravityIndex].magnitude, {
    low: -1.5,
    high: 1.5,
  });
  config = setRobustMagnitude(config, gravityIndex, "high", 2.25);
  assert.equal(config.design.config.channels[gravityIndex].preset, "custom");
  assert.equal(config.design.config.channels[gravityIndex].magnitude.high, 2.25);
  config = updateRobustChannel(config, gravityIndex, { preset: "custom" });
  assert.equal(config.design.config.channels[gravityIndex].magnitude.high, 2.25);
  const poseIndex = config.design.config.channels.findIndex((channel) => channel.target === "initial_hand_pose");
  config = setRobustMagnitude(config, poseIndex, "y_mm", 37);
  config = updateRobustChannel(config, poseIndex, { distribution: "bounded_uniform" });
  assert.equal(config.design.config.channels[poseIndex].magnitude.y_mm, 37);

  for (const target of Object.keys(ROBUST_CHANNELS).slice(1)) {
    config = removeRobustChannel(config, target);
  }
  assert.equal(config.design.config.channels.length, 1);
  const finalTarget = config.design.config.channels[0].target;
  assert.equal(removeRobustChannel(config, finalTarget).design.config.channels.length, 1);
});

test("standard variants keep task mobility consistent with their meaning", async () => {
  const original = await readJson("contract/examples/standard.json");
  const restricted = applyStandardVariant(original, "restricted_range");
  assert.equal(restricted.task.embodiment.mobility, "static_partition");
  assert.ok(restricted.task.embodiment.hands.every((hand) => !hand.mobile));

  const mobility = applyStandardVariant(restricted, "mobility");
  assert.equal(mobility.task.embodiment.mobility, "custom");
  assert.deepEqual(mobility.task.embodiment.hands.map((hand) => hand.mobile), [true, false]);
  const changed = setStandardHandMobile(mobility, 1, true);
  assert.deepEqual(changed.task.embodiment.hands.map((hand) => hand.mobile), [true, true]);
  assert.deepEqual(applyStandardVariant(changed, "mobility").task.embodiment.hands.map((hand) => hand.mobile), [true, true]);

  const general = applyStandardVariant(changed, "general");
  assert.equal(general.task.embodiment.mobility, "unrestricted");
});

test("algorithm choices are track-specific and set the matching MARL critic", async () => {
  const marl = await readJson("contract/examples/marl.json");
  const choices = algorithmsForTrack("marl").map((item) => item.name);
  assert.deepEqual(choices, ["ippo", "mappo", "mappo-own-critic", "ippo-rllib-module"]);
  assert.ok(!choices.includes("happo"));
  assert.ok(algorithmsForTrack("marl", true).some((item) => item.name === "happo"));

  const ablation = selectAlgorithm(marl, "mappo-own-critic");
  assert.equal(ablation.design.config.critic_mode, "own_observation_ablation");
  assert.equal(ablation.algorithm.name, "mappo-own-critic");
  assert.equal(selectAlgorithm(ablation, "ippo").design.config.critic_mode, "local");
  assert.equal(selectAlgorithm(ablation, "mappo").design.config.critic_mode, "centralized");
  ablation.algorithm.parameters.learning_rate = 0.0001;
  assert.equal(selectAlgorithm(ablation, "mappo-own-critic").algorithm.parameters.learning_rate, 0.0001);
  assert.ok(!algorithmsForTrack("safety").some((item) => item.name === "ppo_lagrangian"));
});

test("draft list parsing accepts trailing separators", () => {
  assert.deepEqual(parseIntegerList("0, 1, 2, "), [0, 1, 2]);
  assert.deepEqual(parseStringList("marl, winter, "), ["marl", "winter"]);
});

test("tracking provider changes always produce a compatible mode", async () => {
  const original = await readJson("contract/examples/marl.json");
  const disabled = applyTrackingProvider(original, "none");
  assert.equal(disabled.tracking.mode, "disabled");
  const enabled = applyTrackingProvider(disabled, "wandb");
  assert.equal(enabled.tracking.mode, "offline");
});
