import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import Ajv2020 from "ajv/dist/2020.js";
import {
  applyHandLayout,
  applyMarlAgentCount,
  applyTrackingProvider,
  assignMarlHandOwner,
  makeCustomMarl,
  parameterTemplate,
  parseIntegerList,
  selectAlgorithm,
  withUiDefaults,
} from "../src/config.js";

const ROOT = new URL("../../", import.meta.url);
async function readJson(path) { return JSON.parse(await readFile(new URL(path, ROOT), "utf8")); }

test("all four backend-neutral examples validate after UI defaults", async () => {
  const schema = await readJson("omnipiano/experiments/contract/experiment-v3.schema.json");
  const validate = new Ajv2020({ allErrors: true, strict: false }).compile(schema);
  for (const track of ["standard", "robust", "safety", "marl"]) {
    const document = withUiDefaults(await readJson(`omnipiano/experiments/examples/${track}.json`));
    assert.equal(validate(document), true, JSON.stringify(validate.errors));
    assert.equal(document.schema_version, 3);
  }
});

test("registered MARL can become an exact custom task", async () => {
  const base = withUiDefaults(await readJson("omnipiano/experiments/examples/marl.json"));
  let config = makeCustomMarl(base);
  assert.equal(config.task.source, "custom");
  assert.deepEqual(config.design.config.agents.map((agent) => agent.name), ["agent_1", "agent_2"]);
  config = applyHandLayout(config, 5);
  config = applyMarlAgentCount(config, 3);
  config = assignMarlHandOwner(config, 0, 1);
  const owned = config.design.config.agents.flatMap((agent) => agent.hand_ids);
  assert.deepEqual([...owned].sort((a, b) => a - b), [0, 1, 2, 3, 4]);
  assert.equal(new Set(owned).size, 5);
});

test("algorithm selection clears stale overrides", async () => {
  const base = withUiDefaults(await readJson("omnipiano/experiments/examples/marl.json"));
  base.algorithm.parameters = { learning_rate: 0.1 };
  assert.deepEqual(selectAlgorithm(base, "mappo").algorithm, {
    name: "mappo", preset: "default", parameters: {},
  });
});

test("seed parsing accepts trailing separators", () => {
  assert.deepEqual(parseIntegerList("0, 1, 2, "), [0, 1, 2]);
});

test("tracking provider always produces a compatible mode", async () => {
  const base = withUiDefaults(await readJson("omnipiano/experiments/examples/marl.json"));
  const disabled = applyTrackingProvider(base, "none");
  assert.equal(disabled.run.tracking.mode, "disabled");
  assert.equal(applyTrackingProvider(disabled, "wandb").run.tracking.mode, "offline");
});

test("algorithm parameter templates match the selected backend family", () => {
  assert.deepEqual(
    parameterTemplate("robust", "sac", [
      "net_arch", "batch_size", "buffer_size", "learning_starts", "tau", "utd", "learning_rate",
    ]),
    {
      net_arch: [256, 256],
      batch_size: 256,
      buffer_size: 1_000_000,
      learning_starts: 5_000,
      tau: 0.005,
      utd: 1,
      learning_rate: 0.0003,
    },
  );
  assert.deepEqual(
    parameterTemplate("marl", "mappo", ["learning_rate", "train_batch_size", "hidden_sizes"]),
    { learning_rate: 0.0003, train_batch_size: 4_000, hidden_sizes: [256, 256] },
  );
  assert.deepEqual(parameterTemplate("standard", "ppo", []), {});
});
