import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import react from "@vitejs/plugin-react";
import { createServer } from "vite";

const server = await createServer({
  configFile: false,
  plugins: [react()],
  server: { middlewareMode: true, hmr: false, ws: false, fs: { allow: [".."] } },
});
const { default: App } = await server.ssrLoadModule("/src/App.jsx");
const { AlgorithmSection, DesignSection, ExperimentSection, RunSection, TaskSection } = await server.ssrLoadModule("/src/sections.jsx");
const { makeCustomMarl, stepsForTrack, withUiDefaults } = await server.ssrLoadModule("/src/config.js");
const standard = withUiDefaults(JSON.parse(await readFile(new URL("../../omnipiano/experiments/examples/standard.json", import.meta.url))));
const robust = withUiDefaults(JSON.parse(await readFile(new URL("../../omnipiano/experiments/examples/robust.json", import.meta.url))));
const safety = withUiDefaults(JSON.parse(await readFile(new URL("../../omnipiano/experiments/examples/safety.json", import.meta.url))));
const marl = withUiDefaults(JSON.parse(await readFile(new URL("../../omnipiano/experiments/examples/marl.json", import.meta.url))));
const customMarl = makeCustomMarl(marl);
const panel = (id, open = []) => ({ advanced: open.includes(id), onToggleAdvanced: () => {} });
const noop = () => {};

test.after(async () => server.close());

test("all tracks use focused four-step workflows", () => {
  const html = renderToStaticMarkup(createElement(App));
  assert.match(html, /Choose a track/);
  assert.equal((html.match(/class="workflow-section"/g) ?? []).length, 4);
  for (const title of ["Experiment", "Task design", "Algorithm", "Train &amp; evaluate"]) {
    assert.match(html, new RegExp(title));
  }
  assert.doesNotMatch(html, /Piano task|Track design/);
  for (const track of ["standard", "robust", "safety"]) {
    assert.deepEqual(stepsForTrack(track).map(({ title }) => title), [
      "Experiment", "Piano task", "Algorithm", "Train & evaluate",
    ]);
  }
  assert.doesNotMatch(html, /Review &amp; launch|Validate &amp; preview|Prepare files/);
  assert.match(html, /Download JSON/);
  assert.match(html, /Launch runs/);
  assert.match(html, /Unified JSON · v3/);
});

test("training controls share one compact panel with a comprehensive advanced JSON editor", () => {
  const identity = renderToStaticMarkup(createElement(ExperimentSection, {
    config: marl, track: "marl", onTrack: noop, set: noop, panel: (id) => panel(id),
  }));
  assert.doesNotMatch(identity, /Seeds/);
  assert.match(identity, /Experiment name/);
  assert.match(identity, /field--prominent/);
  assert.doesNotMatch(identity, /Experiment details|Description|Tags|Basic|Advanced/);
  assert.doesNotMatch(identity, /Nominal control|Perturbed tasks|Constrained learning|Multi-agent coordination/);
  const basic = renderToStaticMarkup(createElement(RunSection, {
    config: marl, track: "marl", set: noop, update: noop, panel: (id) => panel(id),
  }));
  assert.match(basic, /Seeds/);
  assert.match(basic, /Training configuration/);
  assert.match(basic, /Total environment steps/);
  assert.match(basic, /Evaluation episodes/);
  assert.match(basic, />Device</);
  assert.doesNotMatch(basic, /Output root/);
  assert.doesNotMatch(basic, /Advanced training configuration/);
  const advanced = renderToStaticMarkup(createElement(RunSection, {
    config: marl, track: "marl", set: noop, update: noop, panel: (id) => panel(id, ["run"]),
  }));
  assert.match(advanced, /Advanced training configuration/);
  assert.match(advanced, /aria-label="evaluation.seed_offset"/);
  assert.match(advanced, /aria-label="video.camera"/);
  assert.match(advanced, /aria-label="compute.num_workers"/);
  assert.match(advanced, /aria-label="tracking.project"/);
  assert.match(advanced, /aria-label="output.root"/);
  assert.match(advanced, /parameter-json__readonly runtime-parameter-json__readonly/);
  assert.doesNotMatch(advanced, /Compute, tracking &amp; output|Run settings/);
});

test("runtime JSON only enables fields consumed by each track adapter", () => {
  const standardHtml = renderToStaticMarkup(createElement(RunSection, {
    config: standard, track: "standard", set: noop, update: noop, panel: (id) => panel(id, ["run"]),
  }));
  assert.match(standardHtml, /aria-label="compute.num_envs"/);
  assert.match(standardHtml, /aria-label="tracking.provider"/);
  assert.match(standardHtml, /aria-label="output.root"/);
  assert.doesNotMatch(standardHtml, /aria-label="video.enabled"/);
  assert.doesNotMatch(standardHtml, /aria-label="compute.num_workers"/);

  const safetyHtml = renderToStaticMarkup(createElement(RunSection, {
    config: safety, track: "safety", set: noop, update: noop, panel: (id) => panel(id, ["run"]),
  }));
  assert.match(safetyHtml, /Gamma <strong>0.8/);
  assert.match(safetyHtml, /aria-label="tracking.provider"/);
  assert.match(safetyHtml, /aria-label="output.root"/);
  assert.doesNotMatch(safetyHtml, /aria-label="compute.num_envs"/);
  assert.doesNotMatch(safetyHtml, /aria-label="evaluation.seed_offset"/);
});

test("standard, robust, and safety tasks are ID-only environment selectors", () => {
  for (const config of [standard, robust, safety]) {
    const current = config.task.environment_id;
    const alternate = `${current}-alternate`;
    const html = renderToStaticMarkup(createElement(TaskSection, {
      config,
      environments: [
        { id: current, label: "Decorated environment label" },
        { id: alternate, label: "Another decorated label" },
      ],
      set: noop,
      update: noop,
    }));
    assert.match(html, /<select/);
    assert.match(html, new RegExp(current));
    assert.match(html, new RegExp(alternate));
    assert.doesNotMatch(html, />Task<|>Choose|Decorated|>Environment ID<|>Basic<|>Advanced</);
  }
});

test("MARL removes registered tasks and folds its custom task controls into Task design", () => {
  const removedTaskSection = renderToStaticMarkup(createElement(TaskSection, {
    config: marl, environments: [{ id: marl.task.environment_id, label: marl.task.environment_id }],
    set: noop, update: noop,
  }));
  assert.equal(removedTaskSection, "");

  const custom = renderToStaticMarkup(createElement(DesignSection, {
    config: customMarl, environments: [], set: noop, update: noop, panel: (id) => panel(id),
  }));
  assert.match(custom, /General configuration/);
  assert.match(custom, />Piece</);
  assert.match(custom, /Number of hands/);
  assert.match(custom, /class="choice-block"><span class="field__label">Number of hands/);
  assert.match(custom, /Number of agents/);
  assert.match(custom, /Sustain owner/);
  assert.match(custom, /Hand assignment &amp; ranges/);
  assert.match(custom, /piano-workbench/);
  assert.equal((custom.match(/class="piano-hand piano-hand--/g) ?? []).length, 4);
  assert.equal((custom.match(/class="piano-keyboard__black-key"/g) ?? []).length, 36);
  assert.match(custom, /H1/);
  assert.match(custom, /Agent 1/);
  assert.match(custom, /aria-label="action start"/);
  assert.match(custom, /aria-label="action end"/);
  assert.match(custom, /aria-label="observation start"/);
  assert.match(custom, /aria-label="observation end"/);
  assert.equal((custom.match(/role="slider"/g) ?? []).length, 8);
  assert.doesNotMatch(custom, /Action keys start|Action keys end|Observation keys start|Observation keys end/);
  assert.doesNotMatch(custom, /Hands \/ agents|Action overlap|Observation overlap|Critic/);
  assert.doesNotMatch(custom, /Maximum 4|Moving an agent|Keys 1–22|wrist key range|visible keyboard state|Action ranges constrain/);
  assert.doesNotMatch(custom, /Registered environment|Custom MARL task|Task source|Collision penalty|hand-overview|Agent 1 name|Action &amp; observation ranges|Show advanced settings/);
  assert.doesNotMatch(custom, /primo|secondo/);
});

test("non-MARL tracks do not render a design section", () => {
  for (const config of [standard, robust, safety]) {
    const html = renderToStaticMarkup(createElement(DesignSection, {
      config, set: noop, update: noop, panel: (id) => panel(id),
    }));
    assert.equal(html, "");
  }
});

test("advanced algorithm settings start with an editable reference template", () => {
  const config = {
    ...robust,
    algorithm: { name: "sac", preset: "default", parameters: {} },
  };
  const html = renderToStaticMarkup(createElement(AlgorithmSection, {
    config,
    algorithms: [{
      name: "sac",
      label: "SAC",
      configurable_parameters: [
        "net_arch", "batch_size", "buffer_size", "learning_starts", "tau", "utd", "learning_rate",
      ],
    }],
    update: noop,
    panel: (id) => panel(id, ["algorithm"]),
  }));
  assert.match(html, />Parameters</);
  assert.match(html, /aria-label="learning_rate" value="0.0003"/);
  assert.match(html, /aria-label="buffer_size" value="1000000"/);
  assert.match(html, /aria-label="net_arch" value="\[256,256\]"/);
  assert.equal((html.match(/class="parameter-json"/g) ?? []).length, 1);
  assert.doesNotMatch(html, /Editable parameters|Read-only parameters|Reference values|Allowed keys/);
});

test("unsupported Safe RL profiles render as read-only parameters", () => {
  const config = {
    ...safety,
    algorithm: { name: "P3O", preset: "default", parameters: {} },
  };
  const html = renderToStaticMarkup(createElement(AlgorithmSection, {
    config,
    algorithms: [{
      name: "P3O",
      label: "P3O",
      configurable_parameters: [],
    }],
    update: noop,
    panel: (id) => panel(id, ["algorithm"]),
  }));
  assert.match(html, />Parameters</);
  assert.match(html, /parameter-json__readonly/);
  assert.doesNotMatch(html, /parameter-json__input|Editable parameters|Read-only parameters/);
  assert.equal((html.match(/class="parameter-json"/g) ?? []).length, 1);
  for (const name of ["algo_cfgs", "update_iters", "target_kl", "kappa", "penalty_coef", "model_cfgs", "actor", "critic"]) {
    assert.match(html, new RegExp(name));
  }
});
