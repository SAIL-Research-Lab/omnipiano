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
const { makeCustomMarl, withUiDefaults } = await server.ssrLoadModule("/src/config.js");
const standard = withUiDefaults(JSON.parse(await readFile(new URL("../../omnipiano/experiments/examples/standard.json", import.meta.url))));
const robust = withUiDefaults(JSON.parse(await readFile(new URL("../../omnipiano/experiments/examples/robust.json", import.meta.url))));
const safety = withUiDefaults(JSON.parse(await readFile(new URL("../../omnipiano/experiments/examples/safety.json", import.meta.url))));
const marl = withUiDefaults(JSON.parse(await readFile(new URL("../../omnipiano/experiments/examples/marl.json", import.meta.url))));
const customMarl = makeCustomMarl(marl);
const panel = (id, open = []) => ({ advanced: open.includes(id), onToggleAdvanced: () => {} });
const noop = () => {};

test.after(async () => server.close());

test("the composer remains one scrollable six-section page", () => {
  const html = renderToStaticMarkup(createElement(App));
  assert.match(html, /Choose a track/);
  assert.equal((html.match(/class="workflow-section"/g) ?? []).length, 6);
  for (const title of ["Experiment", "Piano task", "Track design", "Algorithm", "Train &amp; evaluate", "Review &amp; launch"]) {
    assert.match(html, new RegExp(title));
  }
  assert.match(html, /Unified JSON · v3/);
});

test("seeds stay in run settings and advanced controls stay local", () => {
  const identity = renderToStaticMarkup(createElement(ExperimentSection, {
    config: marl, track: "marl", onTrack: noop, set: noop, panel: (id) => panel(id),
  }));
  assert.doesNotMatch(identity, /Seeds/);
  assert.match(identity, /Experiment name/);
  assert.match(identity, /field--prominent/);
  assert.doesNotMatch(identity, /Experiment details|Description|Tags|Basic|Advanced/);
  assert.doesNotMatch(identity, /Nominal control|Perturbed tasks|Constrained learning|Multi-agent coordination/);
  const basic = renderToStaticMarkup(createElement(RunSection, {
    config: marl, set: noop, update: noop, panel: (id) => panel(id),
  }));
  assert.match(basic, /Seeds/);
  assert.doesNotMatch(basic, /Output root/);
  const advanced = renderToStaticMarkup(createElement(RunSection, {
    config: marl, set: noop, update: noop, panel: (id) => panel(id, ["run.resources"]),
  }));
  assert.match(advanced, /Output root/);
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

test("registered and custom MARL tasks expose the correct controls", () => {
  const registered = renderToStaticMarkup(createElement(TaskSection, {
    config: marl, environments: [{ id: marl.task.environment_id, label: marl.task.environment_id }],
    set: noop, update: noop,
  }));
  assert.match(registered, /Registered environment/);
  assert.doesNotMatch(registered, /Number of hands/);
  assert.doesNotMatch(registered, /Basic|Advanced/);

  const custom = renderToStaticMarkup(createElement(DesignSection, {
    config: customMarl, environments: [], set: noop, update: noop, panel: (id) => panel(id),
  }));
  assert.match(custom, /Hands \/ agents/);
  assert.match(custom, /Number of Agents/);
  assert.match(custom, /Hand assignment/);
  assert.match(custom, /Action &amp; observation ranges/);
  assert.doesNotMatch(custom, /primo|secondo/);
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
