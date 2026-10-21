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
  server: { middlewareMode: true, hmr: false, ws: false },
});
const { default: App } = await server.ssrLoadModule("/src/App.jsx");
const {
  AlgorithmSection,
  DesignSection,
  ExperimentSection,
  RunSection,
  TaskSection,
} = await server.ssrLoadModule("/src/sections.jsx");
const marl = JSON.parse(await readFile(new URL("../contract/examples/marl.json", import.meta.url)));
const robust = JSON.parse(await readFile(new URL("../contract/examples/robust.json", import.meta.url)));
const safety = JSON.parse(await readFile(new URL("../contract/examples/safety.json", import.meta.url)));
const panel = (id, open = []) => ({ advanced: open.includes(id), onToggleAdvanced: () => {} });
const noop = () => {};

test.after(async () => server.close());

test("default page has no global mode, status, or configuration banner", () => {
  const html = renderToStaticMarkup(createElement(App));
  assert.doesNotMatch(html, /status-badge|mode-button|class="notice"/);
  assert.match(html, /Choose a track/);
  assert.doesNotMatch(html, /Benchmark family|The shared piano task stays intact/);
  assert.equal((html.match(/class="workflow-section"/g) ?? []).length, 6);
  for (const title of ["Experiment", "Piano task", "Track design", "Algorithm", "Train &amp; evaluate", "Review &amp; export"]) {
    assert.match(html, new RegExp(title));
  }
  assert.doesNotMatch(html, /editor__footer|>Back<|>Continue</);
});

test("seeds belong to Run and advanced controls open only in their own panel", () => {
  const identity = renderToStaticMarkup(createElement(ExperimentSection, {
    config: marl, track: "marl", onTrack: noop, set: noop,
    panel: (id) => panel(id),
  }));
  assert.doesNotMatch(identity, /Seeds/);
  assert.match(identity, /Show advanced settings for Experiment details/);

  const basic = renderToStaticMarkup(createElement(RunSection, {
    config: marl, set: noop, update: noop, panel: (id) => panel(id),
  }));
  assert.match(basic, /Seeds/);
  assert.doesNotMatch(basic, /Checkpoint frequency|CPUs per runner/);

  const trainingAdvanced = renderToStaticMarkup(createElement(RunSection, {
    config: marl, set: noop, update: noop,
    panel: (id) => panel(id, ["run.training"]),
  }));
  assert.match(trainingAdvanced, /Checkpoint frequency/);
  assert.doesNotMatch(trainingAdvanced, /CPUs per runner/);

  const resourcesAdvanced = renderToStaticMarkup(createElement(RunSection, {
    config: marl, set: noop, update: noop,
    panel: (id) => panel(id, ["run.resources"]),
  }));
  assert.match(resourcesAdvanced, /CPUs per runner/);
  assert.doesNotMatch(resourcesAdvanced, /Checkpoint frequency/);
});

test("task panels expose their own advanced controls", () => {
  const html = renderToStaticMarkup(createElement(TaskSection, {
    config: marl, set: noop, update: noop,
    panel: (id) => panel(id, ["task.music"]),
  }));
  assert.match(html, /Tempo stretch/);
  assert.doesNotMatch(html, /Lookahead steps/);
  assert.match(html, /Show advanced settings for Hands/);
});

test("track panels and algorithm choices omit support labels", () => {
  const robustHtml = renderToStaticMarkup(createElement(DesignSection, {
    config: robust, set: noop, update: noop, panel: (id) => panel(id),
  }));
  assert.match(robustHtml, /Show advanced settings for Action noise/);
  assert.doesNotMatch(robustHtml, /support-chip|Awaiting merge|Experimental/);
  assert.doesNotMatch(robustHtml, /Robustness evaluation/);
  const robustRun = renderToStaticMarkup(createElement(RunSection, {
    config: robust, set: noop, update: noop, panel: (id) => panel(id),
  }));
  assert.match(robustRun, /Robustness evaluation/);

  const safetyHtml = renderToStaticMarkup(createElement(DesignSection, {
    config: safety, set: noop, update: noop, panel: (id) => panel(id),
  }));
  assert.doesNotMatch(safetyHtml, /Cost coefficient/);
  assert.match(safetyHtml, /Show advanced settings for Safety cost/);

  const marlHtml = renderToStaticMarkup(createElement(DesignSection, {
    config: marl, set: noop, update: noop, panel: (id) => panel(id),
  }));
  assert.match(marlHtml, /class="marl-stats"/);
  assert.match(marlHtml, /Hands \/ agents/);
  assert.match(marlHtml, /Action overlap/);
  assert.match(marlHtml, /Observation overlap/);
  assert.match(marlHtml, /Number of Agents/);
  assert.match(marlHtml, /Hand assignment/);
  assert.match(marlHtml, /Action &amp; observation ranges/);
  assert.doesNotMatch(marlHtml, /SCHO|primo|secondo/);
  assert.equal((marlHtml.match(/class="section-card"/g) ?? []).length, 3);
  assert.doesNotMatch(marlHtml, /Teammate hand visibility/);
  const marlRangeAdvanced = renderToStaticMarkup(createElement(DesignSection, {
    config: marl, set: noop, update: noop,
    panel: (id) => panel(id, ["marl.ranges"]),
  }));
  assert.match(marlRangeAdvanced, /Teammate hand visibility/);

  const algorithmHtml = renderToStaticMarkup(createElement(AlgorithmSection, {
    config: marl, update: noop, panel: (id) => panel(id),
  }));
  assert.match(algorithmHtml, /MAPPO · own-critic ablation/);
  assert.doesNotMatch(algorithmHtml, />Experimental<|>Implemented<|>Template in repo/);
  assert.doesNotMatch(algorithmHtml, /Parameters/);
});
