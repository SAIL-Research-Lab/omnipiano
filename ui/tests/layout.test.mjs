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
const { DesignSection, ExperimentSection, RunSection, TaskSection } = await server.ssrLoadModule("/src/sections.jsx");
const { makeCustomMarl, withUiDefaults } = await server.ssrLoadModule("/src/config.js");
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

test("registered and custom MARL tasks expose the correct controls", () => {
  const registered = renderToStaticMarkup(createElement(TaskSection, {
    config: marl, environments: [{ id: marl.task.environment_id, label: marl.task.environment_id }],
    set: noop, update: noop, panel: (id) => panel(id),
  }));
  assert.match(registered, /Registered environment/);
  assert.doesNotMatch(registered, /Number of hands/);

  const custom = renderToStaticMarkup(createElement(DesignSection, {
    config: customMarl, environments: [], set: noop, update: noop, panel: (id) => panel(id),
  }));
  assert.match(custom, /Hands \/ agents/);
  assert.match(custom, /Number of Agents/);
  assert.match(custom, /Hand assignment/);
  assert.match(custom, /Action &amp; observation ranges/);
  assert.doesNotMatch(custom, /primo|secondo/);
});
