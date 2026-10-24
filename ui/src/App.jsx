import { Fragment, useEffect, useMemo, useRef, useState } from "react";

import standardExample from "../../omnipiano/experiments/examples/standard.json";
import robustExample from "../../omnipiano/experiments/examples/robust.json";
import safetyExample from "../../omnipiano/experiments/examples/safety.json";
import marlExample from "../../omnipiano/experiments/examples/marl.json";

import { getCatalog, getEnvironments, getRuns, postConfig, stopRun } from "./api";
import { cloneConfig, environmentNameForConfig, makeCustomMarl, setAtPath, TRACKS, visibleStepsForTrack, withUiDefaults } from "./config";
import {
  AlgorithmSection,
  DesignSection,
  EnvironmentExportSection,
  ExperimentSection,
  ReferenceBaselineToggle,
  RunSection,
  TaskSection,
} from "./sections";
import { validateExperiment } from "./validation";

const RAW_EXAMPLES = { standard: standardExample, robust: robustExample, safety: safetyExample, marl: marlExample };
const EXAMPLES = Object.fromEntries(Object.entries(RAW_EXAMPLES).map(([key, value]) => [key, withUiDefaults(value)]));
EXAMPLES.marl = makeCustomMarl(EXAMPLES.marl);
Object.values(EXAMPLES).forEach((example) => { example.experiment.name = environmentNameForConfig(example); });

function JsonPreview({ config, errors, collapsed, onToggle, onCopy, onDownload, onLaunch, onStop, backend, copyLabel }) {
  const backendError = backend.type === "error";
  const statusError = errors.length > 0 || backendError;
  const statusMessage = errors.length
    ? `${errors.length} JSON issue${errors.length === 1 ? "" : "s"}`
    : backend.message || "Schema valid";
  return (
    <aside id="unified-json-panel" className={`json-panel ${collapsed ? "json-panel--collapsed" : ""}`}>
      <div className="json-panel__header">
        <strong>Unified JSON · v3</strong>
        <div className="json-panel__actions">
          <button type="button" onClick={onCopy}>{copyLabel}</button>
          <button type="button" onClick={onToggle}>{collapsed ? "Show" : "Hide"}</button>
        </div>
      </div>
      {!collapsed && (
        <>
          <div className={`schema-status ${statusError ? "schema-status--error" : ""}`}>
            <span>{statusError ? "!" : "✓"}</span>
            {statusMessage}
          </div>
          <pre>{JSON.stringify(config, null, 2)}</pre>
          {backend.runs?.length > 0 && (
            <div className="json-panel__runs">
              {backend.runs.map((run) => (
                <div key={run.run_id}>
                  <span>{run.run_id}</span>
                  <strong>{run.process?.state ?? "prepared"}</strong>
                  {run.process?.state === "running" && <button type="button" onClick={() => onStop(run.run_id)}>Stop</button>}
                </div>
              ))}
            </div>
          )}
          <div className="json-panel__launch-actions">
            <button type="button" onClick={onDownload} disabled={errors.length > 0}>Download JSON</button>
            <button type="button" className="json-panel__launch" onClick={onLaunch} disabled={errors.length > 0 || backend.busy}>{backend.busy ? "Working…" : "Launch runs"}</button>
          </div>
        </>
      )}
    </aside>
  );
}

function StepContent(props) {
  const { step } = props;
  if (step === "experiment") return <ExperimentSection {...props} />;
  if (step === "task") return (
    <>
      <TaskSection {...props} />
      <EnvironmentExportSection {...props} />
    </>
  );
  if (step === "design") return (
    <>
      <DesignSection {...props} />
      <EnvironmentExportSection {...props} />
    </>
  );
  if (step === "algorithm") return <AlgorithmSection {...props} />;
  return <RunSection {...props} />;
}

export default function App() {
  const [track, setTrack] = useState("marl");
  const [activeStep, setActiveStep] = useState("experiment");
  const [advancedPanels, setAdvancedPanels] = useState({});
  const [previewCollapsed, setPreviewCollapsed] = useState(false);
  const [showBaselines, setShowBaselines] = useState(false);
  const [copyLabel, setCopyLabel] = useState("Copy JSON");
  const [message, setMessage] = useState(null);
  const [catalog, setCatalog] = useState(null);
  const [environments, setEnvironments] = useState({});
  const [backend, setBackend] = useState({ type: "idle", message: "", busy: false, runs: [] });
  const fileInput = useRef(null);
  const [configs, setConfigs] = useState(() => Object.fromEntries(
    Object.entries(EXAMPLES).map(([key, value]) => [key, cloneConfig(value)]),
  ));

  const config = configs[track];
  const steps = visibleStepsForTrack(track, showBaselines);
  const validation = useMemo(() => validateExperiment(config), [config]);
  const { errors } = validation;
  const trackInfo = TRACKS.find((item) => item.id === track);
  const algorithms = catalog?.tracks?.[track]?.algorithms ?? [{
    name: config.algorithm.name,
    label: config.algorithm.name.toUpperCase(),
    configurable_parameters: Object.keys(config.algorithm.parameters),
  }];

  const panel = (id) => {
    const key = `${track}:${id}`;
    return {
      advanced: Boolean(advancedPanels[key]),
      onToggleAdvanced: () => setAdvancedPanels((previous) => ({ ...previous, [key]: !previous[key] })),
    };
  };

  useEffect(() => {
    getCatalog().then(setCatalog).catch(() => setBackend((previous) => ({
      ...previous,
      type: "error",
      message: "Local API unavailable. Start it before backend validation or launch.",
    })));
  }, []);

  useEffect(() => {
    if (track === "marl") return undefined;
    getEnvironments(track).then((payload) => setEnvironments((previous) => ({
      ...previous,
      [track]: payload.environments,
    }))).catch(() => {});
    return undefined;
  }, [track]);

  useEffect(() => {
    let frame = null;
    const updateActiveStep = () => {
      const anchor = Math.min(180, window.innerHeight * 0.25);
      const sections = steps.map((step) => document.getElementById(`step-${step.id}`)).filter(Boolean);
      let current = sections[0];
      sections.forEach((section) => { if (section.getBoundingClientRect().top <= anchor) current = section; });
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) current = sections.at(-1);
      if (current) setActiveStep(current.dataset.step);
    };
    const schedule = () => {
      if (frame !== null) window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(updateActiveStep);
    };
    schedule();
    window.addEventListener("scroll", schedule, { passive: true });
    window.addEventListener("resize", schedule);
    return () => {
      if (frame !== null) window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };
  }, [steps]);

  function set(path, value) {
    setConfigs((previous) => ({ ...previous, [track]: setAtPath(previous[track], path, value) }));
    setBackend({ type: "idle", message: "", busy: false, runs: [] });
  }

  function changeTrack(nextTrack) {
    if (nextTrack === "marl") {
      setConfigs((previous) => ({
        ...previous,
        marl: previous.marl.task.source === "custom" ? previous.marl : makeCustomMarl(previous.marl),
      }));
    }
    setTrack(nextTrack);
    setShowBaselines(false);
    setActiveStep("experiment");
    setBackend({ type: "idle", message: "", busy: false, runs: [] });
    setMessage(null);
  }

  function update(transform) {
    setConfigs((previous) => ({ ...previous, [track]: transform(previous[track]) }));
    setBackend({ type: "idle", message: "", busy: false, runs: [] });
  }

  function resetTrack() {
    setConfigs((previous) => ({ ...previous, [track]: cloneConfig(EXAMPLES[track]) }));
    setShowBaselines(false);
    setBackend({ type: "idle", message: "", busy: false, runs: [] });
    setMessage(null);
  }

  async function importJson(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      let imported = withUiDefaults(JSON.parse(await file.text()));
      const importedTrack = imported?.design?.type;
      if (!TRACKS.some((item) => item.id === importedTrack)) throw new Error("Not a unified OmniPiano schema-v3 configuration.");
      if (importedTrack === "marl" && imported.task.source !== "custom") imported = makeCustomMarl(imported);
      const result = validateExperiment(imported);
      if (result.errors.length) throw new Error(result.errors[0]);
      setConfigs((previous) => ({ ...previous, [importedTrack]: imported }));
      setTrack(importedTrack);
      setShowBaselines(false);
      setMessage({ type: "success", text: `Loaded ${file.name}` });
    } catch (reason) {
      setMessage({ type: "error", text: reason.message });
    }
  }

  async function copyJson() {
    try {
      await navigator.clipboard.writeText(`${JSON.stringify(config, null, 2)}\n`);
      setCopyLabel("Copied");
      window.setTimeout(() => setCopyLabel("Copy JSON"), 1400);
    } catch { setCopyLabel("Copy failed"); }
  }

  function downloadJson() {
    if (errors.length) return;
    const blob = new Blob([`${JSON.stringify(config, null, 2)}\n`], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${config.experiment.name}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function callBackend(action) {
    setBackend({ type: "working", message: `${action}…`, busy: true, runs: [] });
    try {
      const payload = await postConfig(action, config);
      let runs = [];
      if (action === "plan") runs = payload.plan.runs;
      if (action === "materialize") runs = payload.materialized.runs;
      if (action === "launch") {
        const status = await getRuns();
        runs = status.runs.filter((run) => run.run_id.startsWith(`${config.experiment.name}--`));
      }
      setBackend({
        type: "success",
        message: action === "plan" ? "Backend validated; launch commands are ready." :
          action === "materialize" ? "Run directories and manifests prepared." : "Training processes started.",
        busy: false,
        runs,
      });
    } catch (reason) {
      setBackend({ type: "error", message: reason.message, busy: false, runs: [] });
    }
  }

  async function launch() {
    if (!window.confirm(`Launch ${config.experiment.seeds.length} run(s) for ${config.experiment.name}?`)) return;
    await callBackend("launch");
  }

  async function stop(runId) {
    try {
      await stopRun(runId);
      const status = await getRuns();
      setBackend((previous) => ({ ...previous, runs: status.runs.filter((run) => run.run_id.startsWith(`${config.experiment.name}--`)) }));
    } catch (reason) {
      setBackend({ type: "error", message: reason.message, busy: false, runs: [] });
    }
  }

  function toggleBaselines() {
    if (!showBaselines) setPreviewCollapsed(false);
    setShowBaselines(!showBaselines);
  }

  const sharedProps = {
    config,
    track,
    onTrack: changeTrack,
    set,
    update,
    panel,
    errors,
    algorithms,
    trackCapability: catalog?.tracks?.[track] ?? null,
    environments: environments[track] ?? [],
    backend,
    onCopy: copyJson,
    onDownload: downloadJson,
    onLaunch: launch,
    onStop: stop,
    showBaselines,
    onToggleBaselines: toggleBaselines,
  };

  return (
    <div className={`app app--${track}`}>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="OmniPiano Experiment Composer">
          <span className="brand__keys" aria-hidden="true"><i /><i /><i /><i /><i /></span>
          <span><strong>OmniPiano</strong><small>Experiment Composer</small></span>
        </a>
        {showBaselines && (
          <div className="topbar__context">
            <input ref={fileInput} className="visually-hidden" type="file" accept="application/json,.json" onChange={importJson} />
            <button className="reset-button" type="button" onClick={() => fileInput.current?.click()}>Import JSON</button>
            <button className="reset-button" type="button" onClick={resetTrack}>Reset track</button>
          </div>
        )}
      </header>
      {message && <div className={`import-message import-message--${message.type}`} role="status"><span>{message.type === "success" ? "✓" : "!"}</span>{message.text}<button type="button" onClick={() => setMessage(null)}>×</button></div>}
      <div className={`workspace ${showBaselines ? "" : "workspace--environment-only"}`.trim()} id="top">
        <nav className="step-nav" aria-label="Experiment configuration steps">
          <div className="step-nav__track"><span className={`track-card__mark track-card__mark--${track}`}>{trackInfo.short}</span><strong>{trackInfo.title}</strong></div>
          <ol>{steps.map((step) => <li key={step.id}><a href={`#step-${step.id}`} className={activeStep === step.id ? "active" : ""}><span>{step.number}</span><strong>{step.title}</strong></a></li>)}</ol>
        </nav>
        <main className="editor">
          {steps.map((step, index) => (
            <Fragment key={step.id}>
              <section className={`workflow-section ${index === 1 ? "workflow-section--before-baselines" : ""}`.trim()} data-step={step.id} id={`step-${step.id}`}>
                <div className="workflow-section__heading"><span>{step.number}</span><h1>{step.title}</h1></div>
                <StepContent step={step.id} {...sharedProps} />
              </section>
              {index === 1 && (
                <div className="reference-baseline-gateway">
                  <ReferenceBaselineToggle expanded={showBaselines} onToggle={toggleBaselines} />
                </div>
              )}
            </Fragment>
          ))}
        </main>
        {showBaselines && (
          <JsonPreview
            config={config}
            errors={errors}
            collapsed={previewCollapsed}
            onToggle={() => setPreviewCollapsed((value) => !value)}
            onCopy={copyJson}
            onDownload={downloadJson}
            onLaunch={launch}
            onStop={stop}
            backend={backend}
            copyLabel={copyLabel}
          />
        )}
      </div>
    </div>
  );
}
