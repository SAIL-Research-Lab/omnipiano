import { useEffect, useMemo, useRef, useState } from "react";

import standardExample from "../contract/examples/standard.json";
import robustExample from "../contract/examples/robust.json";
import safetyExample from "../contract/examples/safety.json";
import marlExample from "../contract/examples/marl.json";

import { cloneConfig, setAtPath, STEPS, TRACKS, upgradeImportedConfig } from "./config";
import {
  AlgorithmSection,
  DesignSection,
  ExperimentSection,
  ReviewSection,
  RunSection,
  TaskSection,
} from "./sections";
import { validateExperiment } from "./validation";

const EXAMPLES = {
  standard: standardExample,
  robust: robustExample,
  safety: safetyExample,
  marl: marlExample,
};

function JsonPreview({ config, errors, warnings, collapsed, onToggle, onCopy }) {
  return (
    <aside className={`json-panel ${collapsed ? "json-panel--collapsed" : ""}`}>
      <div className="json-panel__header">
        <div>
          <strong>Experiment JSON</strong>
        </div>
        <div className="json-panel__actions">
          <button type="button" onClick={onCopy}>Copy</button>
          <button type="button" onClick={onToggle}>{collapsed ? "Show" : "Hide"}</button>
        </div>
      </div>
      {!collapsed && (
        <>
          <div className={`schema-status ${errors.length ? "schema-status--error" : ""}`}>
            <span>{errors.length ? "!" : "✓"}</span>
            {errors.length
              ? `${errors.length} configuration issue${errors.length === 1 ? "" : "s"}`
              : warnings.length
                ? `Valid with ${warnings.length} warning${warnings.length === 1 ? "" : "s"}`
                : "Configuration valid"}
          </div>
          <pre>{JSON.stringify(config, null, 2)}</pre>
        </>
      )}
    </aside>
  );
}

function StepContent({ step, config, track, onTrack, set, update, panel, errors, warnings, onCopy, onDownload, copyLabel }) {
  if (step === "experiment") {
    return <ExperimentSection config={config} track={track} onTrack={onTrack} set={set} panel={panel} />;
  }
  if (step === "task") return <TaskSection config={config} set={set} update={update} panel={panel} />;
  if (step === "design") {
    return <DesignSection config={config} set={set} update={update} panel={panel} />;
  }
  if (step === "algorithm") return <AlgorithmSection config={config} update={update} panel={panel} />;
  if (step === "run") {
    return <RunSection config={config} set={set} update={update} panel={panel} />;
  }
  return (
    <ReviewSection
      config={config}
      errors={errors}
      warnings={warnings}
      onCopy={onCopy}
      onDownload={onDownload}
      copyLabel={copyLabel}
    />
  );
}

export default function App() {
  const [track, setTrack] = useState("marl");
  const [activeStep, setActiveStep] = useState(STEPS[0].id);
  const [advancedPanels, setAdvancedPanels] = useState({});
  const [previewCollapsed, setPreviewCollapsed] = useState(false);
  const [copyLabel, setCopyLabel] = useState("Copy JSON");
  const [importMessage, setImportMessage] = useState(null);
  const fileInput = useRef(null);
  const [configs, setConfigs] = useState(() => Object.fromEntries(
    Object.entries(EXAMPLES).map(([key, value]) => [key, cloneConfig(value)]),
  ));

  const config = configs[track];
  const validation = useMemo(() => validateExperiment(config), [config]);
  const { errors, warnings } = validation;
  const trackInfo = TRACKS.find((item) => item.id === track);
  const panel = (id) => {
    const key = `${track}:${id}`;
    return {
      advanced: Boolean(advancedPanels[key]),
      onToggleAdvanced: () => setAdvancedPanels((previous) => ({
        ...previous,
        [key]: !previous[key],
      })),
    };
  };

  useEffect(() => {
    let frame = null;
    const updateActiveStep = () => {
      const anchor = Math.min(180, window.innerHeight * 0.25);
      const sections = STEPS
        .map((step) => document.getElementById(`step-${step.id}`))
        .filter(Boolean);
      let current = sections[0];
      sections.forEach((section) => {
        if (section.getBoundingClientRect().top <= anchor) current = section;
      });
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
        current = sections.at(-1);
      }
      if (current) setActiveStep(current.dataset.step);
    };
    const scheduleUpdate = () => {
      if (frame !== null) window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(updateActiveStep);
    };
    updateActiveStep();
    window.addEventListener("scroll", scheduleUpdate, { passive: true });
    window.addEventListener("resize", scheduleUpdate);
    return () => {
      if (frame !== null) window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", scheduleUpdate);
      window.removeEventListener("resize", scheduleUpdate);
    };
  }, []);

  function set(path, value) {
    setConfigs((previous) => ({
      ...previous,
      [track]: setAtPath(previous[track], path, value),
    }));
  }

  function update(transform) {
    setConfigs((previous) => ({
      ...previous,
      [track]: transform(previous[track]),
    }));
  }

  function changeTrack(nextTrack) {
    setTrack(nextTrack);
  }

  function resetTrack() {
    setConfigs((previous) => ({
      ...previous,
      [track]: cloneConfig(EXAMPLES[track]),
    }));
    setImportMessage(null);
  }

  async function importJson(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    try {
      const imported = upgradeImportedConfig(JSON.parse(await file.text()));
      const importedTrack = imported?.design?.type;
      if (!(importedTrack in EXAMPLES)) {
        throw new Error("This is not an OmniPiano canonical UI configuration.");
      }
      const result = validateExperiment(imported);
      if (result.errors.length) {
        throw new Error(result.errors[0]);
      }
      setConfigs((previous) => ({ ...previous, [importedTrack]: imported }));
      setTrack(importedTrack);
      setImportMessage({ type: "success", text: `Loaded ${file.name}` });
    } catch (error) {
      setImportMessage({ type: "error", text: error.message });
    }
  }

  async function copyJson() {
    try {
      await navigator.clipboard.writeText(JSON.stringify(config, null, 2));
      setCopyLabel("Copied");
      window.setTimeout(() => setCopyLabel("Copy JSON"), 1400);
    } catch {
      setCopyLabel("Copy failed");
    }
  }

  function downloadJson() {
    if (errors.length) return;
    const blob = new Blob([`${JSON.stringify(config, null, 2)}\n`], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${config.experiment.name}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className={`app app--${track}`}>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="OmniPiano Experiment Composer">
          <span className="brand__keys" aria-hidden="true"><i /><i /><i /><i /><i /></span>
          <span><strong>OmniPiano</strong><small>Experiment Composer</small></span>
        </a>
        <div className="topbar__context">
          <input
            ref={fileInput}
            className="visually-hidden"
            type="file"
            accept="application/json,.json"
            onChange={importJson}
          />
          <button className="reset-button" type="button" onClick={() => fileInput.current?.click()}>
            Import JSON
          </button>
          <button className="reset-button" type="button" onClick={resetTrack}>Reset track</button>
        </div>
      </header>

      {importMessage && (
        <div className={`import-message import-message--${importMessage.type}`} role="status">
          <span>{importMessage.type === "success" ? "✓" : "!"}</span>
          {importMessage.text}
          <button type="button" aria-label="Dismiss import message" onClick={() => setImportMessage(null)}>×</button>
        </div>
      )}

      <div className="workspace" id="top">
        <nav className="step-nav" aria-label="Experiment configuration steps">
          <div className="step-nav__track">
            <span className={`track-card__mark track-card__mark--${track}`}>{trackInfo.short}</span>
            <div><strong>{trackInfo.title}</strong></div>
          </div>
          <ol>
            {STEPS.map((step) => (
              <li key={step.id}>
                <a
                  href={`#step-${step.id}`}
                  className={activeStep === step.id ? "active" : ""}
                  aria-current={activeStep === step.id ? "step" : undefined}
                >
                  <span>{step.number}</span>
                  <strong>{step.title}</strong>
                </a>
              </li>
            ))}
          </ol>
        </nav>

        <main className="editor">
          {STEPS.map((step) => (
            <section
              className="workflow-section"
              data-step={step.id}
              id={`step-${step.id}`}
              aria-labelledby={`step-${step.id}-title`}
              key={step.id}
            >
              <div className="workflow-section__heading">
                <span>{step.number}</span>
                <h1 id={`step-${step.id}-title`}>{step.title}</h1>
              </div>
              <StepContent
                step={step.id}
                config={config}
                track={track}
                onTrack={changeTrack}
                set={set}
                update={update}
                panel={panel}
                errors={errors}
                warnings={warnings}
                onCopy={copyJson}
                onDownload={downloadJson}
                copyLabel={copyLabel}
              />
            </section>
          ))}
        </main>

        <JsonPreview
          config={config}
          errors={errors}
          warnings={warnings}
          collapsed={previewCollapsed}
          onToggle={() => setPreviewCollapsed((value) => !value)}
          onCopy={copyJson}
        />
      </div>
    </div>
  );
}
