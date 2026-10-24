import { useEffect, useState } from "react";

import {
  ChoiceRow,
  ListField,
  NumberField,
  Section,
  SelectField,
  TextField,
  ToggleField,
} from "./fields";
import {
  applyHandLayout,
  applyTrackingProvider,
  makeCustomMarl,
  parameterDisplayTemplate,
  makeRegisteredTask,
  parameterTemplate,
  parseIntegerList,
  REPERTOIRE,
  selectAlgorithm,
  TRACKS,
} from "./config";
import MarlDesign from "./tracks/marl";

export function ExperimentSection({ config, track, onTrack, set }) {
  return (
    <>
      <section className="track-picker" aria-labelledby="track-heading">
        <div className="section-card__heading"><h2 id="track-heading">Choose a track</h2></div>
        <div className="track-grid">
          {TRACKS.map((item) => (
            <button
              type="button"
              className={`track-card track-card--${item.id} ${track === item.id ? "track-card--active" : ""}`}
              key={item.id}
              aria-pressed={track === item.id}
              onClick={() => onTrack(item.id)}
            >
              <span className="track-card__mark">{item.short}</span>
              <span className="track-card__copy"><strong>{item.title}</strong></span>
            </button>
          ))}
        </div>
      </section>
      <section className="section-card section-card--minimal">
        <div className="field-grid">
          <label className="field field--wide field--prominent">
            <span className="field__label">Experiment name</span>
            <input className="control" value={config.experiment.name} onChange={(event) => set("experiment.name", event.target.value)} />
          </label>
        </div>
      </section>
    </>
  );
}
function HandOverview({ hands }) {
  return (
    <div className="hand-overview field--wide">
      <div className="hand-overview__keyboard">
        {hands.map((hand, index) => {
          const [start, end] = hand.initial_key_range;
          return (
            <span className={`hand-overview__range hand-overview__range--${index + 1}`} style={{ width: `${((end - start + 1) / 88) * 100}%` }} key={hand.id}>
              {start}–{end}
            </span>
          );
        })}
      </div>
      <div className="hand-overview__legend">
        {hands.map((hand, index) => <span key={hand.id}><i className={`hand-dot hand-dot--${index + 1}`} />{hand.name}</span>)}
      </div>
    </div>
  );
}
export function TaskSection({ config, environments, set, update }) {
  const isMarl = config.design.type === "marl";
  const isCustom = config.task.source === "custom";
  const environmentOptions = [...new Set([
    config.task.environment_id,
    ...environments.map((item) => item.id),
  ].filter(Boolean))];
  const chooseEnvironment = (id) => {
    const metadata = environments.find((item) => item.id === id) ?? { id };
    update((current) => makeRegisteredTask(current, metadata));
  };
  const changeSource = (source) => {
    if (source === "custom") update(makeCustomMarl);
    else if (environments.length) chooseEnvironment(environments[0].id);
  };

  if (!isMarl) {
    return (
      <section className="section-card section-card--minimal">
        <div className="field-grid">
          <select
            className="control field--wide environment-select"
            aria-label="Environment ID"
            value={config.task.environment_id}
            onChange={(event) => chooseEnvironment(event.target.value)}
          >
            {environmentOptions.map((id) => <option value={id} key={id}>{id}</option>)}
          </select>
        </div>
      </section>
    );
  }

  return (
    <Section title="Task">
      {isMarl && (
        <ChoiceRow label="Task source" value={config.task.source} options={[
          { value: "registered", label: "Registered environment" },
          { value: "custom", label: "Custom MARL task" },
        ]} onChange={changeSource} />
      )}
      {!isCustom && (
        <SelectField label="Environment" value={config.task.environment_id} options={environmentOptions} onChange={chooseEnvironment} />
      )}
      {isCustom && (
        <>
          <SelectField label="Piece" value={config.task.music.piece} options={REPERTOIRE} onChange={(value) => set("task.music.piece", value)} />
          <ChoiceRow label="Number of hands" value={config.task.embodiment.num_hands} options={[3, 4, 5].map((value) => ({ value, label: String(value) }))} onChange={(value) => update((current) => applyHandLayout(current, value))} />
          <HandOverview hands={config.task.embodiment.hands} />
        </>
      )}
    </Section>
  );
}

export function DesignSection({ config, environments, set, update, panel }) {
  if (config.design.type === "marl" && config.task.source === "custom") {
    return <MarlDesign fullConfig={config} config={config.design.config} set={set} update={update} panel={panel} />;
  }
  if (config.design.type === "robust") {
    return (
      <Section title="Robustness protocol">
        <SelectField label="Evaluation protocol" value={config.design.config.evaluation_protocol ?? "matched"} options={[{ value: "matched", label: "Matched perturbation" }]} onChange={(value) => set("design.config.evaluation_protocol", value)} />
      </Section>
    );
  }
  if (config.design.type === "safety") {
    const selected = environments.find((item) => item.id === config.task.environment_id);
    return (
      <Section title="Safety task">
        <div className="recipe-summary field--wide">
          <span>Group <strong>{config.design.config.group}</strong></span>
          <span>Task <strong>{config.design.config.task_index}</strong></span>
          {selected && <span>Cost <strong>{selected.cost} · {selected.setting}</strong></span>}
        </div>
      </Section>
    );
  }
  if (config.design.type === "marl") {
    return <Section title="Agent assignment"><div className="inline-callout field--wide">This registered task uses its territorial agent layout.</div></Section>;
  }
  return <Section title="Standard task"><div className="inline-callout field--wide">The registered environment fully defines the nominal control task.</div></Section>;
}

function ParameterValue({ name, value, onCommit, comma }) {
  const serialized = JSON.stringify(value);
  const [draft, setDraft] = useState(serialized);
  const [error, setError] = useState("");
  useEffect(() => {
    setDraft(serialized);
    setError("");
  }, [serialized]);
  const commit = () => {
    try {
      const value = JSON.parse(draft);
      onCommit(value);
      setError("");
    } catch (reason) {
      setError(reason instanceof SyntaxError ? "Invalid JSON value" : reason.message);
    }
  };
  return (
    <div className="parameter-json__editable-row">
      <span className="parameter-json__key">{JSON.stringify(name)}: </span>
      <input
        className="parameter-json__input"
        aria-label={name}
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        onBlur={commit}
        onKeyDown={(event) => { if (event.key === "Enter") event.currentTarget.blur(); }}
      />
      <span>{comma ? "," : ""}</span>
      {error && <span className="parameter-json__error">{error}</span>}
    </div>
  );
}

function readOnlyEntry(name, value, comma) {
  const serialized = JSON.stringify(value, null, 2).replaceAll("\n", "\n  ");
  return `${JSON.stringify(name)}: ${serialized}${comma ? "," : ""}`;
}

function ParameterEditor({ parameters, allowed, template, displayTemplate, onChange }) {
  const entries = Object.entries(displayTemplate);
  const updateParameter = (name, value) => {
    const next = { ...parameters };
    if (JSON.stringify(value) === JSON.stringify(template[name])) delete next[name];
    else next[name] = value;
    onChange(next);
  };
  return (
    <div className="parameter-editor field--wide">
      <span className="field__label">Parameters</span>
      <div className="parameter-json" aria-label="Algorithm parameters">
        <code className="parameter-json__brace">{"{"}</code>
        {entries.map(([name, reference], index) => {
          const comma = index < entries.length - 1;
          if (allowed.includes(name)) {
            const value = Object.hasOwn(parameters, name) ? parameters[name] : reference;
            return <ParameterValue key={name} name={name} value={value} comma={comma} onCommit={(next) => updateParameter(name, next)} />;
          }
          return <code className="parameter-json__readonly" aria-disabled="true" key={name}>{readOnlyEntry(name, reference, comma)}</code>;
        })}
        <code className="parameter-json__brace">{"}"}</code>
      </div>
    </div>
  );
}

export function AlgorithmSection({ config, algorithms, update, panel }) {
  const algorithmPanel = panel("algorithm");
  const selected = algorithms.find((item) => item.name === config.algorithm.name);
  const template = selected
    ? parameterTemplate(config.design.type, selected.name, selected.configurable_parameters)
    : {};
  const displayTemplate = selected
    ? parameterDisplayTemplate(config.design.type, selected.name, selected.configurable_parameters)
    : {};
  return (
    <Section title="Available algorithms" {...algorithmPanel}>
      <div className="algorithm-grid field--wide">
        {algorithms.map((item) => (
          <button type="button" key={item.name} className={`algorithm-option ${config.algorithm.name === item.name ? "active" : ""}`} aria-pressed={config.algorithm.name === item.name} onClick={() => update((current) => selectAlgorithm(current, item.name))}>
            <strong>{item.label}</strong>
          </button>
        ))}
      </div>
      {!algorithms.length && <div className="inline-callout field--wide">Start the local API to load the live algorithm catalog.</div>}
      {algorithmPanel.advanced && selected && (
        <ParameterEditor parameters={config.algorithm.parameters} allowed={selected.configurable_parameters} template={template} displayTemplate={displayTemplate} onChange={(parameters) => update((current) => ({ ...current, algorithm: { ...current.algorithm, parameters } }))} />
      )}
    </Section>
  );
}

export function RunSection({ config, track, trackCapability, set, update, panel }) {
  const { evaluation, video, compute, tracking, output } = config.run;
  const training = panel("run.training");
  const resources = panel("run.resources");
  const supportsPeriodicVideo = trackCapability?.features?.video === "periodic_and_final";
  const trackingOptions = [
    "none",
    ...(trackCapability?.features?.tracking ?? [])
      .filter((provider) => ["wandb", "tensorboard", "jsonl"].includes(provider)),
  ];
  const uniqueTrackingOptions = [...new Set(trackingOptions)];
  return (
    <>
      <Section title="Run settings" {...training}>
        <ListField label="Seeds" values={config.experiment.seeds} parse={parseIntegerList} onChange={(value) => set("experiment.seeds", value)} hint="Comma-separated integers." />
        <NumberField label="Total environment steps" value={config.run.total_steps} min={track === "safety" ? 20000 : 1} step={track === "safety" ? 20000 : 1} onChange={(value) => set("run.total_steps", value)} />
        {track === "safety" ? (
          <div className="recipe-summary"><span>Gamma <strong>0.8 (backend fixed)</strong></span></div>
        ) : (
          <NumberField label="Gamma" value={config.run.gamma} min={0.001} max={1} step={0.01} onChange={(value) => set("run.gamma", value)} />
        )}
        {track !== "safety" && <NumberField label="Evaluation frequency" value={evaluation.frequency} min={1} onChange={(value) => set("run.evaluation.frequency", value)} />}
        <NumberField label="Evaluation episodes" value={evaluation.episodes} min={1} onChange={(value) => set("run.evaluation.episodes", value)} />
        {supportsPeriodicVideo && <ToggleField label="Evaluation video" checked={video.enabled} onChange={(value) => set("run.video.enabled", value)} />}
        {training.advanced && (
          <>
            <NumberField label="Evaluation seed offset" value={evaluation.seed_offset} min={0} onChange={(value) => set("run.evaluation.seed_offset", value)} />
            {supportsPeriodicVideo && video.enabled && <NumberField label="Video frequency" value={video.frequency} min={1} onChange={(value) => set("run.video.frequency", value)} />}
            {supportsPeriodicVideo && video.enabled && <TextField label="Camera" value={video.camera} onChange={(value) => set("run.video.camera", value)} />}
          </>
        )}
      </Section>
      <Section title="Compute, tracking & output" {...resources}>
        {!resources.advanced && (
          <div className="recipe-summary field--wide">
            <span>Workers <strong>{compute.num_workers}</strong></span>
            <span>Device <strong>{compute.device}</strong></span>
            <span>Tracking <strong>{tracking.provider === "none" ? "off" : tracking.provider}</strong></span>
            <span>Output <strong>{output.root}</strong></span>
          </div>
        )}
        {resources.advanced && (
          <>
            <SelectField label="Device" value={compute.device} options={["auto", "cpu", "cuda"]} onChange={(value) => set("run.compute.device", value)} />
            {track === "marl" && <NumberField label="Workers" value={compute.num_workers} min={0} onChange={(value) => set("run.compute.num_workers", value)} />}
            {["standard", "robust"].includes(track) && <NumberField label="Vector environments" value={compute.num_envs} min={1} onChange={(value) => set("run.compute.num_envs", value)} />}
            <SelectField label="Tracking provider" value={tracking.provider} options={uniqueTrackingOptions} onChange={(value) => update((current) => applyTrackingProvider(current, value))} />
            {tracking.provider !== "none" && <SelectField label="Tracking mode" value={tracking.mode} options={["offline", "online"]} onChange={(value) => set("run.tracking.mode", value)} />}
            <TextField label="Project" value={tracking.project} onChange={(value) => set("run.tracking.project", value || null)} />
            <TextField label="Run group" value={tracking.group} onChange={(value) => set("run.tracking.group", value || null)} />
            <TextField label="Output root" value={output.root} onChange={(value) => set("run.output.root", value)} hint="The local API confines writes to its configured allowed root." />
          </>
        )}
      </Section>
    </>
  );
}
