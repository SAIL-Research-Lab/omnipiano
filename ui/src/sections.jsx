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
  makeRegisteredTask,
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
              <span className="track-card__copy"><strong>{item.title}</strong><small>{item.description}</small></span>
            </button>
          ))}
        </div>
      </section>
      <section className="section-card section-card--minimal">
        <div className="field-grid">
          <TextField label="Experiment name" value={config.experiment.name} onChange={(value) => set("experiment.name", value)} wide />
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

function ParameterEditor({ parameters, allowed, onChange }) {
  const [draft, setDraft] = useState(() => JSON.stringify(parameters, null, 2));
  const [error, setError] = useState("");
  useEffect(() => setDraft(JSON.stringify(parameters, null, 2)), [parameters]);
  const commit = () => {
    try {
      const value = JSON.parse(draft);
      if (!value || Array.isArray(value) || typeof value !== "object") throw new Error("Use a JSON object");
      const unknown = Object.keys(value).filter((name) => !allowed.includes(name));
      if (unknown.length) throw new Error(`Unsupported: ${unknown.join(", ")}`);
      onChange(value);
      setError("");
    } catch (reason) {
      setError(reason.message);
    }
  };
  return (
    <label className="field field--wide">
      <span className="field__label">Parameter overrides</span>
      <textarea className="control" rows="7" value={draft} onChange={(event) => setDraft(event.target.value)} onBlur={commit} />
      <span className="field__hint">Allowed keys: {allowed.length ? allowed.join(", ") : "none"}</span>
      {error && <span className="field__hint">{error}</span>}
    </label>
  );
}

export function AlgorithmSection({ config, algorithms, update, panel }) {
  const algorithmPanel = panel("algorithm");
  const selected = algorithms.find((item) => item.name === config.algorithm.name);
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
        <ParameterEditor parameters={config.algorithm.parameters} allowed={selected.configurable_parameters} onChange={(parameters) => update((current) => ({ ...current, algorithm: { ...current.algorithm, parameters } }))} />
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

export function ReviewSection({ config, errors, backend, onCopy, onDownload, onPreview, onPrepare, onLaunch, onStop, copyLabel }) {
  const track = TRACKS.find((item) => item.id === config.design.type);
  const taskLabel = config.task.source === "registered" ? config.task.environment_id : `${config.task.music.piece} · ${config.task.embodiment.num_hands} hands`;
  const allErrors = [...errors, ...(backend.type === "error" ? [backend.message] : [])];
  return (
    <>
      <section className="review-hero">
        <span className={`track-card__mark track-card__mark--${track.id}`}>{track.short}</span>
        <div><h2>{config.experiment.name}</h2><p>{taskLabel}</p></div>
      </section>
      <div className="summary-grid">
        <div><span>Track</span><strong>{track.title}</strong></div>
        <div><span>Algorithm</span><strong>{config.algorithm.name}</strong></div>
        <div><span>Steps</span><strong>{config.run.total_steps.toLocaleString()}</strong></div>
        <div><span>Seeds</span><strong>{config.experiment.seeds.join(", ")}</strong></div>
      </div>
      <section className={`validation-panel ${allErrors.length ? "validation-panel--error" : ""}`}>
        <div><span className="validation-panel__icon">{allErrors.length ? "!" : "✓"}</span><div><strong>{allErrors.length ? "Configuration needs attention" : backend.message || "Ready for backend validation"}</strong></div></div>
        {allErrors.length > 0 && <ul>{allErrors.slice(0, 8).map((error) => <li key={error}>{error}</li>)}</ul>}
      </section>
      {backend.runs?.length > 0 && (
        <section className="section-card"><div className="section-card__heading"><h2>Runs</h2></div><div className="field-grid">
          {backend.runs.map((run) => <div className="recipe-summary field--wide" key={run.run_id}><span><strong>{run.run_id}</strong></span><span>{run.process?.state ?? "prepared"}</span>{run.process?.state === "running" && <button type="button" onClick={() => onStop(run.run_id)}>Stop</button>}</div>)}
        </div></section>
      )}
      <div className="review-actions">
        <button type="button" className="button button--secondary" onClick={onCopy}>{copyLabel}</button>
        <button type="button" className="button button--secondary" onClick={onDownload} disabled={errors.length > 0}>Download JSON</button>
        <button type="button" className="button button--secondary" onClick={onPreview} disabled={errors.length > 0 || backend.busy}>Validate & preview</button>
        <button type="button" className="button button--secondary" onClick={onPrepare} disabled={errors.length > 0 || backend.busy}>Prepare files</button>
        <button type="button" className="button button--primary" onClick={onLaunch} disabled={errors.length > 0 || backend.busy}>Launch runs</button>
      </div>
    </>
  );
}
