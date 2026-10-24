import { useEffect, useState } from "react";

import {
  ChoiceRow,
  ListField,
  NumberField,
  Section,
  SelectField,
} from "./fields";
import {
  applyHandLayout,
  applyTrackingProvider,
  deleteAtPath,
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

export function DesignSection({ config, set, update, panel }) {
  if (config.design.type === "marl" && config.task.source === "custom") {
    return <MarlDesign fullConfig={config} config={config.design.config} set={set} update={update} panel={panel} />;
  }
  if (config.design.type !== "marl") {
    return (
      <section className="section-card section-card--fixed" aria-disabled="true">
        <div className="fixed-design-placeholder">Fixed by registered envs, not applicable right now</div>
      </section>
    );
  }
  return <Section title="Agent assignment"><div className="inline-callout field--wide">This registered task uses its territorial agent layout.</div></Section>;
}

function ParameterValue({ name, value, onCommit, comma, ariaLabel = name, className = "" }) {
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
    <div className={`parameter-json__editable-row ${className}`.trim()}>
      <span className="parameter-json__key">{JSON.stringify(name)}: </span>
      <input
        className="parameter-json__input"
        aria-label={ariaLabel}
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

function valueAtPath(source, path, fallback = null) {
  const value = path.split(".").reduce((cursor, part) => cursor?.[part], source);
  return value === undefined ? fallback : value;
}

function runtimeParameterGroups(config, track) {
  const editable = {
    standard: new Set(["compute.num_envs", "tracking.provider", "output.root"]),
    robust: new Set(["compute.num_envs", "output.root"]),
    safety: new Set(["tracking.provider", "output.root"]),
    marl: new Set([
      "evaluation.seed_offset",
      "video.enabled", "video.frequency", "video.camera", "video.width", "video.height",
      "compute.num_workers", "compute.cpus", "compute.gpus",
      "tracking.provider", "tracking.mode", "tracking.project", "tracking.group", "tracking.name",
      "output.root",
    ]),
  }[track] ?? new Set();
  const entry = (name, path, options = {}) => ({
    name,
    path,
    value: valueAtPath(config.run, path, options.fallback),
    editable: editable.has(path),
    ...options,
  });
  const trackingProviders = {
    standard: ["none", "tensorboard"],
    robust: ["none"],
    safety: ["none", "tensorboard"],
    marl: ["none", "wandb", "jsonl"],
  }[track] ?? ["none"];
  return [
    {
      name: "evaluation",
      entries: [
        entry("enabled", "evaluation.enabled"),
        entry("seed_offset", "evaluation.seed_offset"),
      ],
    },
    {
      name: "video",
      entries: [
        entry("enabled", "video.enabled"),
        entry("frequency", "video.frequency"),
        entry("camera", "video.camera"),
        entry("width", "video.width"),
        entry("height", "video.height"),
      ],
    },
    {
      name: "compute",
      entries: [
        entry("num_workers", "compute.num_workers"),
        entry("num_envs", "compute.num_envs"),
        entry("cpus", "compute.cpus", { fallback: null, optional: true }),
        entry("gpus", "compute.gpus", { fallback: null, optional: true }),
      ],
    },
    {
      name: "tracking",
      entries: [
        entry("provider", "tracking.provider", { choices: trackingProviders }),
        entry("mode", "tracking.mode", { choices: ["disabled", "offline", "online"] }),
        entry("project", "tracking.project"),
        entry("group", "tracking.group"),
        entry("name", "tracking.name"),
      ],
    },
    {
      name: "output",
      entries: [entry("root", "output.root")],
    },
  ];
}

function RuntimeParameterEditor({ config, track, update }) {
  const groups = runtimeParameterGroups(config, track);
  const commit = (entry, value) => {
    if (entry.choices && !entry.choices.includes(value)) {
      throw new Error(`Use one of: ${entry.choices.join(", ")}`);
    }
    update((current) => {
      if (entry.path === "tracking.provider") return applyTrackingProvider(current, value);
      if (entry.optional && value === null) return deleteAtPath(current, `run.${entry.path}`);
      const next = structuredClone(current);
      const parts = entry.path.split(".");
      let cursor = next.run;
      for (let index = 0; index < parts.length - 1; index += 1) cursor = cursor[parts[index]];
      cursor[parts.at(-1)] = value;
      return next;
    });
  };
  return (
    <div className="parameter-editor field--wide">
      <span className="field__label">Advanced configuration</span>
      <div className="parameter-json runtime-parameter-json" aria-label="Advanced training configuration">
        <code className="parameter-json__brace">{"{"}</code>
        {groups.map((group, groupIndex) => (
          <div className="runtime-parameter-json__group" key={group.name}>
            <code className="runtime-parameter-json__group-brace">{`${JSON.stringify(group.name)}: {`}</code>
            {group.entries.map((entry, entryIndex) => {
              const comma = entryIndex < group.entries.length - 1;
              if (entry.editable) {
                return (
                  <ParameterValue
                    key={entry.path}
                    name={entry.name}
                    ariaLabel={entry.path}
                    value={entry.value}
                    comma={comma}
                    className="runtime-parameter-json__entry"
                    onCommit={(value) => commit(entry, value)}
                  />
                );
              }
              return (
                <code className="parameter-json__readonly runtime-parameter-json__readonly" aria-disabled="true" key={entry.path}>
                  {readOnlyEntry(entry.name, entry.value, comma)}
                </code>
              );
            })}
            <code className="runtime-parameter-json__group-brace">{`}${groupIndex < groups.length - 1 ? "," : ""}`}</code>
          </div>
        ))}
        <code className="parameter-json__brace">{"}"}</code>
      </div>
    </div>
  );
}

export function RunSection({ config, track = config.design.type, set, update, panel }) {
  const { evaluation, compute } = config.run;
  const training = panel("run");
  return (
    <Section title="Training configuration" {...training}>
      <ListField label="Seeds" values={config.experiment.seeds} parse={parseIntegerList} onChange={(value) => set("experiment.seeds", value)} />
      <NumberField label="Total environment steps" value={config.run.total_steps} min={track === "safety" ? 20000 : 1} step={track === "safety" ? 20000 : 1} onChange={(value) => set("run.total_steps", value)} />
      {track === "safety" ? (
        <div className="recipe-summary recipe-summary--fixed"><span>Gamma <strong>0.8</strong></span></div>
      ) : (
        <NumberField label="Gamma" value={config.run.gamma} min={0.001} max={1} step={0.01} onChange={(value) => set("run.gamma", value)} />
      )}
      {track !== "safety" && <NumberField label="Evaluation frequency" value={evaluation.frequency} min={1} onChange={(value) => set("run.evaluation.frequency", value)} />}
      <NumberField label="Evaluation episodes" value={evaluation.episodes} min={1} onChange={(value) => set("run.evaluation.episodes", value)} />
      <SelectField label="Device" value={compute.device} options={["auto", "cpu", "cuda"]} onChange={(value) => set("run.compute.device", value)} />
      {training.advanced && <RuntimeParameterEditor config={config} track={track} update={update} />}
    </Section>
  );
}
