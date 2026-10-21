import {
  ChoiceRow,
  ListField,
  NumberField,
  NumberInput,
  Section,
  SelectField,
  TextField,
  ToggleField,
} from "./fields";
import {
  applyHandLayout,
  applyMusicSelection,
  applyTrackingProvider,
  parseIntegerList,
  parseStringList,
  REPERTOIRE,
  TRACKS,
} from "./config";
import { algorithmsForTrack, findAlgorithm, selectAlgorithm } from "./algorithmCatalog";
import MarlDesign from "./tracks/marl";
import RobustDesign, { RobustEvaluationViews } from "./tracks/robust";
import SafetyDesign from "./tracks/safety";
import StandardDesign from "./tracks/standard";

export function ExperimentSection({ config, track, onTrack, set, panel }) {
  const identityPanel = panel("identity");
  return (
    <>
      <section className="track-picker" aria-labelledby="track-heading">
        <div className="section-card__heading">
          <h2 id="track-heading">Choose a track</h2>
        </div>
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
              <span className="track-card__copy">
                <strong>{item.title}</strong>
                <small>{item.description}</small>
              </span>
            </button>
          ))}
        </div>
      </section>

      <Section
        title="Experiment details"
        {...identityPanel}
      >
        <TextField
          label="Experiment name"
          value={config.experiment.name}
          onChange={(value) => set("experiment.name", value)}
          hint="Used as the JSON filename."
        />
        <TextField
          label="Description"
          value={config.experiment.description}
          onChange={(value) => set("experiment.description", value)}
          wide
          multiline
        />
        {identityPanel.advanced && (
          <ListField
            label="Tags"
            values={config.experiment.tags}
            parse={parseStringList}
            onChange={(value) => set("experiment.tags", value)}
            hint="Comma-separated labels."
            wide
          />
        )}
      </Section>
    </>
  );
}

function HandOverview({ hands }) {
  return (
    <div className="hand-overview field--wide">
      <div className="hand-overview__keyboard" aria-label="Initial hand keyboard zones">
        {hands.map((hand, index) => {
          const [start, end] = hand.initial_key_range;
          return (
            <span
              className={`hand-overview__range hand-overview__range--${index + 1}`}
              style={{ width: `${((end - start + 1) / 88) * 100}%` }}
              key={hand.id}
              title={`${hand.name}: keys ${start}–${end}`}
            >
              {start}–{end}
            </span>
          );
        })}
      </div>
      <div className="hand-overview__legend">
        {hands.map((hand, index) => (
          <span key={hand.id}>
            <i className={`hand-dot hand-dot--${index + 1}`} />
            {hand.name}
          </span>
        ))}
      </div>
    </div>
  );
}

function HandEditor({ hands, set }) {
  return (
    <div className="hand-editor field--wide">
      <div className="compact-table__header">
        <span>Hand</span><span>Side</span><span>Initial keys</span>
      </div>
      {hands.map((hand, index) => (
        <div className="compact-table__row" key={hand.id}>
          <input
            className="control"
            value={hand.name}
            aria-label={`Hand ${index + 1} name`}
            onChange={(event) => set(["task", "embodiment", "hands", index, "name"], event.target.value)}
          />
          <select
            className="control"
            value={hand.side}
            aria-label={`${hand.name} side`}
            onChange={(event) => set(["task", "embodiment", "hands", index, "side"], event.target.value)}
          >
            <option value="left">Left</option>
            <option value="right">Right</option>
          </select>
          <div className="mini-range">
            <NumberInput
              min="1"
              max="88"
              value={hand.initial_key_range[0]}
              aria-label={`${hand.name} range start`}
              onChange={(value) => set(
                ["task", "embodiment", "hands", index, "initial_key_range"],
                [value, hand.initial_key_range[1]],
              )}
            />
            <span>–</span>
            <NumberInput
              min="1"
              max="88"
              value={hand.initial_key_range[1]}
              aria-label={`${hand.name} range end`}
              onChange={(value) => set(
                ["task", "embodiment", "hands", index, "initial_key_range"],
                [hand.initial_key_range[0], value],
              )}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function RewardEditor({ reward, set }) {
  return (
    <div className="reward-editor field--wide">
      <div className="reward-editor__header">
        <span>Reward term</span><span>Use</span><span>Coefficient</span>
      </div>
      {Object.entries(reward).map(([name, term]) => (
        <div className="reward-editor__row" key={name}>
          <strong>{name.replaceAll("_", " ")}</strong>
          <button
            type="button"
            className={`compact-toggle ${term.enabled ? "active" : ""}`}
            aria-pressed={term.enabled}
            onClick={() => set(["task", "reward", name, "enabled"], !term.enabled)}
          >
            {term.enabled ? "On" : "Off"}
          </button>
          <NumberInput
            min="0"
            step="0.001"
            value={term.coefficient}
            disabled={!term.enabled}
            aria-label={`${name} coefficient`}
            onChange={(value) => set(["task", "reward", name, "coefficient"], value)}
          />
        </div>
      ))}
    </div>
  );
}

export function TaskSection({ config, set, update, panel }) {
  const { music, embodiment, environment, reward } = config.task;
  const musicPanel = panel("task.music");
  const handsPanel = panel("task.hands");
  const behaviorPanel = panel("task.behavior");
  const chooseLayout = (numHands) => {
    update((current) => applyHandLayout(current, numHands, embodiment.mobility));
  };

  return (
    <>
      <Section
        title="Music"
        {...musicPanel}
      >
        <ChoiceRow
          label="Source"
          value={music.source}
          options={[
            { value: "repertoire", label: "Built-in repertoire" },
            { value: "custom_midi", label: "Custom MIDI" },
          ]}
          onChange={(value) => update((current) => applyMusicSelection(current, {
            source: value,
            piece: value === "repertoire" ? "ForElise" : null,
            midi_path: value === "custom_midi" ? "music/example.mid" : null,
          }))}
        />
        {music.source === "repertoire" ? (
          <SelectField
            label="Piece"
            value={music.piece}
            onChange={(value) => update((current) => applyMusicSelection(current, { piece: value }))}
            options={REPERTOIRE}
          />
        ) : (
          <TextField
            label="MIDI path"
            value={music.midi_path}
            onChange={(value) => set("task.music.midi_path", value)}
          />
        )}
        <SelectField
          label="Fingering assignment"
          value={music.fingering_mode}
          onChange={(value) => set("task.music.fingering_mode", value)}
          options={[
            ...(embodiment.num_hands === 2 ? [{ value: "annotated", label: "MIDI annotation · two hands" }] : []),
            { value: "optimal_transport", label: "Optimal transport · any layout" },
          ]}
        />
        {musicPanel.advanced && (
          <>
            <NumberField
              label="Tempo stretch"
              value={music.stretch_factor}
              min={0.1}
              step={0.05}
              onChange={(value) => set("task.music.stretch_factor", value)}
            />
            <NumberField
              label="Pitch shift"
              value={music.shift_factor}
              step={1}
              onChange={(value) => set("task.music.shift_factor", value)}
            />
            <ToggleField
              label="Trim leading silence"
              checked={music.trim_silence}
              onChange={(value) => set("task.music.trim_silence", value)}
            />
          </>
        )}
      </Section>

      <Section
        title="Hands"
        description={config.design.type === "marl" ? "These are initial positions; agent action ranges are set under Track design." : undefined}
        {...handsPanel}
      >
        <ChoiceRow
          label="Number of hands"
          value={embodiment.num_hands}
          options={[1, 2, 3, 4, 5].map((value) => ({ value, label: String(value) }))}
          onChange={(value) => chooseLayout(value)}
        />
        <HandOverview hands={embodiment.hands} />
        {handsPanel.advanced && <HandEditor hands={embodiment.hands} set={set} />}
      </Section>

      <Section title="Environment & reward" {...behaviorPanel}>
        {!behaviorPanel.advanced && (
          <div className="recipe-summary field--wide">
            <span>Lookahead <strong>{environment.n_steps_lookahead}</strong></span>
            <span>Frame stack <strong>{environment.frame_stack}</strong></span>
            <span>Reward terms <strong>{Object.values(reward).filter((term) => term.enabled).length} active</strong></span>
          </div>
        )}
        {behaviorPanel.advanced && (
          <>
            <NumberField
              label="Lookahead steps"
              value={environment.n_steps_lookahead}
              min={1}
              onChange={(value) => set("task.environment.n_steps_lookahead", value)}
            />
            <NumberField
              label="Control timestep"
              value={environment.control_timestep}
              min={0.001}
              step={0.01}
              onChange={(value) => set("task.environment.control_timestep", value)}
            />
            <NumberField
              label="Frame stack"
              value={environment.frame_stack}
              min={1}
              onChange={(value) => set("task.environment.frame_stack", value)}
            />
            <ToggleField
              label="Gravity compensation"
              checked={environment.gravity_compensation}
              onChange={(value) => set("task.environment.gravity_compensation", value)}
            />
            <ToggleField
              label="Clip actions"
              checked={environment.clip_actions}
              onChange={(value) => set("task.environment.clip_actions", value)}
            />
            <ToggleField
              label="Hand collisions"
              checked={environment.hand_collisions}
              onChange={(value) => set("task.environment.hand_collisions", value)}
            />
            <ToggleField
              label="Wrong-press termination"
              checked={environment.wrong_press_termination}
              onChange={(value) => set("task.environment.wrong_press_termination", value)}
            />
            <ToggleField
              label="Reduced action space"
              checked={environment.reduced_action_space}
              onChange={(value) => set("task.environment.reduced_action_space", value)}
            />
            <RewardEditor reward={reward} set={set} />
          </>
        )}
      </Section>
    </>
  );
}

export function DesignSection({ config, set, update, panel }) {
  const design = config.design;
  const props = { fullConfig: config, config: design.config, set, update, panel };
  if (design.type === "robust") return <RobustDesign {...props} />;
  if (design.type === "safety") return <SafetyDesign {...props} />;
  if (design.type === "marl") return <MarlDesign {...props} />;
  return <StandardDesign {...props} />;
}

function ParameterField({ name, value, onChange }) {
  const label = name.replaceAll("_", " ");
  if (typeof value === "boolean") {
    return <ToggleField label={label} checked={value} onChange={onChange} />;
  }
  if (typeof value === "number") {
    return <NumberField label={label} value={value} step="any" onChange={onChange} />;
  }
  if (Array.isArray(value)) {
    return (
      <ListField
        label={label}
        values={value}
        parse={(next) => {
          const values = parseStringList(next);
          const numeric = values.map(Number);
          return numeric.every(Number.isFinite) ? numeric : values;
        }}
        onChange={(next) => {
          onChange(next);
        }}
      />
    );
  }
  return <TextField label={label} value={value ?? ""} onChange={onChange} />;
}

export function AlgorithmSection({ config, update, panel }) {
  const algorithm = config.algorithm;
  const selected = findAlgorithm(config.design.type, algorithm.name);
  const choices = algorithmsForTrack(config.design.type);
  if (selected && !choices.includes(selected)) choices.push(selected);
  const algorithmPanel = panel("algorithm");
  return (
    <Section title="Available algorithms" {...algorithmPanel}>
      <div className="algorithm-grid field--wide">
        {choices.map((item) => (
          <button
            type="button"
            key={item.name}
            className={`algorithm-option ${algorithm.name === item.name ? "active" : ""}`}
            aria-pressed={algorithm.name === item.name}
            onClick={() => update((current) => selectAlgorithm(current, item.name))}
          >
            <strong>{item.label}</strong>
          </button>
        ))}
      </div>
      {!selected && <div className="inline-callout field--wide">This imported algorithm is not in the current track catalog. Choose an available option.</div>}
      {selected?.note && <div className="inline-callout field--wide">{selected.note}</div>}
      {algorithmPanel.advanced && (
        <>
          <h3 className="panel-subtitle field--wide">Parameters</h3>
          {Object.keys(algorithm.parameters).length === 0 && (
            <div className="inline-callout field--wide">No overrides are mapped for this algorithm.</div>
          )}
          {Object.entries(algorithm.parameters).map(([name, value]) => (
            <ParameterField
              key={name}
              name={name}
              value={value}
              onChange={(value) => update((current) => {
                const next = structuredClone(current);
                next.algorithm.parameters[name] = value;
                next.algorithm.preset = "custom";
                return next;
              })}
            />
          ))}
        </>
      )}
    </Section>
  );
}

export function RunSection({ config, set, update, panel }) {
  const { protocol, evaluation, compute, tracking } = config;
  const trainingPanel = panel("run.training");
  const resourcesPanel = panel("run.resources");
  return (
    <>
      <Section title="Run settings" {...trainingPanel}>
        <ListField
          label="Seeds"
          values={config.experiment.seeds}
          parse={parseIntegerList}
          onChange={(value) => set("experiment.seeds", value)}
          hint="Comma-separated integers."
        />
        <NumberField
          label="Total environment steps"
          value={protocol.total_env_steps}
          min={1}
          onChange={(value) => set("protocol.total_env_steps", value)}
        />
        <NumberField
          label="Gamma"
          value={protocol.gamma}
          min={0}
          max={1}
          step={0.01}
          onChange={(value) => set("protocol.gamma", value)}
        />
        <NumberField
          label="Evaluation frequency"
          value={evaluation.frequency}
          min={1}
          onChange={(value) => set("evaluation.frequency", value)}
        />
        <NumberField
          label="Evaluation episodes"
          value={evaluation.episodes}
          min={1}
          onChange={(value) => set("evaluation.episodes", value)}
        />
        <ToggleField
          label="Evaluation video"
          checked={evaluation.video.enabled}
          onChange={(value) => set("evaluation.video.enabled", value)}
        />
        {trainingPanel.advanced && (
          <>
            <NumberField
              label="Checkpoint frequency"
              value={protocol.checkpoint_frequency}
              min={0}
              onChange={(value) => set("protocol.checkpoint_frequency", value)}
            />
            <ToggleField
              label="Keep best checkpoint"
              checked={protocol.keep_best_checkpoint}
              onChange={(value) => set("protocol.keep_best_checkpoint", value)}
            />
            <NumberField
              label="Evaluation seed offset"
              value={evaluation.seed_offset}
              min={0}
              onChange={(value) => set("evaluation.seed_offset", value)}
            />
          </>
        )}
        {trainingPanel.advanced && evaluation.video.enabled && (
          <>
            <NumberField
              label="Video frequency"
              value={evaluation.video.frequency}
              min={1}
              onChange={(value) => set("evaluation.video.frequency", value)}
              hint="Environment steps; use a multiple of evaluation frequency."
            />
            <ToggleField
              label="Record final evaluation"
              checked={evaluation.video.record_final}
              onChange={(value) => set("evaluation.video.record_final", value)}
            />
          </>
        )}
      </Section>

      {config.design.type === "robust" && (
        <RobustEvaluationViews config={config.design.config} set={set} />
      )}

      <Section title="Compute & tracking" {...resourcesPanel}>
        {!resourcesPanel.advanced && (
          <div className="recipe-summary field--wide">
            <span>Runners <strong>{compute.env_runners}</strong></span>
            <span>GPUs / learner <strong>{compute.gpus_per_learner}</strong></span>
            <span>Tracking <strong>{tracking.provider === "none" ? "Off" : tracking.mode}</strong></span>
          </div>
        )}
        {resourcesPanel.advanced && (
          <>
            <NumberField
              label="Environment runners"
              value={compute.env_runners}
              min={0}
              onChange={(value) => set("compute.env_runners", value)}
            />
            <NumberField
              label="Learners"
              value={compute.learners}
              min={0}
              onChange={(value) => set("compute.learners", value)}
            />
            <NumberField
              label="CPUs per runner"
              value={compute.cpus_per_env_runner}
              min={0}
              onChange={(value) => set("compute.cpus_per_env_runner", value)}
            />
            <NumberField
              label="GPUs per learner"
              value={compute.gpus_per_learner}
              min={0}
              step={0.25}
              onChange={(value) => set("compute.gpus_per_learner", value)}
            />
            <SelectField
              label="Tracking provider"
              value={tracking.provider}
              onChange={(value) => update((current) => applyTrackingProvider(current, value))}
              options={["none", "wandb"]}
            />
            {tracking.provider === "wandb" && (
              <>
                <SelectField
                  label="Tracking mode"
                  value={tracking.mode}
                  onChange={(value) => set("tracking.mode", value)}
                  options={["offline", "online"]}
                />
                <TextField
                  label="Team / entity"
                  value={tracking.entity}
                  onChange={(value) => set("tracking.entity", value || null)}
                />
                <TextField
                  label="Project"
                  value={tracking.project}
                  onChange={(value) => set("tracking.project", value || null)}
                />
                <TextField
                  label="Run group"
                  value={tracking.group}
                  onChange={(value) => set("tracking.group", value || null)}
                />
                <ToggleField
                  label="Upload artifacts"
                  checked={tracking.upload_artifacts}
                  onChange={(value) => set("tracking.upload_artifacts", value)}
                />
              </>
            )}
          </>
        )}
      </Section>
    </>
  );
}

export function ReviewSection({ config, errors, warnings, onCopy, onDownload, copyLabel }) {
  const track = TRACKS.find((item) => item.id === config.design.type);
  return (
    <>
      <section className="review-hero">
        <span className={`track-card__mark track-card__mark--${track.id}`}>{track.short}</span>
        <div>
          <h2>{config.experiment.name}</h2>
          <p>{config.experiment.description}</p>
        </div>
      </section>
      <div className="summary-grid">
        <div><span>Track</span><strong>{track.title}</strong></div>
        <div><span>Piece</span><strong>{config.task.music.piece ?? "Custom MIDI"}</strong></div>
        <div><span>Hands</span><strong>{config.task.embodiment.num_hands}</strong></div>
        <div><span>Algorithm</span><strong>{config.algorithm.name}</strong></div>
        <div><span>Steps</span><strong>{config.protocol.total_env_steps.toLocaleString()}</strong></div>
        <div><span>Seeds</span><strong>{config.experiment.seeds.join(", ")}</strong></div>
      </div>
      <section className={`validation-panel ${errors.length ? "validation-panel--error" : ""}`}>
        <div>
          <span className="validation-panel__icon">{errors.length ? "!" : "✓"}</span>
          <div>
            <strong>
              {errors.length
                ? `${errors.length} configuration issue${errors.length === 1 ? "" : "s"}`
                : warnings.length
                  ? `Valid with ${warnings.length} warning${warnings.length === 1 ? "" : "s"}`
                  : "Configuration valid"}
            </strong>
            {errors.length > 0 && <p>Fix these fields before downloading.</p>}
          </div>
        </div>
        {errors.length > 0 && (
          <ul>{errors.slice(0, 8).map((error) => <li key={error}>{error}</li>)}</ul>
        )}
        {!errors.length && warnings.length > 0 && (
          <ul className="validation-panel__warnings">
            {warnings.slice(0, 8).map((warning) => <li key={warning}>{warning}</li>)}
          </ul>
        )}
      </section>
      <div className="review-actions">
        <button type="button" className="button button--secondary" onClick={onCopy}>{copyLabel}</button>
        <button type="button" className="button button--primary" onClick={onDownload} disabled={errors.length > 0}>Download JSON</button>
      </div>
    </>
  );
}
