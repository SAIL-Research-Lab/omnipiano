import { ChoiceRow, NumberField, Section, SelectField } from "../fields";
import {
  addRobustChannel,
  removeRobustChannel,
  ROBUST_CHANNELS,
  setRobustMagnitude,
  updateRobustChannel,
} from "../config";

const EVALUATION_MODES = [
  { value: "clean", label: "Clean baseline" },
  { value: "matched", label: "Matched noise" },
  { value: "cross_severity", label: "Cross severity" },
  { value: "stress", label: "Stress test" },
];

function MagnitudeEditor({ channel, index, update }) {
  const change = (key, value) => update((current) => setRobustMagnitude(current, index, key, value));
  if (channel.target === "initial_hand_pose") {
    return (
      <div className="magnitude-grid">
        <NumberField
          label="Lateral offset · mm"
          value={channel.magnitude.y_mm}
          min={0}
          step={1}
          onChange={(value) => change("y_mm", value)}
        />
        <NumberField
          label="Vertical offset · mm"
          value={channel.magnitude.z_mm}
          min={0}
          step={1}
          onChange={(value) => change("z_mm", value)}
        />
      </div>
    );
  }
  if (channel.distribution === "bounded_uniform") {
    return (
      <div className="magnitude-grid">
        <NumberField
          label="Lower bound"
          value={channel.magnitude.low}
          step={0.01}
          onChange={(value) => change("low", value)}
        />
        <NumberField
          label="Upper bound"
          value={channel.magnitude.high}
          step={0.01}
          onChange={(value) => change("high", value)}
        />
      </div>
    );
  }
  return (
    <NumberField
      label={channel.distribution === "gaussian" ? "Standard deviation" : "Shift"}
      value={channel.magnitude.value}
      step={0.01}
      onChange={(value) => change("value", value)}
    />
  );
}

function ChannelCard({ channel, index, set, update, channelPanel, canRemove }) {
  const metadata = ROBUST_CHANNELS[channel.target];
  return (
    <article className="channel-card">
      <header className="channel-card__header">
        <div>
          <span className={`channel-card__icon channel-card__icon--${metadata.group}`} aria-hidden="true" />
          <div><strong>{metadata.label}</strong>{channel.preset === "custom" && <small>Custom severity</small>}</div>
        </div>
        <div className="channel-card__actions">
          <button
            type="button"
            className={`panel-mode ${channelPanel.advanced ? "panel-mode--active" : ""}`}
            aria-pressed={channelPanel.advanced}
            aria-label={`${channelPanel.advanced ? "Hide" : "Show"} advanced settings for ${metadata.label}`}
            onClick={channelPanel.onToggleAdvanced}
          >
            {channelPanel.advanced ? "Basic" : "Advanced"}
          </button>
          {canRemove && (
            <button
              type="button"
              onClick={() => update((current) => removeRobustChannel(current, channel.target))}
            >
              Remove
            </button>
          )}
        </div>
      </header>
      <ChoiceRow
        label="Severity"
        value={channel.preset}
        options={[
          { value: "low", label: "L" },
          { value: "medium", label: "M" },
          { value: "high", label: "H" },
        ]}
        onChange={(preset) => update((current) => updateRobustChannel(current, index, { preset }))}
      />
      {channelPanel.advanced && (
        <div className="field-grid field-grid--nested">
          <SelectField
            label="Distribution"
            value={channel.distribution}
            onChange={(distribution) => update((current) => updateRobustChannel(current, index, { distribution }))}
            options={["gaussian", "bounded_uniform", "constant_shift"]}
          />
          <SelectField
            label="Resample"
            value={channel.frequency}
            onChange={(frequency) => set(["design", "config", "channels", index, "frequency"], frequency)}
            options={["step", "episode", "run"]}
          />
          <div className="field--wide"><MagnitudeEditor channel={channel} index={index} update={update} /></div>
        </div>
      )}
    </article>
  );
}

function ModeToggle({ mode, enabled, onChange }) {
  return (
    <button
      type="button"
      className={`mode-tile ${enabled ? "active" : ""}`}
      aria-pressed={enabled}
      onClick={onChange}
    >
      <span>{enabled ? "✓" : "+"}</span>
      <strong>{mode.label}</strong>
    </button>
  );
}

export default function RobustDesign({ config, set, update, panel }) {
  const activeTargets = new Set(config.channels.map((channel) => channel.target));
  const byGroup = (group) => config.channels
    .map((channel, index) => ({ channel, index }))
    .filter(({ channel }) => ROBUST_CHANNELS[channel.target].group === group);

  return (
    <Section title="Perturbations">
      {["signal", "physical"].map((group) => (
        <div className="channel-group field--wide" key={group}>
          <div className="channel-group__heading">
            <div><strong>{group === "signal" ? "Signal perturbations" : "Physical perturbations"}</strong><small>{group === "signal" ? "Inputs, outputs, and learning signal" : "Dynamics and initial conditions"}</small></div>
          </div>
          {byGroup(group).map(({ channel, index }) => (
            <ChannelCard
              key={channel.target}
              channel={channel}
              index={index}
              set={set}
              update={update}
              channelPanel={panel(`robust.channel.${channel.target}`)}
              canRemove={config.channels.length > 1}
            />
          ))}
          <div className="channel-add-row">
            {Object.entries(ROBUST_CHANNELS)
              .filter(([target, metadata]) => metadata.group === group && !activeTargets.has(target))
              .map(([target, metadata]) => (
                <button type="button" key={target} onClick={() => update((current) => addRobustChannel(current, target))}>
                  <span>+</span>{metadata.label}
                </button>
              ))}
            {!Object.entries(ROBUST_CHANNELS).some(([target, metadata]) => metadata.group === group && !activeTargets.has(target)) && (
              <small>All channels selected.</small>
            )}
          </div>
        </div>
      ))}
    </Section>
  );
}

export function RobustEvaluationViews({ config, set }) {
  const toggleMode = (mode) => {
    const modes = config.evaluation_modes;
    if (modes.includes(mode)) {
      if (modes.length > 1) set("design.config.evaluation_modes", modes.filter((item) => item !== mode));
    } else {
      set("design.config.evaluation_modes", [...modes, mode]);
    }
  };

  return (
    <Section title="Robustness evaluation">
      <div className="mode-grid field--wide">
        {EVALUATION_MODES.map((mode) => (
          <ModeToggle
            key={mode.value}
            mode={mode}
            enabled={config.evaluation_modes.includes(mode.value)}
            onChange={() => toggleMode(mode.value)}
          />
        ))}
      </div>
    </Section>
  );
}
