import { useRef } from "react";

import { ChoiceRow, NumberField, Section, SelectField } from "../fields";
import { applyHandLayout, applyMarlAgentCount, assignMarlHandOwner, REPERTOIRE } from "../config";

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function RangeSlider({ range, agentIndex, kind, onChange }) {
  const drag = useRef(null);
  const left = ((range[0] - 1) / 88) * 100;
  const width = ((range[1] - range[0] + 1) / 88) * 100;
  const keyAtPointer = (event, track) => {
    const bounds = track.getBoundingClientRect();
    return clamp(Math.round(((event.clientX - bounds.left) / bounds.width) * 87) + 1, 1, 88);
  };
  const beginDrag = (mode, event) => {
    event.preventDefault();
    event.stopPropagation();
    const track = event.currentTarget.closest(".agent-range__track");
    track.setPointerCapture(event.pointerId);
    drag.current = {
      mode,
      pointerId: event.pointerId,
      origin: keyAtPointer(event, track),
      range: [...range],
    };
  };
  const moveDrag = (event) => {
    const current = drag.current;
    if (!current || current.pointerId !== event.pointerId) return;
    const key = keyAtPointer(event, event.currentTarget);
    if (current.mode === "start") onChange([Math.min(key, current.range[1]), current.range[1]]);
    if (current.mode === "end") onChange([current.range[0], Math.max(key, current.range[0])]);
    if (current.mode === "range") {
      const span = current.range[1] - current.range[0];
      const start = clamp(current.range[0] + key - current.origin, 1, 88 - span);
      onChange([start, start + span]);
    }
  };
  const endDrag = (event) => {
    if (!drag.current || drag.current.pointerId !== event.pointerId) return;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    drag.current = null;
  };
  const adjustEdge = (edge, delta) => {
    if (edge === "start") onChange([clamp(range[0] + delta, 1, range[1]), range[1]]);
    else onChange([range[0], clamp(range[1] + delta, range[0], 88)]);
  };
  const handleKey = (edge, event) => {
    if (!["ArrowLeft", "ArrowDown", "ArrowRight", "ArrowUp"].includes(event.key)) return;
    event.preventDefault();
    adjustEdge(edge, ["ArrowLeft", "ArrowDown"].includes(event.key) ? -1 : 1);
  };
  return (
    <div
      className="agent-range__track agent-range__track--interactive"
      aria-label={`${kind} keys ${range[0]} to ${range[1]}`}
      onPointerMove={moveDrag}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      <span
        className={`agent-range__bar agent-range__bar--${agentIndex + 1} agent-range__bar--${kind}`}
        style={{ left: `${left}%`, width: `${width}%` }}
        onPointerDown={(event) => beginDrag("range", event)}
      >
        <span
          className="agent-range__handle agent-range__handle--start"
          role="slider"
          tabIndex="0"
          aria-label={`${kind} start`}
          aria-valuemin="1"
          aria-valuemax={range[1]}
          aria-valuenow={range[0]}
          onPointerDown={(event) => beginDrag("start", event)}
          onKeyDown={(event) => handleKey("start", event)}
        >
          {range[0]}
        </span>
        <span
          className="agent-range__handle agent-range__handle--end"
          role="slider"
          tabIndex="0"
          aria-label={`${kind} end`}
          aria-valuemin={range[0]}
          aria-valuemax="88"
          aria-valuenow={range[1]}
          onPointerDown={(event) => beginDrag("end", event)}
          onKeyDown={(event) => handleKey("end", event)}
        >
          {range[1]}
        </span>
      </span>
    </div>
  );
}

const BLACK_PITCH_CLASSES = new Set([1, 3, 6, 8, 10]);
const BLACK_KEYS = Array.from({ length: 88 }, (_, index) => index + 1)
  .filter((key) => BLACK_PITCH_CLASSES.has((20 + key) % 12));

function HandGlyph({ mirrored }) {
  return (
    <svg viewBox="0 0 64 92" aria-hidden="true">
      <g transform={mirrored ? "translate(64 0) scale(-1 1)" : undefined}>
        <rect x="21" y="2" width="23" height="28" rx="8" />
        <rect x="15" y="25" width="36" height="37" rx="13" />
        <rect x="8" y="46" width="9" height="34" rx="4.5" transform="rotate(13 12.5 63)" />
        <rect x="18" y="53" width="8" height="35" rx="4" transform="rotate(5 22 70.5)" />
        <rect x="28" y="54" width="8" height="36" rx="4" />
        <rect x="38" y="52" width="8" height="35" rx="4" transform="rotate(-5 42 69.5)" />
        <rect x="48" y="48" width="8" height="31" rx="4" transform="rotate(-12 52 63.5)" />
        <rect x="49" y="32" width="9" height="27" rx="4.5" transform="rotate(-42 53.5 45.5)" />
      </g>
    </svg>
  );
}

function PianoKeyboard({ hands, agents, update }) {
  const assignNextAgent = (handId, ownerIndex) => {
    if (agents.length < 2) return;
    update((current) => assignMarlHandOwner(current, handId, (ownerIndex + 1) % agents.length));
  };
  return (
    <div className="piano-assignment-stage">
      <div className="piano-keyboard" aria-hidden="true">
        {BLACK_KEYS.map((key) => (
          <i
            className="piano-keyboard__black-key"
            style={{ left: `${((key - 0.5) / 88) * 100}%` }}
            key={key}
          />
        ))}
      </div>
      <div className="piano-hands">
        {hands.map((hand) => {
          const ownerIndex = agents.findIndex((agent) => agent.hand_ids.includes(hand.id));
          const [start, end] = hand.initial_key_range;
          const center = (((start + end) / 2 - 1) / 87) * 100;
          return (
            <button
              type="button"
              className={`piano-hand piano-hand--${ownerIndex + 1}`}
              style={{ left: `${center}%` }}
              aria-label={`H${hand.id + 1} ${hand.name}, assigned to ${agents[ownerIndex]?.name}; assign next agent`}
              onClick={() => assignNextAgent(hand.id, ownerIndex)}
              key={hand.id}
            >
              <HandGlyph mirrored={hand.side === "left"} />
              <span className="piano-hand__identity">H{hand.id + 1}</span>
              <strong>Agent {ownerIndex + 1}</strong>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function PianoWorkbench({ fullConfig, config, set, update }) {
  const agents = config.agents;
  return (
    <div className="piano-workbench field--wide">
      <div className="piano-range-lanes">
      {agents.map((agent, index) => (
        <article className="piano-range-agent" key={agent.name}>
          <header><i className={`agent-swatch agent-swatch--${index + 1}`} /><strong>Agent {index + 1}</strong></header>
          <div className="piano-range-row">
            <span>Action</span>
            <RangeSlider
              range={agent.action_key_range}
              agentIndex={index}
              kind="action"
              onChange={(value) => set(["design", "config", "agents", index, "action_key_range"], value)}
            />
          </div>
          <div className="piano-range-row">
            <span>Observation</span>
            <RangeSlider
              range={agent.observation_key_range}
              agentIndex={index}
              kind="observation"
              onChange={(value) => set(["design", "config", "agents", index, "observation_key_range"], value)}
            />
          </div>
        </article>
      ))}
      </div>
      <PianoKeyboard hands={fullConfig.task.embodiment.hands} agents={agents} update={update} />
    </div>
  );
}

export default function MarlDesign({ fullConfig, config, set, update }) {
  const hands = fullConfig.task.embodiment.hands;

  return (
    <>
      <Section title="General configuration">
        <SelectField
          label="Piece"
          value={fullConfig.task.music.piece}
          options={REPERTOIRE}
          onChange={(value) => set("task.music.piece", value)}
        />
        <ChoiceRow
          label="Number of hands"
          value={fullConfig.task.embodiment.num_hands}
          options={[3, 4, 5].map((value) => ({ value, label: String(value) }))}
          onChange={(value) => update((current) => applyHandLayout(current, value))}
          wide={false}
        />
        <NumberField
          label="Number of agents"
          value={config.num_agents}
          min={1}
          max={hands.length}
          onChange={(value) => update((current) => applyMarlAgentCount(current, value))}
        />
        <SelectField
          label="Sustain owner"
          value={config.sustain_owner}
          onChange={(value) => set("design.config.sustain_owner", value)}
          options={config.agents.map((agent, index) => ({ value: agent.name, label: `Agent ${index + 1}` }))}
        />
      </Section>

      <Section title="Hand assignment & ranges">
        <PianoWorkbench fullConfig={fullConfig} config={config} set={set} update={update} />
      </Section>
    </>
  );
}
