import { useRef } from "react";

import { ChoiceRow, NumberField, Section, SelectField, TextField } from "../fields";
import { applyMarlAgentCount, assignMarlHandOwner } from "../config";

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
        />
        {range[0]}–{range[1]}
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
        />
      </span>
    </div>
  );
}

function OwnershipEditor({ hands, agents, update }) {
  return (
    <div className="ownership-board field--wide">
      {hands.map((hand) => {
        const ownerIndex = agents.findIndex((agent) => agent.hand_ids.includes(hand.id));
        return (
          <div className={`ownership-hand ownership-hand--${ownerIndex + 1}`} key={hand.id}>
            <strong>H{hand.id + 1} · {hand.name}</strong>
            <select
              className="control"
              value={ownerIndex}
              aria-label={`${hand.name} owner`}
              onChange={(event) => update((current) => assignMarlHandOwner(current, hand.id, Number(event.target.value)))}
            >
              {agents.map((agent, index) => <option value={index} key={agent.name}>{agent.name}</option>)}
            </select>
          </div>
        );
      })}
    </div>
  );
}

function VisibilityEditor({ agent, index, hands, set }) {
  const mode = Array.isArray(agent.visible_teammate_hands) ? "custom" : agent.visible_teammate_hands;
  const teammateHands = hands.filter((hand) => !agent.hand_ids.includes(hand.id));
  const setMode = (nextMode) => {
    set(
      ["design", "config", "agents", index, "visible_teammate_hands"],
      nextMode === "custom" ? teammateHands.map((hand) => hand.id) : nextMode,
    );
  };
  const toggleHand = (handId) => {
    const current = Array.isArray(agent.visible_teammate_hands) ? agent.visible_teammate_hands : [];
    const next = current.includes(handId)
      ? current.filter((id) => id !== handId)
      : [...current, handId];
    set(["design", "config", "agents", index, "visible_teammate_hands"], next);
  };
  return (
    <div className="visibility-editor">
      <ChoiceRow
        label="Teammate hand visibility"
        value={mode}
        options={["none", "boundary", "all", "custom"]}
        onChange={setMode}
      />
      {mode === "custom" && (
        <div className="chip-selector chip-selector--compact">
          {teammateHands.map((hand) => (
            <button
              type="button"
              className={agent.visible_teammate_hands.includes(hand.id) ? "active" : ""}
              aria-pressed={agent.visible_teammate_hands.includes(hand.id)}
              key={hand.id}
              onClick={() => toggleHand(hand.id)}
            >
              H{hand.id + 1} · {hand.name}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function RangeEditor({ agents, hands, set, showVisibility }) {
  return (
    <div className="agent-range-map field--wide">
      {agents.map((agent, index) => (
        <article className="agent-range" key={agent.name}>
          <header><i className={`agent-swatch agent-swatch--${index + 1}`} /><strong>{agent.name}</strong></header>
          <div className="agent-range__label"><span>Action</span></div>
          <RangeSlider
            range={agent.action_key_range}
            agentIndex={index}
            kind="action"
            onChange={(value) => set(["design", "config", "agents", index, "action_key_range"], value)}
          />
          <div className="agent-range__label"><span>Observation</span></div>
          <RangeSlider
            range={agent.observation_key_range}
            agentIndex={index}
            kind="observation"
            onChange={(value) => set(["design", "config", "agents", index, "observation_key_range"], value)}
          />
          {showVisibility && <VisibilityEditor agent={agent} index={index} hands={hands} set={set} />}
        </article>
      ))}
    </div>
  );
}

export default function MarlDesign({ fullConfig, config, set, update, panel }) {
  const hands = fullConfig.task.embodiment.hands;
  const agentsPanel = panel("marl.agents");
  const ownershipPanel = panel("marl.ownership");
  const rangePanel = panel("marl.ranges");

  return (
    <>
      <Section title="Number of Agents" {...agentsPanel}>
        <NumberField
          label="Agents"
          value={config.num_agents}
          min={1}
          max={hands.length}
          onChange={(value) => update((current) => applyMarlAgentCount(current, value))}
        />
        {agentsPanel.advanced && (
          <>
            <NumberField
              label="Collision penalty"
              value={config.coordination.inter_agent_collision_penalty_coefficient}
              min={0}
              step={0.05}
              onChange={(value) => set("design.config.coordination.inter_agent_collision_penalty_coefficient", value)}
            />
          </>
        )}
      </Section>

      <Section title="Hand assignment" {...ownershipPanel}>
        <SelectField
          label="Sustain owner"
          value={config.sustain_owner}
          onChange={(value) => set("design.config.sustain_owner", value)}
          options={config.agents.map((agent) => ({ value: agent.name, label: agent.name }))}
        />
        <OwnershipEditor hands={hands} agents={config.agents} update={update} />
        {ownershipPanel.advanced && config.agents.map((agent, index) => (
          <TextField
            key={agent.name}
            label={`Agent ${index + 1} name`}
            value={agent.name}
            onChange={(value) => update((current) => {
              const next = structuredClone(current);
              const oldName = next.design.config.agents[index].name;
              next.design.config.agents[index].name = value;
              if (next.design.config.sustain_owner === oldName) {
                next.design.config.sustain_owner = value;
              }
              return next;
            })}
          />
        ))}
      </Section>

      <Section title="Action & observation ranges" {...rangePanel}>
        <RangeEditor
          agents={config.agents}
          hands={hands}
          set={set}
          showVisibility={rangePanel.advanced}
        />
      </Section>
    </>
  );
}
