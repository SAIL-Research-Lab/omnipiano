import { ChoiceRow, NumberField, NumberInput, Section, SelectField, TextField } from "../fields";
import { applyMarlAgentCount, assignMarlHandOwner } from "../config";

function rangeOverlap(left, right) {
  return Math.max(0, Math.min(left[1], right[1]) - Math.max(left[0], right[0]) + 1);
}

function totalPairwiseOverlap(agents, key) {
  let total = 0;
  for (let left = 0; left < agents.length; left += 1) {
    for (let right = left + 1; right < agents.length; right += 1) {
      total += rangeOverlap(agents[left][key], agents[right][key]);
    }
  }
  return total;
}

function RangeBar({ range, agentIndex, kind }) {
  const left = ((range[0] - 1) / 88) * 100;
  const width = ((range[1] - range[0] + 1) / 88) * 100;
  return (
    <div className="agent-range__track" aria-label={`${kind} keys ${range[0]} to ${range[1]}`}>
      <span
        className={`agent-range__bar agent-range__bar--${agentIndex + 1} agent-range__bar--${kind}`}
        style={{ left: `${left}%`, width: `${width}%` }}
      >
        {range[0]}–{range[1]}
      </span>
    </div>
  );
}

function RangeInputs({ label, value, onChange }) {
  return (
    <div className="mini-range-editor">
      <span>{label}</span>
      <NumberInput
        min="1"
        max="88"
        aria-label={`${label} start`}
        value={value[0]}
        onChange={(next) => onChange([next, value[1]])}
      />
      <i>–</i>
      <NumberInput
        min="1"
        max="88"
        aria-label={`${label} end`}
        value={value[1]}
        onChange={(next) => onChange([value[0], next])}
      />
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
            <span className="ownership-hand__number">H{hand.id + 1}</span>
            <div><strong>{hand.name}</strong><small>Keys {hand.initial_key_range[0]}–{hand.initial_key_range[1]}</small></div>
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
      <small>Moving an agent's last hand swaps ownership.</small>
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
      <div className="piano-scale"><span>1 · A0</span><span>44</span><span>88 · C8</span></div>
      {agents.map((agent, index) => (
        <article className="agent-range" key={agent.name}>
          <header><i className={`agent-swatch agent-swatch--${index + 1}`} /><strong>{agent.name}</strong><small>{agent.hand_ids.map((id) => `H${id + 1}`).join(" + ")}</small></header>
          <div className="agent-range__label"><span>Action</span><small>wrist key range</small></div>
          <RangeBar range={agent.action_key_range} agentIndex={index} kind="action" />
          <RangeInputs
            label="Action keys"
            value={agent.action_key_range}
            onChange={(value) => set(["design", "config", "agents", index, "action_key_range"], value)}
          />
          <div className="agent-range__label"><span>Observation</span><small>visible keyboard state</small></div>
          <RangeBar range={agent.observation_key_range} agentIndex={index} kind="observation" />
          <RangeInputs
            label="Observation keys"
            value={agent.observation_key_range}
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
  const actionOverlap = totalPairwiseOverlap(config.agents, "action_key_range");
  const observationOverlap = totalPairwiseOverlap(config.agents, "observation_key_range");

  return (
    <>
      <section className="marl-stats" aria-label="Multi-agent configuration summary">
        <div className="metric-strip">
          <div><span>Hands / agents</span><strong>{hands.length} / {config.num_agents}</strong></div>
          <div><span>Action overlap</span><strong>{actionOverlap} keys</strong></div>
          <div><span>Observation overlap</span><strong>{observationOverlap} keys</strong></div>
          <div><span>Critic</span><strong>{config.critic_mode.replaceAll("_", " ")}</strong></div>
        </div>
      </section>

      <Section title="Number of Agents" {...agentsPanel}>
        <NumberField
          label="Agents"
          value={config.num_agents}
          min={1}
          max={hands.length}
          onChange={(value) => update((current) => applyMarlAgentCount(current, value))}
          hint={`Maximum ${hands.length}.`}
        />
        {agentsPanel.advanced && (
          <>
            <SelectField
              label="Policy sharing"
              value={config.policy_sharing}
              onChange={(value) => set("design.config.policy_sharing", value)}
              options={["independent", "shared"]}
            />
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

      <Section
        title="Action & observation ranges"
        description="Action ranges constrain wrists; fingertips may extend beyond them."
        {...rangePanel}
      >
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
