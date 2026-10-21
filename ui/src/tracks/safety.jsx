import { ChoiceRow, NumberField, Section } from "../fields";
import { getSafetyPaperBudget } from "../config";

const COSTS = [
  { value: "range_event", title: "Range event", description: "Binary cost when any selected joint leaves its safe interval.", family: "range" },
  { value: "range_fraction", title: "Range fraction", description: "Fraction of selected joints outside their safe intervals.", family: "range" },
  { value: "range_excess", title: "Range excess", description: "Continuous cost proportional to violation magnitude.", family: "range" },
  { value: "power_excess", title: "Power excess", description: "Continuous actuator-power cost above a threshold.", family: "power" },
];

const JOINTS = ["THJ2", "FFJ2", "MFJ2", "RFJ2", "LFJ2"];
function toggleValue(values, value, allowEmpty = false) {
  if (values.includes(value)) {
    if (!allowEmpty && values.length === 1) return values;
    return values.filter((item) => item !== value);
  }
  return [...values, value];
}

export default function SafetyDesign({ fullConfig, config, set, update, panel }) {
  const cost = config.primary_cost;
  const costPanel = panel("safety.cost");
  const isRangeCost = cost.type.startsWith("range_");
  const suggestedBudget = getSafetyPaperBudget(fullConfig, cost.type);

  const changeCost = (type) => update((current) => {
    if (current.design.config.primary_cost.type === type) return current;
    const next = structuredClone(current);
    const nextCost = next.design.config.primary_cost;
    nextCost.type = type;
    if (type.startsWith("range_")) {
      nextCost.safe_interval_fraction ??= 0.5;
      nextCost.power_threshold = null;
      if (!nextCost.target_joints.length) nextCost.target_joints = [...JOINTS];
    } else {
      nextCost.safe_interval_fraction = null;
      nextCost.power_threshold ??= 4.0;
      nextCost.target_joints = [];
      next.task.reward.energy_regularization.enabled = false;
    }
    const suggested = getSafetyPaperBudget(next, type);
    if (nextCost.budget_mode === "paper_default" && suggested) {
      nextCost.episodic_budget = suggested.value;
    }
    return next;
  });

  const setBudgetMode = (budgetMode) => update((current) => {
    const next = structuredClone(current);
    next.design.config.primary_cost.budget_mode = budgetMode;
    if (budgetMode === "paper_default" && suggestedBudget) {
      next.design.config.primary_cost.episodic_budget = suggestedBudget.value;
    }
    return next;
  });

  return (
    <>
      <Section
        title="Safety cost"
        {...costPanel}
      >
        <div className="cost-grid field--wide">
          {COSTS.map((item) => (
            <button
              type="button"
              className={`cost-option ${cost.type === item.value ? "active" : ""}`}
              aria-pressed={cost.type === item.value}
              key={item.value}
              onClick={() => changeCost(item.value)}
            >
              <span className={`cost-option__mark cost-option__mark--${item.family}`} aria-hidden="true" />
              <strong>{item.title}</strong>
              <small>{item.description}</small>
            </button>
          ))}
        </div>

        <div className="chip-field field--wide">
          <span className="field__label">Affected hands</span>
          <div className="chip-selector">
            {fullConfig.task.embodiment.hands.map((hand) => (
              <button
                type="button"
                className={cost.target_hand_ids.includes(hand.id) ? "active" : ""}
                aria-pressed={cost.target_hand_ids.includes(hand.id)}
                key={hand.id}
                onClick={() => set(
                  "design.config.primary_cost.target_hand_ids",
                  toggleValue(cost.target_hand_ids, hand.id),
                )}
              >
                <span>H{hand.id + 1}</span>{hand.name}
              </button>
            ))}
          </div>
        </div>

        {isRangeCost ? (
          <>
            <div className="chip-field field--wide">
              <span className="field__label">Finger joints</span>
              <div className="chip-selector chip-selector--joints">
                {JOINTS.map((joint) => (
                  <button
                    type="button"
                    className={cost.target_joints.includes(joint) ? "active" : ""}
                    aria-pressed={cost.target_joints.includes(joint)}
                    key={joint}
                    onClick={() => set(
                      "design.config.primary_cost.target_joints",
                      toggleValue(cost.target_joints, joint),
                    )}
                  >
                    {joint}
                  </button>
                ))}
              </div>
            </div>
            <NumberField
              label="Safe interval fraction"
              value={cost.safe_interval_fraction}
              min={0.01}
              max={1}
              step={0.05}
              onChange={(value) => set("design.config.primary_cost.safe_interval_fraction", value)}
              hint="Fraction of each joint's mechanical range considered safe."
            />
          </>
        ) : (
          <>
            <NumberField
              label="Power threshold"
              value={cost.power_threshold}
              min={0.01}
              step={0.1}
              onChange={(value) => set("design.config.primary_cost.power_threshold", value)}
              hint="Uses the backend cost's native power unit."
            />
            <div className="inline-callout">
              <strong>Energy reward disabled</strong>
              <span>This avoids optimizing the same actuator-power objective twice.</span>
            </div>
          </>
        )}
        {costPanel.advanced && (
          <NumberField
            label="Cost coefficient"
            value={cost.coefficient}
            min={0}
            step={0.1}
            onChange={(value) => set("design.config.primary_cost.coefficient", value)}
          />
        )}
      </Section>

      <Section
        title="Cost budget"
      >
        <ChoiceRow
          label="Budget source"
          value={cost.budget_mode}
          options={[
            ...(suggestedBudget ? [{ value: "paper_default", label: "Paper default" }] : []),
            { value: "custom", label: "Custom" },
          ]}
          onChange={setBudgetMode}
        />
        {cost.budget_mode === "paper_default" && suggestedBudget ? (
          <div className="budget-formula field--wide">
            <div><span>Episode horizon</span><strong>{suggestedBudget.horizon}</strong></div>
            <i>×</i>
            <div><span>Cost rate</span><strong>{suggestedBudget.rate}</strong></div>
            <i>=</i>
            <div className="budget-formula__result"><span>Budget</span><strong>{suggestedBudget.value}</strong></div>
          </div>
        ) : (
          <NumberField
            label="Episodic budget"
            value={cost.episodic_budget}
            min={0}
            step={0.01}
            onChange={(value) => set("design.config.primary_cost.episodic_budget", value)}
            hint={cost.budget_mode === "paper_default" ? "No paper horizon is registered for this piece; review this value manually." : undefined}
          />
        )}
      </Section>
    </>
  );
}
