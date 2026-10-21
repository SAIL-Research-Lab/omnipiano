import { Section } from "../fields";
import { applyStandardVariant, setStandardHandMobile } from "../config";

const VARIANTS = [
  {
    value: "general",
    title: "General control",
    description: "Mobile hands, full keyboard.",
  },
  {
    value: "mobility",
    title: "Mobility study",
    description: "Choose which hands may reposition.",
  },
  {
    value: "restricted_range",
    title: "Restricted range",
    description: "Keep hands in fixed keyboard partitions.",
  },
];

export default function StandardDesign({ fullConfig, config, update }) {
  const chooseVariant = (variant) => {
    update((current) => applyStandardVariant(current, variant));
  };

  return (
    <Section
      title="Task variant"
    >
      <div className="design-option-grid field--wide">
        {VARIANTS.map((variant) => (
          <button
            type="button"
            className={`design-option ${config.benchmark_variant === variant.value ? "active" : ""}`}
            aria-pressed={config.benchmark_variant === variant.value}
            key={variant.value}
            onClick={() => chooseVariant(variant.value)}
          >
            <span className="design-option__mark" aria-hidden="true" />
            <strong>{variant.title}</strong>
            <small>{variant.description}</small>
          </button>
        ))}
      </div>
      <div className="design-summary field--wide">
        <span>Current task</span>
        <strong>{fullConfig.task.embodiment.num_hands} hands · {fullConfig.task.embodiment.mobility.replaceAll("_", " ")}</strong>
      </div>
      {config.benchmark_variant === "mobility" && (
        <div className="chip-field field--wide">
          <span className="field__label">Hands allowed to move</span>
          <div className="chip-selector">
            {fullConfig.task.embodiment.hands.map((hand) => (
              <button
                type="button"
                className={hand.mobile ? "active" : ""}
                aria-pressed={hand.mobile}
                key={hand.id}
                onClick={() => update((current) => setStandardHandMobile(current, hand.id, !hand.mobile))}
              >
                H{hand.id + 1} · {hand.name}
              </button>
            ))}
          </div>
        </div>
      )}
    </Section>
  );
}
