import Ajv2020 from "ajv/dist/2020.js";
import schema from "../../omnipiano/experiments/contract/experiment-v3.schema.json";

const validateSchema = new Ajv2020({ allErrors: true, strict: false }).compile(schema);

function label(error) {
  const path = error.instancePath ? error.instancePath.replaceAll("/", ".") : "configuration";
  return `${path}: ${error.message}`;
}

export function validateExperiment(config) {
  const errors = [];
  const warnings = [];
  if (!validateSchema(config)) errors.push(...validateSchema.errors.map(label));
  if (config.task?.source === "custom") {
    const hands = config.task.embodiment?.hands ?? [];
    if (hands.length !== config.task.embodiment?.num_hands) {
      errors.push("task.embodiment: hand count does not match num_hands");
    }
    hands.forEach((hand) => {
      if (hand.initial_key_range?.[0] > hand.initial_key_range?.[1]) {
        errors.push(`task.embodiment: ${hand.name} has a reversed key range`);
      }
    });
  }
  if (config.run?.tracking?.provider === "none" && config.run.tracking.mode !== "disabled") {
    errors.push("run.tracking.mode must be disabled when tracking is off");
  }
  return { errors: [...new Set(errors)], warnings };
}
