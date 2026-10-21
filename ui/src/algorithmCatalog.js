// Curated UI snapshot of algorithms exposed by the repository's existing
// trainers. The canonical UI JSON is not itself a launchable backend config.
const PPO = {
  learning_rate: 0.0003,
  train_batch_size: 4000,
  minibatch_size: 4000,
  num_epochs: 5,
  gae_lambda: 0.95,
  clip_param: 0.2,
  entropy_coefficient: 0,
  hidden_sizes: [256, 256],
};

const MARL_PPO = {
  ...PPO,
  critic_learning_rate: 0.0003,
  adam_epsilon: 0.00001,
  value_clip_param: 0.2,
  use_kl_loss: false,
  gradient_clip: 10,
  input_layer_norm: true,
  value_norm: true,
};

const SAC = {
  learning_rate: 0.0003,
  batch_size: 256,
  buffer_size: 1000000,
  learning_starts: 5000,
  hidden_sizes: [256, 256, 256],
};

const SAFE_PPO = {
  learning_rate: 0.0003,
  lagrangian_learning_rate: 0.035,
  train_batch_size: 4000,
  minibatch_size: 4000,
  num_epochs: 5,
  gae_lambda: 0.95,
  clip_param: 0.2,
  hidden_sizes: [256, 256],
};

const ALGORITHMS = {
  standard: [
    { name: "ppo", label: "PPO", framework: "stable_baselines3", family: "ppo", status: "template", parameters: PPO },
    { name: "sac", label: "SAC", framework: "stable_baselines3", family: "sac", status: "template", parameters: SAC },
    { name: "tqc", label: "TQC", framework: "stable_baselines3", family: "sac", status: "template", parameters: { ...SAC, n_quantiles: 25, n_critics: 2, top_quantiles_to_drop_per_net: 2 } },
  ],
  robust: [
    { name: "ppo", label: "PPO", framework: "stable_baselines3", family: "ppo", status: "template", parameters: PPO },
    { name: "sac", label: "SAC", framework: "stable_baselines3", family: "sac", status: "template", parameters: SAC },
    { name: "tqc", label: "TQC", framework: "stable_baselines3", family: "sac", status: "template", parameters: { ...SAC, n_quantiles: 25, n_critics: 2, top_quantiles_to_drop_per_net: 2 } },
  ],
  safety: [
    { name: "PPOLag", label: "PPO-Lagrangian", framework: "omnisafe", family: "safe_policy_optimization", status: "template", preset: "paper_default", parameters: SAFE_PPO },
    { name: "CPO", label: "CPO", framework: "omnisafe", family: "safe_policy_optimization", status: "template", parameters: {} },
    { name: "SACLag", label: "SAC-Lagrangian", framework: "omnisafe", family: "safe_policy_optimization", status: "template", parameters: {} },
  ],
  marl: [
    { name: "ippo", label: "IPPO", framework: "rllib", family: "ppo", status: "supported", critic: "local", parameters: MARL_PPO },
    { name: "mappo", label: "MAPPO", framework: "rllib", family: "ppo", status: "supported", critic: "centralized", parameters: MARL_PPO },
    { name: "mappo-own-critic", label: "MAPPO · own-critic ablation", framework: "rllib", family: "ppo", status: "supported", critic: "own_observation_ablation", parameters: MARL_PPO, note: "The critic reads each agent's own observation." },
    { name: "ippo-rllib-module", label: "IPPO · RLlib module control", framework: "rllib", family: "ppo", status: "supported", critic: "local", parameters: { ...MARL_PPO, value_clip_param: 1000 }, note: "Uses a different network from the matched IPPO/MAPPO comparison." },
    { name: "happo", label: "HAPPO", framework: "rllib", family: "ppo", status: "experimental", critic: "centralized", parameters: MARL_PPO, note: "Experimental; requires an explicit validation run before paper results." },
    { name: "mappo-noshuffle", label: "MAPPO · no-shuffle control", framework: "rllib", family: "ppo", status: "experimental", critic: "centralized", parameters: MARL_PPO, note: "Experimental HAPPO comparison control." },
    { name: "happo-m1-control", label: "HAPPO · M=1 control", framework: "rllib", family: "ppo", status: "experimental", critic: "centralized", parameters: MARL_PPO, note: "Validation-only; not a publishable baseline." },
  ],
};

export function algorithmsForTrack(track, includeExperimental = false) {
  return ALGORITHMS[track].filter((item) => includeExperimental || item.status !== "experimental");
}

export function findAlgorithm(track, name) {
  return ALGORITHMS[track]?.find((item) => item.name === name);
}

export function selectAlgorithm(source, name) {
  const next = structuredClone(source);
  const item = findAlgorithm(next.design.type, name);
  if (!item) return next;
  if (
    next.algorithm.name === item.name
    && next.algorithm.framework === item.framework
    && next.algorithm.family === item.family
    && (next.design.type !== "marl" || next.design.config.critic_mode === item.critic)
  ) return next;
  next.algorithm = {
    framework: item.framework,
    name: item.name,
    family: item.family,
    preset: item.preset ?? "omnipiano_default",
    parameters: structuredClone(item.parameters),
  };
  if (next.design.type === "marl") next.design.config.critic_mode = item.critic;
  return next;
}
