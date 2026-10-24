import safetyProfiles from "./safetyParameterProfiles.json" with { type: "json" };

const PPO_SB3 = {
  learning_rate: 0.0003,
  n_steps: 2048,
  batch_size: 64,
  n_epochs: 10,
  gae_lambda: 0.95,
  clip_range: 0.2,
  clip_range_vf: null,
  normalize_advantage: true,
  ent_coef: 0.0,
  vf_coef: 0.5,
  max_grad_norm: 0.5,
  use_sde: false,
  sde_sample_freq: -1,
  target_kl: null,
  stats_window_size: 100,
  net_arch: [64, 64],
};

const OFF_POLICY_COMMON = {
  buffer_size: 1_000_000,
  learning_starts: 100,
  batch_size: 256,
  train_freq: 1,
  gradient_steps: -1,
  optimize_memory_usage: false,
  n_steps: 1,
  stats_window_size: 100,
};

const SAC_COMMON = {
  ...OFF_POLICY_COMMON,
  learning_rate: 0.0003,
  tau: 0.005,
  ent_coef: "auto",
  target_entropy: "auto",
  use_sde: false,
  sde_sample_freq: -1,
  use_sde_at_warmup: false,
  net_arch: [256, 256],
};

const STANDARD = {
  ppo: PPO_SB3,
  sac: { ...SAC_COMMON, target_update_interval: 1 },
  td3: {
    ...OFF_POLICY_COMMON,
    learning_rate: 0.001,
    tau: 0.005,
    policy_delay: 2,
    target_policy_noise: 0.2,
    target_noise_clip: 0.5,
    net_arch: [400, 300],
  },
  crossq: {
    ...OFF_POLICY_COMMON,
    learning_rate: 0.001,
    ent_coef: "auto",
    target_entropy: "auto",
    policy_delay: 3,
    use_sde: false,
    sde_sample_freq: -1,
    use_sde_at_warmup: false,
    net_arch: [256, 256],
  },
  tqc: { ...SAC_COMMON, target_update_interval: 1, top_quantiles_to_drop_per_net: 2 },
};

const ROBUST_PPO = {
  net_arch: [256, 256],
  n_steps: 2048,
  n_epochs: 10,
  gae_lambda: 0.95,
  clip_epsilon: 0.2,
  entropy_coeff: 0.0,
  critic_coeff: 0.5,
};

const ROBUST_SAC = {
  net_arch: [256, 256],
  batch_size: 256,
  buffer_size: 1_000_000,
  learning_starts: 5_000,
  tau: 0.005,
  utd: 1,
  learning_rate: 0.0003,
};

const ROBUST = {
  ppo: ROBUST_PPO,
  sac: ROBUST_SAC,
  eppo: { ...ROBUST_PPO, eppo_mode: "cor", kappa: 0.05, evidential_reg: 0.001 },
  a2p_sac: {
    ...ROBUST_SAC,
    a2p_epsilon_mode: "adaptive",
    a2p_epsilon: 0.1,
    adversary_learning_rate: 0.0003,
    adversary_update_freq: 10,
    a2p_warmup_steps: null,
  },
  ompo: {
    ...ROBUST_SAC,
    ompo_local_buffer_size: 1_000,
    ompo_updates_per_step: 3,
    ompo_discriminator_observation: "dynamic",
    ompo_exponent: 1.5,
    ompo_correction_coeff: 0.001,
    ompo_reward_max: 1.0,
  },
  scpo: { ...ROBUST_SAC, scpo_state_noise: 0.005, scpo_gradient_observation: "dynamic" },
};

const MARL_PPO = {
  learning_rate: 0.0003,
  critic_learning_rate: 0.0003,
  train_batch_size: 4_000,
  minibatch_size: 4_000,
  num_epochs: 5,
  adam_epsilon: 0.00001,
  gae_lambda: 0.95,
  clip_param: 0.2,
  value_clip_param: 0.2,
  value_loss_coefficient: 1.0,
  entropy_coefficient: 0.0,
  use_kl_loss: false,
  gradient_clip: 10.0,
  gradient_clip_mode: "global_norm",
  hidden_sizes: [256, 256],
  activation: "tanh",
  hidden_orthogonal_gain: Math.sqrt(2),
  policy_output_gain: 0.01,
  value_output_gain: 1.0,
  initial_log_std: 0.0,
  log_std_min: -5.0,
  log_std_max: 2.0,
  input_layer_norm: true,
  value_norm: true,
  value_norm_beta: 0.99999,
  value_norm_epsilon: 0.00001,
  value_norm_variance_floor: 0.01,
};

const FACMAC = {
  actor_hidden_sizes: [400, 400],
  utility_hidden_sizes: [400, 400],
  actor_lr: 0.001,
  critic_lr: 0.001,
  adam_epsilon: 0.01,
  grad_clip: 0.5,
  replay_capacity: 1_000_000,
  batch_size: 100,
  buffer_warmup: 1_000,
  random_action_steps: 10_000,
  collect_steps: 24,
  updates_per_env_step: 1.0,
  tau: 0.001,
  noise_std: 0.1,
  action_l2: 0.001,
  mixer_embed: 64,
  hypernet_embed: 64,
  max_replay_gib: 32.0,
  checkpoint_replay_transitions: 50_000,
};

const MARL = {
  ippo: MARL_PPO,
  "ippo-rllib-module": MARL_PPO,
  mappo: MARL_PPO,
  "mappo-own-critic": MARL_PPO,
  happo: MARL_PPO,
  a2po: {
    ...MARL_PPO,
    order_score_epsilon: 0.00000001,
    preceding_ratio_clip: 0.1,
    adaptive_clip_weight: 0.5,
  },
  facmac: FACMAC,
  "ppo-monolithic": MARL_PPO,
};

const PROFILES = {
  standard: STANDARD,
  robust: ROBUST,
  safety: safetyProfiles.algorithms,
  marl: MARL,
};

export function completeParameterTemplate(track, algorithm) {
  return structuredClone(PROFILES[track]?.[algorithm] ?? {});
}
