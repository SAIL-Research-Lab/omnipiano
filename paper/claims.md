# OmniPiano paper — core claims(单一事实来源)

> 每条 claim:paper-ready 英文措辞 + 证据指针 + 状态。
> 状态:`[verified]` 有实测数字/测试锁定 · `[pending]` 实验排队中 · `[design]` 设计性质、无需实验。
> abstract 只引用 `[verified]` / `[design]`。

## C0 — Problem statement(论文动机)

**Claim**: RL research on safety, robustness, and multi-agent coordination has advanced
on *separate* benchmarks with deliberately simple dynamics (point/car navigation in
Safety-Gymnasium, low-DoF locomotion in Robust-Gymnasium, abstracted physics in most
MARL suites). The community lacks a benchmark where these evaluation axes meet a task
that is itself hard: fine-grained, high-frequency, contact-rich dexterous control.
OmniPiano fills this gap. `[design]`

- 对比锚点:RoboPianist (task only, CoRL'23) / Safety-Gymnasium (NeurIPS'23 D&B) /
  Robust-Gymnasium (ICLR'25) / SMAC & MPE 系 MARL。
- 注意措辞:是 "these axes have not been combined on a fine-manipulation task",
  **不是** "没有任何 benchmark 有这些轴"——防 reviewer 反例。

## C1 — Unified benchmark(总纲 claim,abstract 主干)

**Claim**: OmniPiano is a unified RL benchmark built on simulated Shadow-Hand piano
playing that covers four axes in one codebase, one Gymnasium-standard API, and one
ground-truth metric: (i) a **morphology ladder** of 2–5 hands (45–111+ continuous
action dims), (ii) **safe RL** with six constraint families, (iii) **robust RL** with
observation/action/reward perturbations aligned with Robust-Gymnasium, and (iv)
**cooperative multi-agent** variants via PettingZoo. `[design]`

- API/framework-agnostic:SB3 / OmniSafe / RLlib / CleanRL 均已实际接入或有模板
  (README "Training + evaluation")。
- 安全约束族(README):joint magnitude limits、hand-hand collision(binary + 连续力)、
  actuator power、joint-injury budgets、shared per-joint ceilings、summed-chain budgets。

## C2 — Noise-immune ground-truth metric(方法论卖点,差异化最强)

**Claim**: The headline metric — note-level F1 computed from ground-truth physics
(piano-key activation vs. the MIDI score) — is structurally immune to all three noise
channels: observation, action, and reward perturbations cannot contaminate the
measurement, only honestly change the trajectory being measured. Return-based metrics
lack this property (reward noise pollutes `ep_return`); OmniPiano additionally logs
`ep_return_true` and per-channel noise sums, a strict superset of Robust-Gymnasium's
logging, whose recorded return under reward noise is the polluted value with no clean
counterpart. `[verified]`

- 证据:`robust_task_design.md` §14(指标×通道污染矩阵,代码核实)+ §14.6
  (RG 上游逐行对照,af6de64)。
- "污染-as-measurement vs 诚实轨迹变化" 的二分是可写进 paper 的概念贡献。

## C3 — Morphology ladder + partition finding(classic 腿)

**Claim**: Scaling piano playing from 2 to 5 hands exposes a reward-design pathology:
with OT-based multi-hand reward and no spatial constraint, outer hands collapse to
idle "winner-takes-all" behavior. Physically enforced register partitions (Level-1,
MuJoCo joint-range clamps) restore full-hand engagement — the 5-hand partitioned
policy attains the ladder's best F1 (0.46) despite the largest action space (111-dim).
`[verified]`(单 seed demo 数字;多 seed 待补)

- 数字(README demos):3-hand proto SAC 5M F1=0.52 · 4-hand proto TQC 10M F1=0.42 ·
  4-hand L1 TQC 5M F1=0.39 · 5-hand L1 TQC 8M **F1=0.46**。
- 双手基线:ClairDeLune PPO 5M clean F1=0.626(robust_task_design §10)。
- 机理与 hard-core/soft-boundary 语义:`static_partition_design.md` §3(PAPER-CRITICAL)。

## C4 — Robust 腿:三大实证发现(robust_task_design §10,matched eval,PPO,seed=0)

**C4a — Distribution ordering**: At equal numerical level, difficulty orders
Shift (easy) < Uniform < Gaussian (hard), consistently across action and reward
channels — policies learn to compensate a constant bias (empirical std 0) but cannot
fight white noise. A-channel: 0.704/0.371/0.277; R-channel: 0.589/0.300/0.117.
`[verified]`(单 seed)

**C4b — Channel sensitivity**: Action ≈ Reward ≫ Observation. Obs noise at the
highest level costs only 4–14% F1 (policies partially filter sensor noise; goal dims
untouched); Gaussian action/reward noise costs 56–81%. `[verified]`(单 seed)

**C4c — Noise induces conservatism**: Under every channel, precision stays high
(0.90–0.999) while recall collapses — perturbed policies miss notes rather than play
wrong ones. `[verified]`

**C4d — Directional bias is not degradation**: A constant +0.15 action shift *helps*
(F1 0.704 > clean 0.626) while −0.15 hurts (0.553) — monotone through clean, near
symmetric. Systematic bias is qualitatively different from zero-mean noise and
challenges the "perturbation ⇒ degradation" framing; benchmarks that only test one
sign miss half the picture. `[verified]`(方向对照 N15 已确认;多 seed 待补)

**C4e — Reward noise cripples learning while returns look healthy**: R-Gauss-P50
destroys the value function (explained_variance 0.27, F1 0.117) yet received return
stays within ~10% of clean; R-Shift inflates return above clean (+294) at F1 0.589.
Without a noise-immune metric this failure is invisible. `[verified]`(连接 C2 的杀手案例)

- Caveats 必须 disclose(§10):单 seed;仅 matched eval(scale sweep 曲线 pending);
  仅最高档 P15/P50;obs 噪声只打 position-only 本体感觉通道(与 RoboPianist 对齐);
  σ 的物理单位跨通道不同(§4.5)。

## C5 — Safety 腿(实验排队中,abstract 用能力措辞)

**Claim (capability)**: Six constraint families expose cost signals via `info` without
touching reward, with cost-vs-return trade-offs tunable per task (cost limits set at
10% of unconstrained PPO cost). `[design]`

**Claim (empirical, 占位)**: PPO-Lagrangian 的 "safety price" 及 OT-dodge 现象
(policy 通过少弹音符规避 cost 而非真正 trade-off)。`[pending]`
— 3 组 PPOLag 10%-baseline 实验排队(ForearmInjury cl=76 / ThumbInjury cl=85 /
PolonaiseOp53-Power cl=814)+ CollisionSafe-v1 disable_forearm_reward 消融。

## C6 — Multi-agent 腿

**Claim**: OmniPiano decomposes N-hand control into cooperative agents by morphology
(e.g. secondo/primo duet pairs) under PettingZoo's ParallelEnv, with a tunable
shared-territory axis (disjoint per-agent clamps with an emergent shared zone), and
trains with off-the-shelf MARL (RLlib MAPPO). `[verified]`(4-hand Duet Territorial
MAPPO 5M eval reward 768;F1 surface 修复在 MA bug 队列中)

- `multi_agent_design.md` §5(shared-zone-width axis,PAPER-CRITICAL)、§6(安全约束映射)。

## C7 — Scope 卫语句(防 reviewer)

- Adversarial perturbations 与 dynamics randomization **有意 out of scope**,理由已成文
  (robust_task_design §10 预期 claim 2、§11):对抗需 per-method 基建;dynamics 随机化
  动机是 sim-to-real(纯仿真基准不适用)且纠缠 F1 语义。
- Reduced action space 不启用:全 DoF Shadow Hand 是 benchmark 定义的一部分
  (thumb-budget 安全任务依赖 THJ1/THJ5)。

## Abstract 取舍建议(讨论用)

- 主干 = C0 + C1;方法论差异化 = C2;findings 精选 2–3 条(候选:C3 winner-takes-all、
  C4d directional bias、C4e reward-noise invisibility——三条分属三条腿,覆盖面好)。
- C5 empirical 未出数前,abstract 安全腿只用能力措辞。
