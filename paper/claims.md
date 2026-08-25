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
- 安全约束:**paper 口径 = 4 族 / 7 类**(abstract v3 定稿;`constraints.py` 代码核实
  2026-07-20)。按物理量归组,取代 README 的 6 族口径(**发布前 README 需统一**):
  | 族(paper 口径) | 约束类 |
  |---|---|
  | joint magnitude | `JointMagnitudeConstraint` + `MultiJointSharedMagnitudeConstraint` + `MultiJointSummedMagnitudeConstraint` |
  | inter-hand collision force | `HandCollisionConstraint`(binary)+ `HandCollisionForceConstraint`(连续力) |
  | actuator power | `TotalActuatorPowerConstraint` |
  | joint-injury risk | `InjuredJointPowerConstraint` |

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

## C4 — Robust 腿:实证发现(`paper/robust_notes.md` §5,matched eval,PPO,seed=0)

> **数字口径(2026-07-21 定稿)**:最终策略 + **10 集** deterministic matched eval
> (`eval_noise_scale=1.0`)。Shift 类任务集间方差恰为 0(常数偏置不抽随机数),
> Gaussian/Uniform 有真实方差。**旧版 claims 里的 A 通道数字已作废**——那批实验
> 是在另一种威胁模型(扰动对策略隐藏)下训练的,且混用了两种测量口径。

**C4a — Distribution ordering**: At equal numerical level, difficulty orders
Shift (easy) < Uniform < Gaussian (hard) on the **action and reward** channels —
policies learn to compensate a constant bias (which carries zero variance) but
cannot fight white noise. A-channel: 0.615/0.491/0.267; R-channel: 0.589/0.327/0.109.
**The observation channel is a deliberate exception worth reporting**: there the
order reverses at the easy end (Uniform 0.616 > Gaussian 0.562 > Shift 0.539) — a
constant sensor bias is *harder* than sensor white noise. `[verified]`(单 seed)

**C4b — Channel sensitivity**: Action ≈ Reward ≫ Observation. At the highest level,
Gaussian noise costs 57% (action) and 83% (reward) of F1 but only 10% on observation
— policies partially filter sensor noise, and the 979 goal dimensions are untouched.
`[verified]`(单 seed)

**C4c — Noise induces conservatism**: Under every channel, precision stays high
(0.92–0.99) while recall collapses — perturbed policies miss notes rather than play
wrong ones. The F1 drop is almost entirely a recall drop. `[verified]`

**C4d — Constant bias is sign-dependent across every channel, and the *helpful*
sign differs per channel**: what a constant offset does is decided by how its
direction aligns with the task's demands, not by its magnitude. Action ±0.15:
0.615 (−2%) vs 0.545 (−13%) — the negative side removes the full-press end of the
clipped action range, exactly what piano playing needs. Observation ±0.15: 0.539
(−14%) vs **0.677 (+8%, above clean)** — a negative bias makes the policy perceive
its fingers as *less* depressed than they are, so it presses more, lifting recall
0.506 → 0.554 at precision 0.994. Reward ±0.50: 0.589 (−6%) vs 0.493 (−21%).
Zero-mean noise cannot produce this; a benchmark that tests only one sign measures
only half the effect — and on the observation channel would report a 14% loss where
the opposite sign gives an 8% gain. `[verified]`(三通道各有正负对照;多 seed 待补)

**C4d-i — Constant reward offsets are not policy-invariant, and an ablation
localises the effect to the observation path**: with fixed 588-step episodes,
adding a constant to every step's reward adds a constant to every trajectory's
return, so classical MDP reasoning predicts an unchanged optimal policy. Both signs
nonetheless degrade performance (−6% at +0.50, −21% at −0.50). Re-running both with
the observation-augmentation disabled — which removes `obs["reward"]`, the only
path by which the perturbation reaches the policy as an *input*, while leaving it
in the learning signal — separates the two candidate mechanisms and gives an
**asymmetric answer**: the large negative-shift penalty vanishes (−21% → +7%
relative to the OAR-off baseline), whereas the small positive-shift penalty
persists (−6% → −9%). A constant reward offset therefore harms the policy chiefly
by perturbing an observation feature, not by corrupting the learning signal — which
makes the benchmark's threat-model declaration (what the OAR slots carry) a
determinant of measured robustness rather than an implementation detail.
`[verified]`(单 seed;正向那一格的小幅残余未归因,可能是 critic 动态或种子噪声)

**C4e — Reward noise cripples learning while returns look healthy**: R-Gauss-P50
destroys the value function (explained_variance 0.27) and drives F1 to 0.109 (−83%),
yet the received return stays within 9% of clean; R-Shift-P50 *inflates* return to
2141 (+16% above clean) at F1 0.589. Without a noise-immune metric this failure mode
is invisible. `[verified]`(连接 C2 的杀手案例,robust 腿最强的单条 claim)

**C4f — Compound perturbations compose differently by noise family**: with
*stochastic* noise the two-channel task falls *below both* single-channel
components (super-additive degradation) — AO-Gauss 0.177 vs A 0.267 / O 0.562;
AR-Uniform 0.232 vs A 0.491 / R 0.327. With *constant shifts* the compound instead
tracks its better component and the harmful one is masked (sub-additive) —
OR-Shift 0.672 ≈ O-Shift-N15 0.677, despite R-Shift-N50 alone costing 21%.
Robust-Gymnasium cannot express multi-channel perturbation at all, so this class of
interaction is unmeasurable there. `[verified]`(单 seed;同向单通道对照已补齐)

**C4g — Robustness numbers depend on whether the perturbation is observable**
(threat-model control): the same +0.15 action shift *helps* when the disturbance is
hidden from the policy (F1 0.704 vs clean 0.626) but is neutral once the policy
observes its executed action (0.615). The harmful direction is unaffected (−0.15:
0.553 → 0.545), consistent with C4d's clipping mechanism — the compensable side gets
compensated away, the capability loss does not. A benchmark must therefore declare
its threat model. `[verified for the shift channel]`(两侧均为确定性测量,协议无关;
gauss/uniform 侧的旧值为单集口径,不足以支撑,需重训才能扩展)

- Caveats 必须 disclose:单 seed(最大统计弱点);仅 matched eval(scale sweep 曲线 pending);
  主表仅最高档 P15/P50;obs 噪声只打 position-only 本体感觉通道 141 维、不碰 979 维 goal
  (与 RoboPianist 对齐的声明,非遗漏);σ 的物理单位跨通道不同,不应跨通道比较同一数值 level;
  仅 PPO(robust 任务无 cost)。

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
trains with off-the-shelf MARL (RLlib independent PPO / IPPO-style). `[verified]`
(historical 4-hand Duet Territorial IPPO-style 5M artifact reports eval reward
768; the reproducible F1/team-return baseline infrastructure is tracked separately)

- `multi_agent_design.md` §5(shared-zone-width axis,PAPER-CRITICAL)、§6(安全约束映射)。

## C7 — Scope 卫语句(防 reviewer)

- Adversarial perturbations 与 dynamics randomization **有意 out of scope**,理由已成文
  (robust_task_design §10 预期 claim 2、§11):对抗需 per-method 基建;dynamics 随机化
  动机是 sim-to-real(纯仿真基准不适用)且纠缠 F1 语义。
- Reduced action space 不启用:全 DoF Shadow Hand 是 benchmark 定义的一部分
  (thumb-budget 安全任务依赖 THJ1/THJ5)。

## Abstract 取舍(已定,2026-07-20,四轮)

- Venue = **AAAI 2027**;**v3 = 用户定稿(唯一现役版本)**,全文见 abstract.md。
- Findings:v3 结尾纳入两条**定性**finding(不带数字)——"coordination failures
  under shared multi-hand rewards"(=C3)与 "sign-dependent responses to action
  perturbations"(=C4d;注意措辞必须是 sign-dependent,**不是** non-monotonic——
  实测三点单调,§10 claim 5)。其余 C4x 留正文。
- v0-B(metric-led)已删:叙事与主线不符。C2 只作正文 metric 小节的性质论证
  (污染矩阵、RG 对照),不作论文 hook;图表布局:**正文 EpReward 学习曲线、
  F1 图表进 supplementary**。
- ⚠️ 连带约束:R-通道任务正文 EpReward 图必须画 `ep_return_true`(实收 `ep_return`
  被噪声污染,§14.2),或图注显式 disclose——否则 C2 的故事自打脸。
- C5 empirical 未出数前,abstract 安全腿只用能力措辞。
