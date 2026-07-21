# Abstract — 草稿与决策日志

## 决策日志

| 日期 | 决策 | 状态 |
|---|---|---|
| 2026-07-20 | 论文 scope = 整个 OmniPiano benchmark(morphology ladder + safety + robust + multi-agent 四条腿) | ✅ 用户拍板 |
| 2026-07-20 | 问题定义 = RL 学术界缺少"精细化操作/控制 × {safety, robustness, MA} 评测轴"合一的 benchmark(claims.md C0) | ✅ 用户拍板 |
| 2026-07-20 | 目标 venue = **AAAI 2027**(main track;~7 页正文;按往年惯例 abstract deadline ≈ 2026-07 月底、全文 ≈ 8 月初——**时间紧**) | ✅ 用户拍板 |
| 2026-07-20 | framing = **v0-A breadth-led**(缺口→任务→四条腿→F1 性质→findings) | ✅ 用户拍板 |
| 2026-07-20 | ~~abstract findings = C3 + C4d~~ → **推翻(用户第三轮):abstract 暂不写任何具体 empirical finding**(winner-takes-all、action-bias-improves 等都会让 reviewer 困惑/节外生枝);findings 留给正文。abstract 结尾只陈述 baseline 覆盖面 | ✅ 用户拍板 2026-07-20 |
| 2026-07-20 | **删除 Draft v0-B(metric-led)**:叙事逻辑与 work 主线不符(用户拍板)。F1 noise-immunity 只作为正文 metric 小节的性质,不作为论文 hook | ✅ 已删 |
| 2026-07-20 | 图表布局:**正文放 EpReward 学习曲线,F1 图表放 supplementary**(training rollout 原生只有 EpReward,F1 来自 periodic eval,见 robust_task_design §14.3) | ✅ 用户拍板 |
| 2026-07-20 | 新增 **Draft v2 = 用户 task-first framing 修订版**(与 v1 gap-first 并列备选);修 Level-1/-3 术语、duet 单手误述、per-key obs 误述、原稿结尾与 C4d 矛盾的 "systematic degradation" 措辞 | ✅ 当前主推 v2,v1 备选 |
| 2026-07-20 | ❌ 撤回 "millisecond-precise" 措辞(代码核实:控制步 50 ms/20 Hz,`base.py:31` + `configs/__init__.py:334`;F1 按控制步采键激活,`evaluation.py:67-71`;5 ms 只是物理子步)。改为 "note-accurate timing … at a 20 Hz control rate"。全文写作禁用毫秒级表述 | ✅ 已核实并修正 |
| — | ⚠️ 上一条的连带约束:R-通道任务的正文 EpReward 图必须用 `ep_return_true`(或显式 disclose 实收带噪,§14.2);abstract 中 F1 从 "headline metric" 措辞降为 "every task additionally reports…"(v1 已改),避免与正文图表主次矛盾 | 写作纪律 |

## Draft v2(用户 task-first framing 修订版,2026-07-20)~200 词

> Robotic piano playing has emerged as a uniquely measurable testbed for dexterous
> control: high-dimensional continuous action, contact-rich dynamics, and unambiguous
> MIDI-grounded evaluation. Yet existing piano benchmarks stop at two human-sized
> hands, impose no safety constraints, and assume noise-free perception — gaps that
> keep them out of reach for safe, robust, and multi-agent RL research. We introduce
> **OmniPiano**, an algorithm- and framework-agnostic Gymnasium benchmark that extends
> RoboPianist along three axes. (i) **Morphology**: a ladder of 2–5 hands (45 to 111+
> actuated degrees of freedom), each playable either unconstrained or physically
> partitioned, with every hand hard-clamped to its own non-overlapping keyboard
> register. (ii) **Safety**: six cost-constraint families — from hand–hand collision
> forces to per-joint injury power budgets — exposed through a standard per-step
> cost signal. (iii) **Robustness**: action, observation,
> and reward perturbations under three noise distributions, protocol-aligned with
> Robust-Gymnasium. OmniPiano further provides cooperative duet tasks in which each
> agent commands a pair of hands within its own keyboard territory, supporting
> centralized-training-decentralized-execution methods such as MAPPO. We provide
> reference baselines spanning standard, safe, and multi-agent RL (PPO, SAC, TQC,
> PPO-Lagrangian, MAPPO), together with a unified evaluation protocol and
> ground-truth, MIDI-grounded metrics across all task families. Code and tasks: <URL>.

### v2 相对用户原稿的修改清单
1. **Level-1/Level-3 术语移除**(用户指出读者必困惑):
   "unconstrained Level-3 prototypes and hard-clamped Level-1 StaticPartition tasks"
   → "playable either unconstrained or physically partitioned, with every hand
   hard-clamped to its own non-overlapping keyboard register"。Level 编号是内部
   分类法,留给正文定义后再用。
2. **首句去重**:"high-dimensional … high-dimensional" 出现两次 → 第二处并入冒号列举
   (action 维度 / contact-rich / MIDI 评测三要素)。
3. **事实修正:duet 每 agent 控一对手**(secondo=bass 侧 LH+RH,primo=treble 侧
   LH+RH;README demo + multi_agent_design §3),原稿 "controls one hand" 不实。
4. **事实修正:"per-key observation" 删掉**——obs 噪声打的是白名单 141 维
   (joints_pos + piano/state + sustain_state,robust_task_design §10 caveat d),
   且 per-key σ 校准明确不做(§11)。
5. **safety 列举改为显式采样措辞** "six … families — from hand–hand collision forces
   to per-joint injury power budgets"。原稿及 v2 初版都是"说 6 列 4"的半列举,计数
   与列表打架;全 6 族(7 个约束类)的映射表在 claims.md C1,正文用。
6. **结尾不写任何具体 finding**(用户第三轮拍板):原稿的 "systematic robustness
   degradation" 与 C4d 矛盾、"constraint satisfaction" 实证 [pending];而 C3/C4d
   本身又易引 reviewer 困惑。最终结尾只写 baseline 覆盖(5 算法)+ 统一评测协议 +
   MIDI-grounded 指标,全部 [design]/[verified] 级、零争议。findings 留正文。
7. "calibrated" → "protocol-aligned with Robust-Gymnasium"(档位对齐 RG paper 图,
   claims.md C1 措辞)。
8. 补 45–111+ DoF 数字(morphology 轴的规模感)。20 Hz 未进本版(task-first 首句
   已够长;若要可加回)。

## Draft v1(gap-first framing)【存档;结尾 findings 句已被第三轮决策否决,如启用需同步删改】

> Progress on safe, robust, and multi-agent reinforcement learning is measured on
> largely disjoint benchmarks whose underlying tasks — point navigation, low-DoF
> locomotion, abstracted particle worlds — are far simpler than the control problems
> motivating the field. We present **OmniPiano**, a unified benchmark built on
> simulated piano playing with Shadow Hands: a fine-grained, contact-rich control
> problem demanding note-accurate timing from 45 to 111+ actuated degrees
> of freedom at a 20 Hz control rate. Within one codebase and one framework-agnostic Gymnasium API, OmniPiano
> extends RoboPianist along four axes: a **morphology ladder** from two to five hands
> with optional physically enforced register partitions; **safe-RL** tasks with six
> constraint families, from collision forces to joint-injury budgets; **robust-RL**
> tasks injecting observation, action, and reward perturbations under three
> distributions, protocol-aligned with Robust-Gymnasium; and **cooperative
> multi-agent** variants with a tunable shared-territory axis. Every task additionally
> reports a ground-truth, physics-derived note-level F1 that no noise channel can
> contaminate. Baselines across five algorithms (PPO, SAC, TQC, PPO-Lagrangian,
> MAPPO) surface findings simpler suites cannot: unpartitioned multi-hand reward
> collapses to winner-takes-all idling, while a constant action bias can *improve*
> performance — challenging the perturbation-equals-degradation framing. Code and
> tasks: <URL>.

### v1 相对 v0-A 的改动(均由决策日志驱动)
1. F1 句从 "A key design property: the headline metric …" 降格为
   "Every task additionally reports …" —— 配合正文 EpReward 为主、F1 进 supplementary。
2. findings 三连删掉 C4e(reward noise cripples learning …),保留 C3 + C4d。
3. 压到 ~190 词,符合 AAAI 摘要惯例(150–200)。

## Draft v0-A(breadth-led:先讲统一 benchmark,再讲 metric 性质)~215 词【存档】

> Progress on safe, robust, and multi-agent reinforcement learning is measured on
> largely disjoint benchmarks whose underlying tasks — point navigation, low-DoF
> locomotion, abstracted particle worlds — are far simpler than the control problems
> motivating the field. We present **OmniPiano**, a unified benchmark built on
> simulated piano playing with Shadow Hands: a fine-grained, contact-rich control
> problem demanding note-accurate timing from 45 to 111+ actuated degrees
> of freedom at a 20 Hz control rate. Within one codebase and one framework-agnostic Gymnasium API, OmniPiano
> extends RoboPianist along four axes: a **morphology ladder** from two to five hands
> with optional physically enforced register partitions; **safe-RL** tasks with six
> constraint families, from collision forces to joint-injury budgets; **robust-RL**
> tasks injecting observation, action, and reward perturbations under three
> distributions, protocol-aligned with Robust-Gymnasium; and **cooperative
> multi-agent** variants with a tunable shared-territory axis. A key design property:
> the headline metric — note-level F1 computed from ground-truth physics — is
> structurally immune to all three noise channels, so robustness is measured without
> metric contamination. Baselines across five algorithms surface findings invisible
> to simpler suites: unpartitioned multi-hand reward collapses to winner-takes-all
> idling; a constant action bias can *help*, challenging the perturbation-equals-
> degradation framing; and reward noise cripples learning while received returns
> look healthy. Code and tasks: <URL>.

> (v0-B metric-led 草稿已删除,2026-07-20:叙事逻辑与 work 主线不符。历史版本见 git。)

## 数字来源(不在本文件手编,引用 claims.md → 设计文档)

全部实验数字见 `claims.md` C3/C4/C6,溯源 `robust_task_design.md` §10、README demos。
