# Abstract — 草稿与决策日志

## 决策日志

| 日期 | 决策 | 状态 |
|---|---|---|
| 2026-07-20 | 论文 scope = 整个 OmniPiano benchmark(morphology ladder + safety + robust + multi-agent 四条腿) | ✅ 用户拍板 |
| 2026-07-20 | 问题定义 = RL 学术界缺少"精细化操作/控制 × {safety, robustness, MA} 评测轴"合一的 benchmark(claims.md C0) | ✅ 用户拍板 |
| 2026-07-20 | 目标 venue = **AAAI 2027**(main track;~7 页正文;按往年惯例 abstract deadline ≈ 2026-07 月底、全文 ≈ 8 月初——**时间紧**) | ✅ 用户拍板 |
| 2026-07-20 | framing = **v0-A breadth-led**(缺口→任务→四条腿→F1 性质→findings) | ✅ 用户拍板 |
| 2026-07-20 | abstract findings = **C3 winner-takes-all + C4d directional bias**(C4e 不进 abstract) | ✅ 用户拍板 |
| 2026-07-20 | 图表布局:**正文放 EpReward 学习曲线,F1 图表放 supplementary**(training rollout 原生只有 EpReward,F1 来自 periodic eval,见 robust_task_design §14.3) | ✅ 用户拍板 |
| — | ⚠️ 上一条的连带约束:R-通道任务的正文 EpReward 图必须用 `ep_return_true`(或显式 disclose 实收带噪,§14.2);abstract 中 F1 从 "headline metric" 措辞降为 "every task additionally reports…"(v1 已改),避免与正文图表主次矛盾 | 写作纪律 |

## Draft v1(当前版:AAAI 2027,v0-A framing,findings=C3+C4d)~190 词

> Progress on safe, robust, and multi-agent reinforcement learning is measured on
> largely disjoint benchmarks whose underlying tasks — point navigation, low-DoF
> locomotion, abstracted particle worlds — are far simpler than the control problems
> motivating the field. We present **OmniPiano**, a unified benchmark built on
> simulated piano playing with Shadow Hands: a fine-grained, contact-rich control
> problem demanding millisecond-precise coordination of 45 to 111+ actuated degrees
> of freedom. Within one codebase and one framework-agnostic Gymnasium API, OmniPiano
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
> problem demanding millisecond-precise coordination of 45 to 111+ actuated degrees
> of freedom. Within one codebase and one framework-agnostic Gymnasium API, OmniPiano
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

## Draft v0-B(metric-led:先讲"尺子"问题,适合更学术的 main-track 语气)~200 词

> How should robustness be measured when the perturbation being studied can corrupt
> the measurement itself? Return-based evaluation — standard in robust RL — cannot
> distinguish a degraded policy from a polluted reward channel. We present
> **OmniPiano**, an RL benchmark built on simulated Shadow-Hand piano playing, where
> the headline metric — note-level F1 against the MIDI score, computed from
> ground-truth physics — is structurally immune to observation, action, and reward
> perturbations. On this foundation OmniPiano unifies four evaluation axes that
> existing benchmarks treat in isolation, on a single fine-grained control task
> family with 45 to 111+ action dimensions: a 2-to-5-hand morphology ladder,
> safe RL with six constraint families, Robust-Gymnasium-aligned perturbation
> protocols, and cooperative multi-agent play. Baselines demonstrate why the
> combination matters: reward noise destroys learning while received return stays
> within 10% of clean — invisible without a noise-immune metric; a constant action
> bias improves performance, showing systematic bias is qualitatively unlike
> zero-mean noise; and multi-hand reward without physical partition collapses to
> winner-takes-all idling. OmniPiano is algorithm- and framework-agnostic, with
> baselines spanning PPO, SAC, TQC, PPO-Lagrangian, and MAPPO. Code and tasks: <URL>.

## 两版差异(讨论要点)

| | v0-A breadth-led | v0-B metric-led |
|---|---|---|
| 第一句卖点 | benchmark 缺口(C0) | 测量方法论问题(C2) |
| 适合 venue | NeurIPS D&B / benchmark track | ICLR/ICML main track |
| 风险 | "又一个 benchmark" 的审稿疲劳 | metric 故事只强撑 robust 一条腿,另三条腿显得挂靠 |
| findings 位置 | 结尾三连 | 中后段,与 metric 论证交织 |

## 数字来源(不在本文件手编,引用 claims.md → 设计文档)

全部实验数字见 `claims.md` C3/C4/C6,溯源 `robust_task_design.md` §10、README demos。
