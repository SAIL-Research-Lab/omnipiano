# Abstract — 当前版本与决策日志

## 当前版本(v3,用户定稿 2026-07-20,含三处事实性小修)

Safe, robust, and multi-agent reinforcement learning have largely evolved around separate benchmark ecosystems, leaving their behavior and trade-offs under a shared high-dimensional embodied control problem underexplored. We present OmniPiano, an algorithm- and framework-agnostic Gymnasium benchmark built on simulated dexterous piano playing with Shadow Hands. Piano playing provides a contact-rich continuous-control problem that requires precise spatial and temporal coordination, while allowing task complexity to scale systematically through both embodiment and musical content. OmniPiano extends RoboPianist into a unified task family organized along four benchmark axes. First, a morphology ladder scales from two to five hands, covering 45 to 111 action dimensions, with optional physically enforced keyboard partitions. Second, safe RL variants expose four per-step cost families based on joint magnitude, inter-hand collision force, actuator power, and joint-injury risk. Third, robust RL variants introduce observation, action, and reward perturbations under three noise distributions following Robust-Gymnasium protocols. Fourth, cooperative multi-agent tasks partition the hands among decentralized agents and vary the amount of shared keyboard territory. Evaluations across algorithms and task variants reveal substantial variation in task performance, constraint satisfaction, scalability, and robustness. OmniPiano provides a common testbed for studying capability, safety, robustness, and coordination in high-dimensional dexterous control.

### v3 相对用户定稿原文的三处事实性修正(2026-07-20 代码/文档核对)

1. "45 to **more than 111**" → "45 to **111**"。5-hand 动作维度 = 22×5+1 = **恰好 111**
   (README morphology ladder;bimanual = 45 同源),无超过 111 的已注册形态。
2. "assign **individual hands** to decentralized agents" → "**partition the hands
   among** decentralized agents"。canonical 分解里 agent 拥有**手对**(4-hand duet:
   secondo/primo 各一对 LH+RH;仅 3/5-hand 各有一个单手 soloist)——
   `multi_agent_design.md` §3.2–3.4。

### 口径记录(非修正)

- **cost 归组改为 4 族**(joint magnitude / inter-hand collision force / actuator
  power / joint-injury risk),取代 README 的 6 族口径——7 个约束类按物理量重新
  分组,映射表见 claims.md C1。**正文与 README 发布前需统一到 4 族口径。**

## 决策日志

| 日期 | 决策 | 状态 |
|---|---|---|
| 2026-07-20 | 论文 scope = 整个 OmniPiano benchmark(morphology ladder + safety + robust + multi-agent 四条腿) | ✅ 用户拍板 |
| 2026-07-20 | 问题定义 = RL 学术界缺少"精细化操作/控制 × {safety, robustness, MA} 评测轴"合一的 benchmark(claims.md C0) | ✅ 用户拍板 |
| 2026-07-20 | 目标 venue = **AAAI 2027**(main track;按往年惯例 abstract deadline ≈ 7 月底、全文 ≈ 8 月初——**时间紧**) | ✅ 用户拍板 |
| 2026-07-20 | 图表布局:**正文放 EpReward 学习曲线,F1 图表放 supplementary**;R-通道正文图必须用 `ep_return_true`(§14.2) | ✅ 用户拍板 |
| 2026-07-20 | ❌ 撤回 "millisecond-precise"(控制步 50 ms/20 Hz,`base.py:31`;F1 按控制步采样,`evaluation.py:67-71`)。全文禁用毫秒级表述 | ✅ 已核实 |
| 2026-07-20 | 删除 v0-B(metric-led):叙事与主线不符。F1 noise-immunity 只作正文 metric 小节性质,不作论文 hook | ✅ 已删 |
| 2026-07-20 | 第三轮:abstract 零 finding → **第四轮部分推翻**:用户定稿(v3)重新纳入两条**定性**finding(不带数字) | ✅ 用户拍板 |
| 2026-07-20 | **v3 = 唯一现役版本**;v0-A/v1/v2 草稿全部删除(历史见 git:f96c6a2→1cd3871) | ✅ 用户拍板 |
| 2026-07-20 | cost 族口径:abstract 用 **4 族**分组(README 现为 6 族,发布前统一) | 待 README 同步 |

## 数字来源(不在本文件手编,引用 claims.md → 设计文档)

全部实验数字见 `claims.md` C3/C4/C6;robust 腿溯源 `paper/robust_notes.md` §5(2026-07-21 定稿口径:最终策略 + 10 集 matched eval),其余溯源 `static_partition_design.md`、README demos。
