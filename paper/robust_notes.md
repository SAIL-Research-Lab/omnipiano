# Robust leg — paper working notes(给合作者的自足文档)

**用途**:写 robust section 需要的全部信息都在这里,**不需要读** `omnipiano/docs/robust_task_design.md`(那是 187KB 的开发设计文档,含大量与论文无关的实施细节)。

**约定**(沿用 `paper/README.md`):可直接搬进论文的措辞用英文;导航/说明用中文。数字**只从本文件或 `claims.md` 引用**,不要自行到设计文档里捞。

**状态**:§5 结果表**正在重新测量中**(见该节说明),其余各节可以立即用于写作。

---

## 1. One-paragraph framing(可直接改写进论文)

OmniPiano-Robust turns a fixed-repertoire bimanual piano-playing benchmark into a
controlled testbed for perturbation robustness. Three perturbation channels —
**action**, **observation**, and **reward** — can be perturbed independently or in
combination, each by one of three noise families (Gaussian, Uniform, constant
Shift) at a configurable magnitude, and each channel may use a *different* family
within the same task. Crucially, the benchmark's headline metric (note-level F1
against the MIDI score) is computed from **physical ground truth** inside the
simulator and is therefore *never* contaminated by the injected noise: it changes
only because the perturbed policy genuinely plays differently. This separates
"the agent performs worse" from "we can no longer measure the agent", a confound
that reward-based robustness benchmarks cannot avoid.

---

## 2. 自由度设计(the design space —— 论文里最该讲清楚的部分)

五个正交的自由度:

| # | 自由度 | 取值 | 说明 |
|---|---|---|---|
| 1 | **扰动通道** | action / observation / reward | 三个通道对应智能体与环境交互的三个接口 |
| 2 | **通道组合** | **7 种非空组合** | 3 单通道 + 3 双通道(AO/AR/OR) + 1 三通道(AOR) |
| 3 | **噪声类型** | Gaussian(σ) / Uniform[low,high] / Shift(常数偏置) | 用各分布的**自然参数**,不做跨分布 std 归一 |
| 4 | **强度 level** | 每通道**独立** | action/obs: {0.05, 0.10, 0.15};reward: {0.10, 0.30, 0.50}。组合任务里每个通道可用**不同** level |
| 5 | **混分布** | 每通道**独立**选噪声类型 | 如 action=Shift + obs=Gaussian 同存于一个任务(2026-07-21 实施) |

外加一个**正交于任务定义**的测量轴:

| | **eval 噪声缩放** `eval_noise_scale` | 评测时把所有幅度乘以一个系数:1.0=matched(默认)、0.0=干净、>1=压力测试 |
|---|---|---|

> **设计决策(值得在论文里点一句)**:`eval_noise_scale` **不属于任务身份**,它是调用参数而非注册项——同一个训练好的策略可以在任意多个 scale 下被测量,不需要为每个 scale 注册新任务。相对地,通道/分布/强度**都烘焙进 env id**,使 env id 成为实验配置的唯一句柄。

**组合空间**:7 组合 × 3 分布 = **21 个任务/曲目**(不计 level 自由度);放开混分布后,每通道独立选 {关闭, Gauss, Uniform, Shift} → **63 种**。当前注册:27 个单通道(3 通道 × 3 分布 × 3 level)+ 3 个双通道 + 1 个 Clean + 2 个消融(A-Shift-N15 方向对照、Clean-NoOAR)。

**命名规范**:`OmniPiano-<Piece>-<Channels>-<Dist>-P<level×100>-v0`,例如
`OmniPiano-ClairDeLune-A-Gauss-P15-v0`(action 通道,高斯 σ=0.15)、
`OmniPiano-ClairDeLune-AR-Uniform-A15-R50-v0`(action ±0.15 + reward ±0.50,同为均匀分布)。

---

## 3. 与 Robust-Gymnasium 的差异(**代码核实**,不是从论文措辞推断)

核实对象:`github.com/SafeRL-Lab/Robust-Gymnasium` @ `af6de64`(2026-03-19,最新)。**这是 robust section 最重要的差异化论证**,建议在 related work 或 benchmark comparison 表里用。

| 能力 | Robust-Gymnasium | OmniPiano | RG 侧代码证据 |
|---|---|---|---|
| 多通道**同时**扰动 | ❌ 不支持 | ✅ 7 种组合 | `noise_factor` 是**单个字符串**,三个通道块 `if args.noise_factor == "action"/"state"/"reward"` 全是对同一变量的 exact `==`,互斥;全库无 substring/split 匹配 |
| 每通道**不同分布** | ❌ 不支持 | ✅ | `noise_type` 同为单个全局字符串;无任何 `{channel}_noise_type` 参数 |
| 每通道**不同强度** | ❌(单通道无从谈起) | ✅ | 同上 |
| 扰动**烘焙进注册任务** | ❌ | ✅ | RG 189 个 `register()` 调用**全部不含噪声配置**;噪声完全由调用方每步传参 |
| 噪声**按维度独立**采样 | ❌ 标量广播 | ✅ per-dim iid | RG 每步只抽**一个标量**加到整个 action/obs 向量 |
| 噪声可由 env seed **复现** | ❌ | ✅ 专用 RNG 流 | RG 用 Python 全局 `random` 模块,不经 env 的 `np_random`;`reset(seed=)` 无法复现噪声序列 |
| 噪声频率一致 | ⚠️ 版本间不一致 | ✅ 每步 | RG v4 族被 `llm_disturb_interval`(默认 500)门控 = **每 500 步才注一次**;v5 族每步注入。官方示例用的 `Ant-v4` 属前者 |
| 干净的评测指标 | ❌ 只有 return | ✅ F1(读物理真值) | — |

**RG 唯一的"双通道"路径**是 LLM 对抗(`llm_guide` 独立开关),但不可用于基准:它把 `str` 当作 observation 返回(破坏观测空间)、需要硬编码 OpenAI key、只存在于 11 个 v4 文件。

**措辞建议**:不要写成"RG 做不到"式的贬低,而是陈述能力边界——例如
"Robust-Gymnasium exposes one perturbation channel and one noise family per run,
selected by two global strings; OmniPiano registers the channel/family/magnitude
combination as part of the task identity, which additionally permits simultaneous
multi-channel and mixed-family perturbations."

---

## 4. 威胁模型(**必须在论文中明确声明**)

OmniPiano 沿用 RoboPianist 的观测增强:obs 里包含**上一步的 action(45 维物理量)和上一步的 reward(1 维)**(所谓 OAR)。这使"扰动是否对策略可观测"成为一个必须交代的建模选择:

- **Action 通道**:`obs["action"]` 放**加噪后实际执行的动作**。对应**硬件磨损模型**——关节老化/磨损使真实执行的动作就是带噪动作,智能体的本体感觉理应反映实际执行值。
- **Reward 通道**:`obs["reward"]` 放**加噪后观测到的 reward**。对应**传感器磨损模型**——观测到的、学习用的、记忆里的是同一个被污染的值。

> **为什么这必须写进论文**:我们**同时拥有另一种威胁模型下的完整测量**(此前的实现把 `obs["action"]` 置为未加噪的指令动作,即"扰动对策略隐藏")。同一个 +0.15 的 action shift,在"隐藏扰动"下是 **+12% 的性能增益**,在"可观测扰动"下是**零效应**。这是一个 RG 结构上无法产生的对照(RG 没有 OAR、没有此类语义选择),也是一条独立的 finding(见 §5 待补的 C4f)。

---

## 5. 实验结果(⏳ **正在重新测量,勿引用旧数字**)

**协议(已定稿)**:PPO(纯 PPO,robust 任务无 cost 故不跑 PPOLag)、seed=0、5M env steps、
**最终策略 + 10 集 deterministic matched eval**(`eval_noise_scale=1.0`)。

**为什么在重测**:此前表中数字混用了两种口径(部分为"训练末段 10 次周期性评测的均值",部分为"训练结束后单集评测"),且单集评测对随机噪声任务是高方差点估计。现统一为"最终策略 + 10 集",给出 mean±std。

**预期的一条结构性性质**(重测将验证):**Shift 类任务的集间方差应为 0**——常数偏置不消耗随机数,且环境初始姿态确定、MIDI 固定、评测策略确定,故 10 集结果应完全相同;Gaussian/Uniform 任务则有真实的集间方差。

结果表出来后填入这里,并同步更新 `claims.md` §C4。**在此之前请勿从 `robust_task_design.md` §10 直接引用数字**(该表正在同步修订)。

结论的**结构**(具体数值待定,但方向已被多次测量支持):
1. 分布难度排序 **Shift(易)> Uniform > Gaussian(难)**,在 action 与 reward 通道上一致;
2. 通道敏感度 **Action ≈ Reward ≫ Observation**;
3. 噪声导致**保守化**:precision 保持高位而 recall 崩塌(漏音多于误按);
4. **Reward 噪声在 return 看似健康时已摧毁学习**——需要 noise-immune 指标才能看见(与 §6 的方法论卖点直接呼应);
5. **Action-Shift 的效果由符号决定**且强烈不对称(负向伤害显著、正向接近持平);
6. **威胁模型对照**(隐藏 vs 可观测扰动)——见 §4。

---

## 6. 评测协议与指标(方法论部分)

- **Headline metric = F1**(note-level,对照 MIDI 谱面)。它由模拟器内的**物理真值**算出(承袭上游 RoboPianist 的 `MidiEvaluationWrapper` 定义,未改动),因此**不被任何噪声通道污染**:三个通道的噪声都不会改变"哪个琴键真的被按下"的测量。数值变化完全来自策略行为的真实改变。
- **Reward 指标 = 实际收到的(带噪)`ep_return`**。**不使用**去噪后的干净 return,即使我们能算出它——理由:现实中干净 reward 不可观测,智能体也只能看到带噪值,与现实应用保持一致(RG 亦如此)。因此 reward-shift 任务的 return 会系统性偏高,这是**扰动效应本身**,在图注中说明即可,不做修正。
- **Matched eval 为默认**(`eval_noise_scale=1.0`):评测时的噪声与训练时同级。这是与 RG 可比的鲁棒性数字(RG 在训练级扰动下评测,没有"干净评测"的概念)。
- **鲁棒性曲线**:同一策略在 scale ∈ {0, 0.5, 1, 2, 4} 下重复评测,得到退化曲线(基础设施已就绪,曲线数据待跑)。

---

## 7. 必须 disclose 的 caveats

1. **单 seed**(seed=0)——多 seed 待补,是当前最大的统计弱点;
2. **仅 matched eval**——scale sweep 曲线尚未跑;
3. **主表仅最高强度档**(action/obs 的 0.15、reward 的 0.50),中间档待补;
4. **obs 噪声只作用于策略实际消费的本体感觉通道**:关节角 `joints_pos`、感知键态 `piano/state`、`sustain_state`,共 141 维;**不扰动 979 维的 goal(谱面前瞻)**。关节速度不在其中,因为**上游 RoboPianist 本就是 position-only 观测**(`joints_vel` 在上游定义但从未启用),策略从不观测速度、故无速度可扰——这是与上游对齐的**声明**,不是遗漏;
5. **σ 的物理单位跨通道不同**(action 是 canonical [-1,1] 的位置目标,obs 是各观测量自身的单位,reward 是 reward 单位),因此**不应跨通道比较同一数值 level 的"强度"**;
6. **仅 PPO**——robust 任务无安全约束,故不跑 PPOLag。

---

## 8. Scope guards(防 reviewer 追问"为什么不做 X")

- **不做对抗扰动**:RG 唯一的对抗实现是 LLM-prompted,需要每步 API 调用(5M 步训练约 700 小时);非 LLM 的对抗方法需要按论文各自搭基础设施(梯度访问 / 单独训练的 adversary),无法在基线间共享。
- **不做 dynamics randomization**:(a) 其经典动机是 sim-to-real 迁移,而本基准是固定曲目的纯仿真任务,动机不适用;(b) 扰动物理参数(如琴键弹簧刚度)会把鲁棒性轴与**"正确演奏"的定义**纠缠在一起——F1 依赖"力→按键激活"的映射,改了它就等于改了评分标准;而 obs/action/reward 噪声不触碰这一映射,ground-truth F1 测量保持完好。

---

## 9. 复现指引

```python
import omnipiano
# 训练:注册 id 即完整实验配置
env = omnipiano.make("OmniPiano-ClairDeLune-A-Gauss-P15-v0")
# 评测:matched(默认)
env = omnipiano.make("OmniPiano-ClairDeLune-A-Gauss-P15-v0", mode="eval", log_dir=d)
# 评测:压力测试(2 倍),无需注册新任务
env = omnipiano.make("OmniPiano-ClairDeLune-A-Gauss-P15-v0", mode="eval",
                     eval_noise_scale=2.0, log_dir=d)
```

- 训练脚本:`examples/run_sb3_baseline.py --algo ppo --env <id> --seed 0`
- 鲁棒性曲线 / 多集评测:`examples/robust_eval_sweep.py --ckpt <final_model.zip> --env <id> --scales 1.0 --num-eval-eps 10`
- 每集指标 CSV 由环境自动写出(F1/precision/recall、cost、reward 分解、注入噪声量、eval scale)。

---

## 10. 与其他 claims 的接口

- **C2(noise-immune metric)**:robust 腿是这条方法论卖点的**主要证据来源**——尤其"reward 噪声让 return 看起来健康、实则学习已被摧毁"这个案例。
- **C1(unified benchmark)**:robust 是四条腿之一,与 safety / morphology / multi-agent 共享同一套任务基座和指标定义。
- 具体 claim 措辞见 `claims.md` §C4(数字更新后同步)。
