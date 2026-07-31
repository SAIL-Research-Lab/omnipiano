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

## 5. 实验结果(✅ 已定稿,2026-07-21)

**协议**:PPO(纯 PPO,robust 任务无 cost 故不跑 PPOLag)、seed=0、5M env steps、
**最终策略 + 10 集 deterministic matched eval**(`eval_noise_scale=1.0`)。曲目 = Clair de Lune(588 步/集)。

| task | F1 (mean±std, 10 ep) | ΔF1 vs clean | precision | recall | return\* |
|---|---|---|---|---|---|
| **Clean-v0**(参照) | 0.626 ± 0.000 | — | 0.974 | 0.506 | 1851 |
| **A-Gauss-P15** | 0.267 ± 0.035 | −0.359 (−57%) | 0.958 | 0.182 | 1671 |
| **A-Uniform-P15** | 0.491 ± 0.051 | −0.135 (−22%) | 0.970 | 0.372 | 1802 |
| **A-Shift-P15** | 0.615 ± 0.000 | −0.011 (−2%) | 0.991 | 0.492 | 1859 |
| **A-Shift-N15**(诊断) | 0.545 ± 0.000 | −0.081 (−13%) | 0.945 | 0.419 | 1698 |
| **O-Gauss-P15** | 0.562 ± 0.015 | −0.064 (−10%) | 0.980 | 0.428 | 1832 |
| **O-Uniform-P15** | 0.616 ± 0.032 | −0.010 (−2%) | 0.985 | 0.497 | 1849 |
| **O-Shift-P15** | 0.539 ± 0.000 | −0.087 (−14%) | 0.980 | 0.433 | 1832 |
| **R-Gauss-P50** | 0.109 ± 0.004 | −0.517 (−83%) | 0.992 | 0.072 | 1692 |
| **R-Uniform-P50** | 0.327 ± 0.029 | −0.300 (−48%) | 0.991 | 0.236 | 1774 |
| **R-Shift-P50** | 0.589 ± 0.000 | −0.038 (−6%) | 0.971 | 0.475 | **2141** |
| **AO-Gauss-P15**(A+O) | 0.177 ± 0.024 | −0.449 (−72%) | 0.923 | 0.121 | 1624 |
| **AR-Uniform-A15-R50**(A+R) | 0.232 ± 0.031 | −0.394 (−63%) | 0.979 | 0.157 | 1717 |
| **OR-Shift-ON15-RN50**(O+R) | 0.672 ± 0.000 | **+0.046 (+7%)** | 0.991 | 0.544 | 1592 |
| **Clean-NoOAR**(消融) | 0.520 ± 0.000 | −0.106 (−17%) | 0.964 | 0.401 | 1745 |

\* return = **实际收到的(带噪)** episode return。reward 通道任务的该列被噪声污染——这正是 finding 4 的素材:`R-Shift` 的 2141 里约 +294 是噪声灌入(0.5 × 588 步)。

### 一条可写进方法节的结构性性质(实测验证)

**Shift 类任务的评测集间方差恰好为零**(表中 5 个 shift 任务 std = 0.000,10 集给出**同一个** F1 值),因为常数偏置不消耗随机数,且环境初始姿态确定、MIDI 固定、评测策略确定;Gaussian/Uniform 任务则有真实方差(std 0.004–0.051)。**含义**:shift 任务单集评测即充分,随机噪声任务必须多集平均——这解释了为什么本表统一采用 10 集协议。

### 六条 finding(措辞见 `claims.md` §C4)

1. **分布难度排序**:在 **action 与 reward 通道**上一致为 Shift(易)> Uniform > Gaussian(难)。
   A: 0.615 / 0.491 / 0.267;R: 0.589 / 0.327 / 0.109。
   ⚠️ **observation 通道是例外**:Uniform 0.616 > Gaussian 0.562 > **Shift 0.539(最差)**。恒定的观测偏置比观测白噪声更伤——论文里应如实指出这个反直觉的例外,不要笼统说"三通道一致"。
2. **通道敏感度**:Action ≈ Reward ≫ Observation。最高档高斯噪声下:A −57%、R −83%、O 仅 −10%。
3. **噪声导致保守化**:所有任务的 precision 保持 0.92–0.99,而 recall 崩塌——被扰动的策略是**漏音**而非**按错**。F1 的下跌几乎全部来自 recall。
4. **Reward 噪声在 return 看似健康时已摧毁学习**:R-Gauss-P50 的 F1 只剩 0.109(−83%),而实收 return 1692 仅比 clean 低 9%;R-Shift-P50 的 return 反而被抬高到 2141(+16%)而 F1 −6%。**没有 noise-immune 指标就看不见这类失败**——这是 C2 方法论卖点最强的案例。
5. **Action-Shift 的效果由符号决定,且强烈不对称**:+0.15 → −2%(近乎无损),−0.15 → −13%(伤害是前者的 7 倍)。
   **机理假说(待验证)**:加噪后动作被 `clip` 到 [−1,1],故 −0.15 把可达范围压成 [−1, 0.85],**损失"完全按下"这一端**——而弹琴恰恰需要它(能力性损失,策略无法补偿);+0.15 只损失"完全抬起"端,可被补偿。验证方法:统计动作触顶/触底的裁剪率与按键力度分布。
6. **组合通道:随机噪声超可加退化,恒定偏置不然**。AO-Gauss(0.177)低于其两个分量(A 0.267 / O 0.562);AR-Uniform(0.232)同样低于分量(A 0.491 / R 0.327)。而 OR-Shift(0.672)**高于 clean 7%**。
   ⚠️ **OR-Shift 这条须谨慎**:(a) 单 seed;(b) **缺少单通道对照**——OR 用的是 obs **−0.15**,而已注册的 O-Shift-P15 是 **+0.15**,方向相反,无法分解。要下结论需补一个 `O-Shift-N15` 单通道 run。

### 威胁模型对照(finding 6 的延伸,数据部分可用)

同一批任务在"扰动对策略隐藏"(旧实现)与"扰动可观测"(现实现)两种威胁模型下各训练过一次:

| 任务 | 隐藏扰动 | 可观测扰动 | Δ | 证据强度 |
|---|---|---|---|---|
| A-Shift-P15 | 0.704 | 0.615 | **−0.089** | **强**(两侧均确定性,协议无关) |
| A-Shift-N15 | 0.553 | 0.545 | −0.008 | **强**(同上) |
| A-Gauss-P15 | 0.277 | 0.267 ± 0.035 | ≈0 | 中(旧值为单集) |
| A-Uniform-P15 | 0.371 | 0.491 ± 0.051 | +0.120? | **弱**(旧值为单集,可能是不利抽样) |

**可直接使用的结论**(仅基于强证据两行):**恒定正向偏置的"意外增益"在扰动变得可观测后消失**(+0.15:0.704 → 0.615),**而负向偏置的伤害不受可观测性影响**(−0.15:0.553 → 0.545,几乎不变)。这与 finding 5 的裁剪假说**相互印证**:可补偿的一侧(+)在策略能看见扰动后被补偿掉;不可补偿的一侧(−)是能力性损失,看得见也没用。

> **写作提醒**:表中 gauss/uniform 两行的旧值来自单集评测,与新的 10 集均值口径不同,**不要把它们当作严格对照**;若要在论文中主张"可观测性对随机噪声的影响",需要用同一协议重测旧语义(旧 run 已删除,需重训)。

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
