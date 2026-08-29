# Robust leg — paper working notes(给合作者的自足文档)

**用途**:写 robust section 需要的全部信息都在这里,**不需要读** `omnipiano/docs/robust_task_design.md`(那是 187KB 的开发设计文档,含大量与论文无关的实施细节)。

**约定**(沿用 `paper/README.md`):可直接搬进论文的措辞用英文;导航/说明用中文。数字**只从本文件或 `claims.md` 引用**,不要自行到设计文档里捞。

**指标口径(2026-08-28 决定,本文件已按此重写)**:**主指标 = `ep_return`,F1 为次要指标但始终记录并呈现。**
全部结果表 return 在前、F1 在后。唯一的读数陷阱是 **reward-shift 任务的 `ep_return` 不能直接和 clean 比**(§5 表下说明)。

**状态**:§5 结果表已定稿(2026-07-21 测量,2026-08-28 换口径重排);其余各节可以立即用于写作。

---

## 1. One-paragraph framing(可直接改写进论文)

OmniPiano-Robust turns a fixed-repertoire bimanual piano-playing benchmark into a
controlled testbed for perturbation robustness. Three perturbation channels —
**action**, **observation**, and **reward** — can be perturbed independently or in
combination, each by one of three noise families (Gaussian, Uniform, constant
Shift) at a configurable magnitude, and each channel may use a *different* family
within the same task. The benchmark reports **episode return as its primary
metric**, as any RL benchmark does, and pairs it with a **secondary metric that
the perturbation cannot touch**: note-level F1 against the MIDI score, computed
from **physical ground truth** inside the simulator. F1 moves only because the
perturbed policy genuinely plays differently, never because the measurement
itself was corrupted. The pair is what makes the perturbation legible: under
reward-channel noise the two metrics **come apart**, and it is precisely that
gap — a return that still looks healthy over a policy that has stopped playing
the piece — which a return-only robustness benchmark reports as success.

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
| **不受噪声污染的第二指标** | ❌ 只有 return | ✅ return + F1(读物理真值) | — |

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

**主指标 return 在前,F1 在后**(2026-08-28 口径)。return = **实际收到的(带噪)**`ep_return`,不做去噪修正。

| task | **return** (mean±std, 10 ep) | **Δret vs clean** | F1 | ΔF1 | precision | recall |
|---|---|---|---|---|---|---|
| **Clean-v0**(参照) | **1851 ± 0** | — | 0.626 ± 0.000 | — | 0.974 | 0.506 |
| **A-Gauss-P15** | **1671 ± 10** | −10% | 0.267 ± 0.035 | −57% | 0.958 | 0.182 |
| **A-Uniform-P15** | **1802 ± 24** | −3% | 0.491 ± 0.051 | −22% | 0.970 | 0.372 |
| **A-Shift-P15** | **1859 ± 0** | +0% | 0.615 ± 0.000 | −2% | 0.991 | 0.492 |
| **A-Shift-N15**(诊断) | **1698 ± 0** | −8% | 0.545 ± 0.000 | −13% | 0.945 | 0.419 |
| **O-Gauss-P15** | **1832 ± 11** | −1% | 0.562 ± 0.015 | −10% | 0.980 | 0.428 |
| **O-Uniform-P15** | **1849 ± 17** | −0% | 0.616 ± 0.032 | −2% | 0.985 | 0.497 |
| **O-Shift-P15** | **1832 ± 0** | −1% | 0.539 ± 0.000 | −14% | 0.980 | 0.433 |
| **R-Gauss-P50** | **1692 ± 12** | −9% | 0.109 ± 0.004 | **−83%** | 0.992 | 0.072 |
| **R-Uniform-P50** | **1774 ± 8** | −4% | 0.327 ± 0.029 | −48% | 0.991 | 0.236 |
| **R-Shift-P50** ⚠️ | **2141 ± 0** | ~~+16%~~ | 0.589 ± 0.000 | −6% | 0.971 | 0.475 |
| **AO-Gauss-P15**(A+O) | **1624 ± 13** | −12% | 0.177 ± 0.024 | −72% | 0.923 | 0.121 |
| **AR-Uniform-A15-R50**(A+R) | **1717 ± 19** | −7% | 0.232 ± 0.031 | −63% | 0.979 | 0.157 |
| **OR-Shift-ON15-RN50**(O+R) ⚠️ | **1592 ± 0** | ~~−14%~~ | 0.672 ± 0.000 | **+7%** | 0.991 | 0.544 |
| **Clean-NoOAR**(消融) | **1745 ± 0** | −6% | 0.520 ± 0.000 | −17% | 0.964 | 0.401 |

> ### ⚠️ 唯一的读数陷阱:**带 reward-shift 的任务,Δret 无意义**
>
> reward 通道的常数偏置直接加在**被测量的那个量**上,每步 +c 累积成每集 +588c:
>
> | 任务 | reward shift | `ep_return` | `ep_return_true` | 差值 | 真实 Δret |
> |---|---|---|---|---|---|
> | **R-Shift-P50** | +0.50 | 2140.6 | 1846.6 | **+294.0** = 0.5×588 | **−0.2%**(不是 +16%) |
> | **R-Shift-N50** | −0.50 | 1480.7 | 1774.7 | **−294.0** | **−4.1%**(不是 −20%) |
> | **OR-Shift-ON15-RN50** | −0.50 | 1592.3 | 1886.3 † | −294.0 † | **+1.9%** †(不是 −14%) |
>
> † OR-Shift 的 `ep_return_true` 由同一常数推算(该 run 的 reward shift 同为 −0.50、
> 集长同为 588 步),未单独从 CSV 读出;上面两行是实测,差值精确等于 shift×588。
>
> **零均值的 reward 噪声不受影响**:`r_gauss_p50` 差值 +2.6、`r_uniform_p50` −1.1。
> action / obs 通道完全不受影响(噪声不碰奖励信号)。全表只有上面标 ⚠️ 的三格。

### 一条可写进方法节的结构性性质(实测验证)

**Shift 类任务的评测集间方差恰好为零**(表中 6 个 shift 任务 return 和 F1 的 std 双双 = 0.000,10 集给出**同一个**值),因为常数偏置不消耗随机数,且环境初始姿态确定、MIDI 固定、评测策略确定;Gaussian/Uniform 任务则有真实方差。**两个指标的方差量级差一个数量级**:

| | return CV | F1 CV |
|---|---|---|
| shift / clean | 0% | 0% |
| gaussian / uniform | **0.46 – 1.34%** | **2.6 – 13.5%** |

**含义**:shift 任务单集评测即充分;随机噪声任务必须多集平均,而且**按主指标 return 算所需集数远少于按 F1 算**(return 的相对波动小 5–10 倍)。本表统一采用 10 集,对两个指标都够用。

### 六条 finding(措辞见 `claims.md` §C4)

每条先给主指标 return,再给 F1。**两者何时一致、何时分歧,本身是 robust 腿最重要的一组结果**。

1. **分布难度排序:Shift(易)> Uniform > Gaussian(难)——在 action 与 reward 通道上,两个指标一致。**
   - return —— A: 1859 / 1802 / 1671;R(用 `ep_return_true`): 1847 / 1774 / 1692。
   - F1 —— A: 0.615 / 0.491 / 0.267;R: 0.589 / 0.327 / 0.109。
   - **机理**:shift 是可学习的恒定偏置,uniform 的经验 std = level/√3 < gaussian 的 level,白噪声最难。
   ⚠️ **observation 通道两个指标给出不同的答案**:
   - **F1 上 obs 是例外**:Uniform 0.616 > Gaussian 0.562 > **Shift 0.539(最差)**——恒定观测偏置比观测白噪声更伤。
   - **return 上这个例外看不见**:1849 / 1832 / 1832,极差仅 **17 分**(clean 1851 的 0.9%),且 Gaussian 与 Shift **完全相等**。
   → 写作建议:主表用 return,obs 通道的分布排序**必须引 F1 副表才讲得清楚**;不要笼统说"三通道一致"。

2. **通道敏感度:Action ≈ Reward ≫ Observation——排序两个指标一致,但幅度差一个数量级。**
   最高档高斯噪声下:

   | | Action | Reward | Observation |
   |---|---|---|---|
   | **Δreturn** | −10% | −9% | −1% |
   | **ΔF1** | **−57%** | **−83%** | −10% |

   排序相同,但 **return 把严重程度压缩了 6–9 倍**。只看主指标会把"策略基本不会弹了"读成"掉了不到一成"。这是 finding 4 的一般化:**低估不限于 reward 通道,三个通道都存在**,只是 reward 通道最极端。

3. **噪声导致保守化(F1-only,主指标无法呈现)**:所有任务 precision 保持 0.92–0.99 而 recall 崩塌——被扰动的策略是**漏音**而非**按错**,F1 的下跌几乎全部来自 recall。**return 结构上给不出这条**:它是一个标量,没有 precision/recall 的分解。要讲"失败的形态"就必须用 F1 侧的分解。

4. **Reward 噪声在主指标看似健康时已摧毁学习(C2 最强的案例)**:
   - `R-Gauss-P50`:return 1692,**仅 −9%**,看上去只是轻伤;F1 只剩 0.109,**−83%**,策略已经不在弹这首曲子。
   - `R-Shift-P50`:return **被抬高到 2141(+16%)**,主指标显示"变好了";F1 −6%。去掉灌入的 +294 后真值 1846.6(−0.2%)。
   **正因为 return 是主指标,这条才成立**:一个只报 return 的基准会把前者判为轻微退化、把后者判为改进。noise-immune 的第二指标不是锦上添花,是**判断主指标何时失效的唯一手段**。

5. **Action-Shift 的效果由符号决定,且强烈不对称——两个指标一致。**
   return:+0.15 → 1859(**+0%**),−0.15 → 1698(**−8%**);F1:+0.15 → 0.615(−2%),−0.15 → 0.545(−13%)。
   **机理假说(待验证)**:加噪后动作被 `clip` 到 [−1,1],故 −0.15 把可达范围压成 [−1, 0.85],**损失"完全按下"这一端**——而弹琴恰恰需要它(能力性损失,策略无法补偿);+0.15 只损失"完全抬起"端,可被补偿。验证方法:统计动作触顶/触底的裁剪率与按键力度分布。

6. **组合通道:随机噪声超可加退化,恒定偏置次可加——两个指标一致。**

   | 组合 | return(分量) | F1(分量) | 组合律 |
   |---|---|---|---|
   | **AO-Gauss** | **1624** < A 1671 / O 1832 | **0.177** < A 0.267 / O 0.562 | **超可加**(低于两个分量) |
   | **AR-Uniform** | **1717** < A 1802 / R 1774 | **0.232** < A 0.491 / R 0.327 | **超可加** |
   | **OR-Shift** | **1886**\* ≈ O-N15 1887 / R-N50 1775\* | **0.672** ≈ O-N15 0.677 / R-N50 0.493 | **次可加**(≈ 较好分量,有害分量被掩盖) |

   \* return 用 `ep_return_true`(OR-Shift 与 R-Shift-N50 都含 reward −0.50,实收值各偏低 294;OR-Shift 的真值为推算,见上方 ⚠️ 框)。**用实收值会把次可加读成超可加**——1592 低于两个分量,方向完全相反。这是本文件里 reward-shift 污染唯一一处**会翻转结论**的地方。

   两类扰动的组合律不同,是 RG 结构上无法测量的一类交互(它连多通道都不支持)。⚠️ 单 seed。

### 符号不对称性是**跨通道**性质,且"有益方向"因通道而异(2026-07-21 新增对照)

补跑了两个负向单通道对照(`O-Shift-N15`、`R-Shift-N50`),得到完整的 shift 家族:

| 通道 | shift = **+** | shift = **−** | 有益方向 |
|---|---|---|---|
| **Action**(±0.15) · return | 1859(+0%) | 1698(**−8%**) | 都不有益;负向明显更差 |
| **Action** · F1 | 0.615(−2%) | 0.545(−13%) | 同上 |
| **Observation**(±0.15) · return | 1832(−1%) | **1887(+2%)** | **负向有益** |
| **Observation** · F1 | 0.539(−14%) | **0.677(+8%)** | 同上 |
| **Reward**(±0.50) · return\* | 1847(−0.2%) | 1775(−4.1%) | 都有害;负向更差 |
| **Reward** · F1 | 0.589(−6%) | 0.493(−21%) | 同上 |

\* reward 行用 `ep_return_true`;实收值 2141 / 1481 各含 ±294 的灌入,**符号会被读反**(实收看上去是"正向大幅有益"),见 §5 的 ⚠️ 框。

**两个指标在三个通道上给出同一个方向判断**(action 双向有害、obs 负向有益、reward 双向有害),只是幅度不同——这是符号不对称性最扎实的一处:结论不依赖指标选择。

**机理**:obs shift 是"感知到的关节位置"的系统性偏置。−0.15 让策略**以为手指比实际按得更浅** → 补偿性多按 → recall 0.506→**0.554**(precision 同时升到 0.994)→ F1 与 return 双双超过 clean;+0.15 反之(recall 掉到 0.433)。这与 action 通道的裁剪机理是**不同的**机制,却产生同一类现象:**恒定偏置的效果由"偏置方向与任务需求的对齐关系"决定,而非由扰动强度决定**。零均值噪声无法产生这种现象——这是"只测一个符号就会得出错误结论"的直接证据。

### ⚠️ 一个待解释的反常:恒定 reward 偏置**不是**策略不变的

所有 episode 均为 **588 步定长**(已核实),因此"每步加常数 c ⇒ 每条轨迹回报加 588c ⇒ 轨迹排序不变 ⇒ 最优策略不变"这一 MDP 层论证**成立**。但实测两个方向都显著伤害性能(−6% / −21%)。效应只能来自 MDP 之外,候选有二:

1. **OAR 观测通路(主要嫌疑)**:本基准的威胁模型让 `obs["reward"]` 携带带噪 reward,故恒定 reward 偏置同时是**策略输入特征的恒定平移**——这一点完全不受上述不变性保护。若成立,则说明 reward 通道的影响**主要经观测通路传导,而非学习信号通路**。
2. **critic 学习动态**:γ=0.8 下折扣偏置 ≈ ±2.5;收敛后优势不变,但训练途中价值网络需重新拟合该偏移。

**判定实验已完成(2026-07-21)**:注册并训练 `R-Shift-{P50,N50}-NoOAR`(`action_reward_observation=False` → 切断 `obs["reward"]` 这条输入通路,但 reward 标量仍被扰动,故学习信号通路保留)。**对照基线必须是 Clean-NoOAR 0.520,不是 Clean 0.626**(单是关 OAR 就掉 17%)。

| **F1** | 各自的 clean 基线 | R-Shift-**P50** | R-Shift-**N50** |
|---|---|---|---|
| **OAR 开** | 0.626 | 0.589(−6%) | 0.493(**−21%**) |
| **OAR 关** | 0.520 | 0.472(−9%) | **0.558(+7%)** |

> **本实验只能用 F1,主指标在这里不可用。** 四个格子全部是 reward-shift 任务,`ep_return`
> 被 ±294 的常数灌入直接改写(§5 ⚠️ 框),而 `ep_return_true` 未在台账中逐格记录。
> 这不是取舍,是**主指标在这一格失效**——恰好也是保留 noise-immune 第二指标的最直接理由。

**结论(按符号分化,比原假设更细)**:
- **负向 shift 的大幅伤害是纯观测通路效应**——−21% 在切断 `obs["reward"]` 后**完全消失**(变为 +7%)。机制 1 坐实;
- **正向 shift 的小幅伤害不是 OAR 造成的**(−6% → −9%,同一量级),归因未定:可能是 critic 动态,也可能是单 seed 噪声。

**对论文的意义**:reward 通道的**主要效应经由观测通路传导**,而不是经由学习信号。这把 §4 的威胁模型声明从"实现细节"提升为**决定测得的鲁棒性数值的关键设定**——同一个 reward 扰动,在带 OAR 的基准上掉 21%,在不带 OAR 的基准上不掉。任何 robust 基准都必须声明这一点,否则数字无法互相解读。

⚠️ 单 seed;正向那一格的残余效应未归因。

### 威胁模型对照(finding 6 的延伸,数据部分可用)

同一批任务在"扰动对策略隐藏"(旧实现)与"扰动可观测"(现实现)两种威胁模型下各训练过一次:

| 任务(**F1**) | 隐藏扰动 | 可观测扰动 | Δ | 证据强度 |
|---|---|---|---|---|
| A-Shift-P15 | 0.704 | 0.615 | **−0.089** | **强**(两侧均确定性,协议无关) |
| A-Shift-N15 | 0.553 | 0.545 | −0.008 | **强**(同上) |
| A-Gauss-P15 | 0.277 | 0.267 ± 0.035 | ≈0 | 中(旧值为单集) |
| A-Uniform-P15 | 0.371 | 0.491 ± 0.051 | +0.120? | **弱**(旧值为单集,可能是不利抽样) |

> **本表是 F1-only,不是选择而是限制**:旧语义(隐藏扰动)的 run 已删除,只留下当时记下的
> F1 值,`ep_return` 未逐格记录且无法回算。若要在论文中以主指标呈现这条对照,**必须重训旧
> 语义的那一批**。以现有数据只能用 F1 陈述。

**可直接使用的结论**(仅基于强证据两行):**恒定正向偏置的"意外增益"在扰动变得可观测后消失**(+0.15:0.704 → 0.615),**而负向偏置的伤害不受可观测性影响**(−0.15:0.553 → 0.545,几乎不变)。这与 finding 5 的裁剪假说**相互印证**:可补偿的一侧(+)在策略能看见扰动后被补偿掉;不可补偿的一侧(−)是能力性损失,看得见也没用。

> **写作提醒**:表中 gauss/uniform 两行的旧值来自单集评测,与新的 10 集均值口径不同,**不要把它们当作严格对照**;若要在论文中主张"可观测性对随机噪声的影响",需要用同一协议重测旧语义(旧 run 已删除,需重训)。

---

## 6. 评测协议与指标(方法论部分)

- **主指标 = `ep_return`**(2026-08-28 决定),用**实际收到的(带噪)**回报,**不使用**去噪后的干净 return,即使我们能算出它。理由:现实部署中干净 reward 不可观测,智能体也只能看到带噪值,读者据以判断稳健性的应当是智能体真正经历的那条曲线(RG 亦如此)。选它作主指标还有一个务实理由:它是所有 RL 基准共有的量,跨基准可比,而 F1 是本任务特有的。
  - **代价**:reward-shift 任务的 return 被系统性平移(每步 +c ⇒ 每集 +588c),该格的 Δret 不可解读。**表格里不做修正**——这是扰动效应本身;标注 ⚠️ 并给出 `ep_return_true` 作为脚注(§5 的框)。
  - ⚠️ **与 `abstract.md` 的一条既有决策存在冲突,需要写作时定夺**:2026-07-20 用户拍板"**R-通道正文图必须用 `ep_return_true`**"。本节的"不做修正"是**表格**口径,那条是**正文图**口径,可以并存(表格报实收 + 脚注真值,R 通道正文图报真值),但两处措辞目前都写成了绝对句。**正文图到底用哪个,由写 robust 章节的同学统一。**
- **次要指标 = F1**(note-level,对照 MIDI 谱面),**始终记录并绘制**。它由模拟器内的**物理真值**算出(承袭上游 RoboPianist 的 `MidiEvaluationWrapper` 定义,未改动),因此**不被任何噪声通道污染**:三个通道的噪声都不会改变"哪个琴键真的被按下"的测量。数值变化完全来自策略行为的真实改变。
- **为什么次要指标不能省**(方法节的核心论证):主指标在 robust 设定下有两种失效方式,两种都只有 F1 能揭示——
  1. **被污染**:reward-shift 直接改写被测量的那个量(`R-Shift-P50` return +16%,实为 −0.2%);
  2. **系统性低估**:即便未被污染,return 对策略退化的敏感度也远低于 F1(最高档高斯噪声:Δreturn −1%…−10% 对 ΔF1 −10%…−83%)。
  F1 不是"另一个视角",它是**判定主指标何时可信的工具**。
- **两个指标的方差量级不同,采样预算按 F1 定**:同一批 10 集数据上 return CV 0.46–1.34%,F1 CV 2.6–13.5%。若只报 return,随机噪声任务的集数可以更少;因为要同时报 F1,统一取 10 集。
- **Matched eval 为默认**(`eval_noise_scale=1.0`):评测时的噪声与训练时同级。这是与 RG 可比的鲁棒性数字(RG 在训练级扰动下评测,没有"干净评测"的概念)。
- **鲁棒性曲线**:同一策略在 scale ∈ {0, 0.5, 1, 2, 4} 下重复评测,得到退化曲线(基础设施已就绪,曲线数据待跑)。

---

## 7. 必须 disclose 的 caveats

1. **单 seed**(seed=0)——多 seed 待补,是当前最大的统计弱点;
2. **仅 matched eval**——scale sweep 曲线尚未跑;
3. **主表仅最高强度档**(action/obs 的 0.15、reward 的 0.50),中间档待补;
4. **obs 噪声只作用于策略实际消费的本体感觉通道**:关节角 `joints_pos`、感知键态 `piano/state`、`sustain_state`,共 141 维;**不扰动 979 维的 goal(谱面前瞻)**。关节速度不在其中,因为**上游 RoboPianist 本就是 position-only 观测**(`joints_vel` 在上游定义但从未启用),策略从不观测速度、故无速度可扰——这是与上游对齐的**声明**,不是遗漏;
5. **σ 的物理单位跨通道不同**(action 是 canonical [-1,1] 的位置目标,obs 是各观测量自身的单位,reward 是 reward 单位),因此**不应跨通道比较同一数值 level 的"强度"**;
6. **仅 PPO**——robust 任务无安全约束,故不跑 PPOLag;
7. **主指标在 3 个格子上不可解读**:含 reward-shift 的任务(`R-Shift-P50`、`R-Shift-N50`、`OR-Shift-ON15-RN50`)的 `ep_return` 被常数灌入平移,Δret 无意义,必须读 `ep_return_true` 或 F1。已在 §5 逐格标注 ⚠️,但**论文正文必须显式声明**,否则读者会把 `R-Shift-P50` 的 +16% 当成鲁棒性优势。

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

- **C2(noise-immune metric)**:robust 腿是这条方法论卖点的**主要证据来源**。注意论证结构随主指标改判而变得更强,不是更弱:我们**照常把 return 作为主指标报告**(与所有 RL 基准可比),再用 F1 指出主指标在何处失效——被污染(`R-Shift-P50` +16% 实为 −0.2%)与系统性低估(高斯档 Δreturn −9% 对 ΔF1 −83%)。"benchmark 应当配一个噪声碰不到的第二指标"这一主张,由主指标自己的失效案例支撑,比"我们干脆不用 return"有力得多。
- **C1(unified benchmark)**:robust 是四条腿之一,与 safety / morphology / multi-agent 共享同一套任务基座和指标定义。
- 具体 claim 措辞见 `claims.md` §C4(数字更新后同步)。
