# OAR × Robust —— 与 mentor 讨论用说明

> ## ✅ 讨论已完成,决定如下(2026-07-21 与博后会议)
>
> 1. **prev_action 槽 → noised action(已实施)**:采用硬件磨损威胁模型——关节老化/磨损使真实执行的 action 就是 noised action,OAR 记录的就应是它。§3 描述的 Method 6 clean-a_cmd 覆写**已从代码移除**(robust_wrapper.py 删除 `_clean_physical`/action slice 缓存/a_cmd 保存),`obs["action"]` 恢复上游 OAR 天然行为 = physical(a_exec)。
> 2. **prev_reward 槽 → noised reward(维持现状)**:测量传感器磨损模型——观测到的、学习用的、记忆里的都是同一个 noisy reward。reward 槽覆写保留(它是必需的:reward 噪声在 OAR 之上的 gym 层注入,不覆写则槽里是 clean)。
>
> 即 §4.2 的"方案 A"胜出(reward),而 §3/§4.1 中 action 槽的"覆写回 clean"方案被推翻。正式记录见 `robust_task_design.md` §4.7 supersession 说明;以下正文保留为讨论时点的分析存档。

> 目的:向 mentor 讲清 OmniPiano 里 **ObservationActionRewardWrapper(OAR)** 的作用、
> 它在 wrapper 链中的位置、我们在 robust 任务里对 OAR 的具体处理(prev-action / prev-reward
> 分别放 clean 还是 noised),以及帮助判断"这个实施是否合理"所需的背景信息。
> 所有代码引用均为实测(2026-07-06),文件:行号可直接点开核对。

---

## 1. OAR 是什么、有什么作用、为什么 RoboPianist 默认开启

### 1.1 它做的事(一句话)
OAR 把 **上一步的 action 和上一步的 reward** 追加进当前观测 `obs`,作为策略网络额外的输入特征。

上游实现只有几行(`dm_env_wrappers/_src/observation_action_reward.py`):

```python
def _augment_observation(self, action, reward, timestep):
    observation = timestep.observation
    observation["action"] = action      # 上一步的 action
    observation["reward"] = reward      # 上一步的 reward
    return timestep._replace(observation=observation)
```

在 OmniPiano 的 ClairDeLune 任务里,这给 obs 向量额外加了 **46 维**(45 维 action + 1 维 reward):
开 OAR → obs = 1176 维;关 OAR → obs = 1130 维。

### 1.2 为什么这有用 —— POMDP / 信用分配视角
弹钢琴是一个**部分可观测**任务:纯前馈 MLP 策略只看到"当前时刻"的本体感觉 + 目标,
看不到"我上一帧手指往哪按、按出了多好的效果"。把 `(prev_action, prev_reward)` 喂进 obs,
等价于给无记忆的策略一点点**短期记忆**,帮助它:

- 维持连贯的多步按键序列(靠 prev_action);
- 感知"上一步弹得好不好"从而调整(靠 prev_reward)。

这是 meta-RL / 循环智能体里的标准做法(RL²、R2D2 都把 action+reward 拼进观测)。

### 1.3 为什么 RoboPianist 默认开
RoboPianist 原论文的训练配置就用 `--action-reward-observation`,把它作为默认。
我们沿用是为了 **与已发表基线可比**(记忆里也记着:基准设计默认沿用 RoboPianist / CleanRL 等既有惯例)。

### 1.4 实测:开/关 OAR 对性能的影响(ClairDeLune, PPO, seed=0, 5M steps)

| 配置 | obs 维度 | F1 | recall | precision |
|---|---|---|---|---|
| **Clean, OAR 开**(默认) | 1176 | **0.626** | 0.506 | — |
| **Clean, OAR 关** | 1130 | **0.520** | 0.400 | ≈ 不变 |
| **Δ(关掉 OAR)** | −46 | **−0.106 (−17%)** | **−0.106 (−21%)** | ≈ 0 |

结论:OAR 有**实打实但非生死攸关**的收益。掉的几乎全在 **recall**(漏音变多),
precision 基本不动 —— 即"没有动作记忆时,按下的还是对的,只是按得少了",与 1.2 的机理一致。

---

## 2. OAR 在 wrapper 链中的位置(自底向上流程图)

关键点先说:**OAR 位于 dm_env 层,在 `ConcatObservationWrapper` 之前、`CanonicalSpecWrapper` 之下。**
它必须在 Concat 之前(需要 Dict 结构才能按 key 塞入 action/reward);它在 Canonical 之下,
意味着它记录的 action 是**已经 rescale 到物理执行单位**的那个值 —— 这一点对 §3 很重要。

```
        ┌─────────────────────────────────────────────────────────────┐
        │                RL algorithm  (PPO / PPOLag …)                │
        │      输入: obs(含 OAR 的 prev a/r)   输出: action a_cmd∈[-1,1] │
        └───────────────▲───────────────────────────────┬─────────────┘
             (obs,reward,│ term,trunc,info)              │ action ↓
   ╔═════════════════════╪═══════════════════════════════╪══════════════╗
   ║  GYM 层 (gymnasium.Env, 5-元组接口)                                  ║
   ║                                                                     ║
   ║   RobustWrapper      ★ ← 这里注入 action 噪声                        ║
   ║                        ★ ← 这里给 reward 标量【加噪】               ║
   ║                        ★ ← 这里【覆写】obs 的 reward 槽             ║
   ║                            (2026-07-21 起 action 槽不再覆写)        ║
   ║   SafetyWrapper         ← 不改 reward,只把 cost 写进 info           ║
   ║   MetricsWrapper        ← 不改 reward,只记录各 reward 子项          ║
   ║   DmEnvToGymnasium      ← ts.reward → float,组 5-元组               ║
   ╚═════════════════════╪═══════════════════════════════╪══════════════╝
                         │        dm_env ↔ gym 边界        │
   ╔═════════════════════╪═══════════════════════════════╪══════════════╗
   ║  DM_ENV 层 (dm_env.Environment, TimeStep)                           ║
   ║                                                                     ║
   ║   SinglePrecisionWrapper       ← obs 转 float32                     ║
   ║   CanonicalSpecWrapper         ← action [-1,1] → 物理执行单位        ║
   ║   ConcatObservationWrapper     ← Dict obs → 扁平向量(此后无 key)   ║
   ║   ObservationActionRewardWrapper (OAR) ★                            ║
   ║        └─ 把 prev_action + prev_reward 塞进 obs Dict                 ║
   ║   [DmEnvObsNoiseWrapper]        ★ ← 这里注入 obs 噪声(仅 obs 通道)  ║
   ║   SafetyWrapper(dm) / MetricsWrapper(dm) / MidiEvaluationWrapper     ║
   ║   PianoTask (composer.Environment)                                  ║
   ╚═════════════════════╪═══════════════════════════════════════════════╝
                         │
              ┌──────────▼──────────┐
              │   MuJoCo 物理引擎     │  (MJCF: 双 Shadow Hand + 钢琴)
              └─────────────────────┘
```

代码对应(`omnipiano/envs/registration.py`):
- OAR 插入:`registration.py:431-432`(`if action_reward_observation: env = ObservationActionRewardWrapper(env)`)
- 紧接着 Concat:`registration.py:435`
- gym 层顺序:`registration.py:467-469`(Metrics → Safety → **Robust 最外层**)

**为什么 Robust 在最外层很关键**:action 噪声在 gym 最外层注入,再一路往下经过 Canonical 被 rescale;
等 action 到达 dm_env 层的 OAR 时,OAR 记录的已经是 **"物理单位的、被加过噪的执行动作" a_exec**,
而不是策略真正输出的干净 a_cmd。这正是 §3 需要"覆写"的原因。

---

## 3. 讨论时点的实施:prev-action 用 clean,prev-reward 用 noised(⚠️ action 部分已按会议决定改为 noised,见顶部)

### 3.1 上游默认会发生什么(如果我们什么都不做)
- **prev_action 槽**:OAR 在 Canonical 之下,记录的是 `a_exec = physical(a_cmd + 噪声)` → **含噪**。
- **prev_reward 槽**:OAR 记录 `timestep.reward`,即 dm_env 层的**任务原始 reward**;而 reward 噪声是在
  更上面的 gym RobustWrapper 才加的 → 所以上游默认 obs 里的 prev_reward 是 **clean(干净任务 reward)**。

也就是说,如果放任不管,会出现一个**不一致 + 泄漏**的组合:
action 记忆泄漏了噪声(策略能看到自己被扰动后的真实执行动作,从而学会抵消噪声,
让 action-robust 任务失去意义);而 reward 记忆却是干净的、和智能体实际学习用的 noisy reward 对不上。

### 3.2 我们的处理(Method 6 / Option B,`robust_wrapper.py:266-284`)
在 gym 最外层 `RobustWrapper.step` 里,**主动覆写** obs 的两个槽:

| obs 槽 | 覆写成 | 语义 | 代码 |
|---|---|---|---|
| `obs["action"]` | **clean commanded** `physical(a_cmd)` | 隐藏注入的 action 噪声,策略只记得"我本想怎么做" | `robust_wrapper.py:279-282` |
| `obs["reward"]` | **noised** 实际收到的 reward `r_obs` | 让 reward 记忆 = 训练信号,匹配 eval 时不漂移分布 | `robust_wrapper.py:283-284` |

两个覆写**各自独立**、对任意分布(gaussian/uniform/shift)都生效,只在对应通道 active 时才触发
(`override_action` / `override_reward`,`robust_wrapper.py:275-276`);非 robust 环境完全不碰 obs
(连 copy 都不做),因此所有已有 baseline **结构上不受影响**。

### 3.3 为什么这么设计
- **prev_action 放 clean** —— 遵循标准鲁棒 RL 威胁模型(Disrupted-MDP, Gu et al. 2025):
  策略对"自己的动作"的记忆应当是它**发出的指令**,而不是被环境篡改后的执行结果。
  否则策略能观测到噪声 → 学会精确反补偿 → action 噪声形同虚设,实验测不出鲁棒性。
- **prev_reward 放 noised** —— 对应的威胁模型是 **"reward 传感器被污染"**:
  智能体**观测到的、学习用的、记忆里的**是同一个被污染的值,三者自洽。
  同时保证 **train / eval 分布一致**(matched-eval,decision 11;Wang 2020):训练时 obs 里是 noisy reward,
  eval 时也是 noisy reward,不会因为 obs 分布漂移而额外掉分。

### 3.4 具体影响
- **action 通道**:噪声**不会**经 OAR 二次泄漏进 obs → action-robust 实验干净有效。
- **obs 通道**:obs 噪声加在白名单的 141 个物理维度上,**完全不碰** OAR 的 action/reward 槽 → 无耦合。
- **reward 通道**:唯一有耦合的通道。noisy reward 同时进入(a)训练目标 和(b)obs 特征(经 OAR reward 槽)。
  这是**有意为之**的"污染传感器"建模,不是 bug —— 但它确实是三个通道里唯一需要 mentor 重点判断的设计点(见 §4)。

---

## 4. 帮助判断"实施是否合理"的其他必要信息

### 4.1 三个通道 × OAR 的耦合一览(核心表)

| 噪声通道 | 是否经 OAR 泄漏进 obs? | 我们的处理 | 是否有争议 |
|---|---|---|---|
| **Action** | 默认会(记录 a_exec) | 覆写回 clean a_cmd | 否,标准做法 |
| **Obs** | 否(不碰 OAR 槽) | 无需处理 | 否 |
| **Reward** | 默认不会(OAR 记 clean),我们**主动改成 noised** | 覆写成 noisy reward | **是,唯一设计选择点** |

### 4.2 Reward 通道两种建模的对比(供 mentor 拍板)

| | 方案 A(现状):obs 槽放 **noised** reward | 方案 B:obs 槽放 **clean** true reward |
|---|---|---|
| 威胁模型 | reward 传感器被污染,观测=学习=记忆同一污染值 | 只污染**学习目标**,观测保持干净 |
| 一致性 | train/eval/obs 三者自洽 | 观测与学习信号不一致,需额外解释 |
| 代码 | 已实现(`robust_wrapper.py:283-284`) | 一行改动(reward 槽改放 true reward) |
| 适合的故事 | "损坏的奖励传感器" | "只有 critic 目标带噪" |

两者都合法,取决于想讲哪个故事。**我们目前选 A**,理由是自洽 + 与 matched-eval 协议一致。

### 4.3 "要不要干脆全局关掉 OAR"—— 已评估,不建议
曾考虑"所有 robust 任务默认关 OAR,一劳永逸绕开这个问题",结论是**不划算**:
1. **压缩动态范围**:关 OAR 把干净基线从 0.626 拉到 0.520(−17%),
   而鲁棒性研究测的正是"加噪掉多少",基线降低会让信噪比变差。
2. **丢失与上游可比性**:RoboPianist 默认开 OAR,关掉后 clean 数字无法和已发表结果对齐。
3. **收益是假象**:关 OAR 不删任何代码(覆写逻辑已实现且被测试锁定),只是改个 flag,并不简化架构。
4. **问题本就只在 reward 一个通道**(见 4.1),用"全局关 OAR"这把大锤解决一个通道的小问题不成比例。

因此保留 OAR 默认开,把 **Clean-NoOAR 作为一条消融轴**按需注册(env id
`OmniPiano-ClairDeLune-Clean-NoOAR-v0`,`envs/__init__.py:1297`),用来量化 OAR 贡献。

### 4.4 正确性保证(已有测试)
- `test_robust_multichannel.py`:A+R 组合时两个 obs 槽各自独立覆写、§14 逐步交叉校验
  (`received reward == clean 分解和 + reward_noise`)、同种子 bit 级可复现、frame_stack>1 fail-fast。
- Method 6 覆写用 dm_env_wrappers 自带的 `_scale_nested_action` + 缓存的 spec,保证与执行路径
  **逐比特一致**(`test_robust_v1_method6.py` 把关),避免自行重写 rescale 导致 1-ulp 偏差。

### 4.5 已知边界
- `frame_stack > 1` 不支持覆写(扁平 obs 变成逐帧交错,只能改到最新帧)→ fail-fast。
  所有 OmniPiano / RoboPianist 协议都用 `frame_stack = 1`,不受影响。
- OAR 关闭时(`action_reward_observation=False`)obs 无 action/reward key,覆写自动跳过,不报错。

---

## 5. (额外)reward 到底在哪里、什么时候传给 RL algorithm?

> 你的疑问:一般 RL pipeline 是"给 agent obs → agent 给 action → 环境反馈 reward"。
> 在 OmniPiano 里 reward 是不是也像 obs 那样,层层穿过 gym wrapper 才到 RL algo?什么时候到的?

**先厘清一个关键点:reward 有两条完全不同的路径,别混淆。**

### 路径 ①(标准 RL 路径):reward 是 `env.step()` 返回值的第 2 个元素
这才是"环境反馈给智能体、用来学习"的 reward。它**不是 obs 的一部分**,而是和 obs 并列的独立标量:

```
obs, reward, terminated, truncated, info = env.step(action)
                ↑
        RL 算法拿它算 return / advantage / critic 目标
```

它的生命周期(自底向上,和 §2 流程图同向):
1. **诞生**:MuJoCo 步进后,`PianoTask` 算出任务 reward。
2. **过 dm_env 边界**:`DmEnvToGymnasium.step` 把 `ts.reward` 转成 float,放进 5-元组
   (`dm_env_adapter.py:147, 158`)。
3. **穿过 gym 层**(逐层往上,和 obs 同一趟 `step` 调用里):
   - `MetricsWrapper`:**不改** reward,只记录子项。
   - `SafetyWrapper`:**不改** reward(明确注释 "MUST NOT modify the reward directly",
     `safety_wrapper.py:44-48`),cost 只写进 `info`。
   - `RobustWrapper`(最外层):**唯一**会改 reward 的地方 —— 若 reward 通道 active 就加噪
     (`robust_wrapper.py:258-264`)。
4. **交付**:`RobustWrapper.step` return 出来的那个 `reward`,就是 RL 算法本步拿到的最终奖励。

**所以:是的,reward 和 obs 一样,是在同一次 `env.step()` 里、层层穿过 gym wrapper 之后一起返回的**
—— obs 是元组第 1 个、reward 是第 2 个。时机上,**智能体在 t 步发出 action → 环境 step →
同一次返回里同时拿到 `obs_{t+1}` 和 `reward_t`**。这就是标准 gym 的
"agent 出 action,环境同步反馈 reward" 的时机,OmniPiano 没有改变它。

### 路径 ②(OAR 特征路径):reward 又被"抄"进 obs 向量里当特征
这是 §1–§3 讲的 OAR。它和路径 ① **独立且并行**:OAR 额外把(某个版本的)reward 塞进 obs 的一个槽,
让**策略网络**能把"上一步 reward"当输入特征来 condition。这纯粹是为了对付部分可观测(POMDP),
**不影响** reward 作为学习信号本身。

两条路径的关系,用一张小表说清:

| | 路径 ① 学习信号 reward | 路径 ② OAR 里的 prev_reward 特征 |
|---|---|---|
| 身份 | `env.step()` 返回的标量 | obs 向量中的 1 维 |
| 谁用它 | RL 算法(算 return / critic 目标) | 策略网络(当输入特征) |
| 时机 | 每步 step 返回时 | 下一步 obs 里携带 |
| 在 robust 里的值 | noisy reward | **也**被覆写成 noisy reward(§3,保持一致) |
| 可否关闭 | 不可(RL 必需) | 可(关 OAR 即去掉,obs 1176→1130) |

一句话总结:**reward 作为"学习信号"始终走路径 ①,在每次 `env.step()` 返回时交给 RL 算法;
OAR 只是把它的一个副本额外放进 obs 当"记忆特征"(路径 ②)。我们在 robust 任务里让两条路径的
reward 取同一个 noisy 值,保证智能体"学到的""记住的"是同一个被污染的奖励。**

---

### 附:关键代码索引
| 内容 | 文件:行 |
|---|---|
| OAR 上游实现 | `dm_env_wrappers/_src/observation_action_reward.py`(`_augment_observation`) |
| OAR 插入链中 | `omnipiano/envs/registration.py:431-432` |
| gym 层顺序(Metrics→Safety→Robust) | `omnipiano/envs/registration.py:467-469` |
| action 加噪 + a_cmd 保存 | `omnipiano/wrappers/robust_wrapper.py:228-238` |
| reward 加噪 | `omnipiano/wrappers/robust_wrapper.py:258-264` |
| obs 两槽覆写(Method 6) | `omnipiano/wrappers/robust_wrapper.py:275-284` |
| reward 过 gym 边界 | `omnipiano/envs/dm_env_adapter.py:147, 158` |
| Safety 不改 reward | `omnipiano/wrappers/safety_wrapper.py:44-48` |
| Clean-NoOAR 消融 env | `omnipiano/envs/__init__.py:1297` |
