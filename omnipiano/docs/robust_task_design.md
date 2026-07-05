# Robust Tasks (v1) —— 设计文档

**目的**：用于 review OmniPiano v1 robust 任务实施方案。记录架构决策、范围边界（v1 包含什么、什么延期）、与现有代码的 diff、以及单看代码无法体现的 paper 写作所需的设计动机。

**生成时间**：2026-06-27（在审计完 OmniPiano 现有 robust 代码 + Robust-Gymnasium 参考 + paper arXiv 2502.19652 之后）。

**作者意图**：reviewer 应当能够端到端读完本文档，回答"v1 究竟要实现什么、为什么、以及不实现什么"而无需翻源码。reviewer 也应当能够看到 Phase 0 后**逐项**批准或否决每个改动。

---

## 0. 范围 (v1) —— 包含什么、不包含什么

**v1 包含**：

- **3 个扰动 channel**：observation (state) noise、action noise、reward noise
- **每个 channel 3 种噪声分布**：Gaussian、Uniform、Constant Shift
- **噪声频率**：按分布类型自然确定（与 RG 完全一致）：
  - **Gaussian** 和 **Uniform**：**step-level**（每 step 独立采样，每个 step 得到一个新的随机噪声）
  - **Shift**：**program-run-level**（全程恒定 `+shift_value`，其中 `shift_value` 来自 `_noise_shift` 独立字段，无 RNG 参与）
  - 见 §0.4 for the per-distribution frequency semantics
- **2 套评估协议**：in-training（训练和评估都开噪声）+ post-training（训练开噪，评估不开）
- **形态**：仅 bimanual（2-hand SA）
- **曲目**：ClairDeLune 作为首曲
- **首批 task 注册**：1 曲 × 3 channel × 3 分布 × 约 3 噪声级别 = 约 27 个 task id（具体清单见 §6）

**明确不做（out of scope，非延期）**：

- **Dynamics / environment 扰动**（mass / friction / actuator gain / spring stiffness 等）—— **从 plan 中移除**（2026-07-03 决定）。理由:dynamics randomization 的核心动机是 sim-to-real transfer,而 OmniPiano 是纯仿真、无真实硬件目标的 benchmark,该动机不成立;且改物理参数容易纠缠 F1 测量的语义(如 key spring 改变力→键激活映射),破坏"robustness vs 换任务"的边界;工程上还需 MJCF composer surgery(数百可改参数 + MidiEval 缓存风险)换取薄弱收益。覆盖 obs/action/reward 三个 disruptor family 已是干净可辩护的 scope。详见 §11。

**延期到 v2+（v1 不包含,但保留未来做的可能）**：

- 任何形式的对抗扰动（LLM-prompted、gradient-based、worst-case ball、单独训练的对抗 policy）
- Episode 级噪声频率（每个 reset 采一个噪声，整 episode 固定）
- Per-dimension 噪声 std（每个 action 维度或 obs key 不同 std）
- N-hand 单 agent robust 变体（3/4/5-hand Prototype/StaticPartition）
- 多 agent（PettingZoo ParallelEnv）robust 变体

**理由**：见 §11 "为什么这些不做 / 延期"。

---

## 1. 代码库当前状态（2026-06-27 审计完成）

### 1.1 已存在且能正常工作的部分

现有 robust 基础设施由 3 个文件实现：


| 文件                                     | 角色                                                                        | 状态                       |
| -------------------------------------- | ------------------------------------------------------------------------- | ------------------------ |
| `omnipiano/configs/__init__.py:26-31`  | `RobustConfig` dataclass，含 `action_noise_std`、`obs_noise_std`（均默认 0.0）    | 极简，需要扩展                  |
| `omnipiano/wrappers/robust_wrapper.py` | gym 层 `RobustWrapper` —— 注入 action noise + 读取并报告 dm_env 层的 obs noise L2   | action 部分正确，将扩展支持 reward |
| `omnipiano/envs/dm_env_obs_noise.py`   | dm_env 层 `DmEnvObsNoiseWrapper` —— 对选中的 obs key 注入 per-key Gaussian noise | 正确，将扩展支持分布选择             |


`omnipiano/envs/registration.py` 中的衔接：

- `RobustConfig` 已接入 `TaskSpec`（line 148）和 `make()`（line 257）
- `DmEnvObsNoiseWrapper` 在 dm_env chain 的 line 347-352 插入，位置在 `ConcatObservationWrapper` **之前**（这个位置是正确的，**不能改动**）
- Obs-noise RNG 通过 `master_seed + 31415` magic offset 设置 seed（line 347）

已注册了 2 个 robust task（`omnipiano/envs/__init__.py:222-229, 276-283`）：

- `OmniPiano-FantaisieImpromptu-ActionRobust-v0`：`action_noise_std=0.01, obs_noise_std=0.0`
- `OmniPiano-ClairDeLune-ObservationRobust-v0`：`action_noise_std=0.0, obs_noise_std=0.01`

这两个为了向后兼容**保留原样**，但**不**属于 v1 批次；它们没有遵循 v1 命名/级别规范。Phase 1 可以删除。

### 1.2 已经正确、**不应改动**的部分

**这些设计决策是好的，我们把它们作为 v1 的基线保留。** Reviewer 应特别确认这些应被保留再开始 Phase 0：

1. **双层噪声注入**（obs 在 dm_env 层、action 在 gym 层）。原因：`ConcatObservationWrapper` 在到达 gym 层之前把 Dict obs 拍平成 ndarray；在 dm_env 层 obs 仍是 Dict，per-key 选择(白名单跳过 goal/fingering 等)**最自然**。
  **准确说法(不是"只能")**：per-key 选择性**并非只能**在 dm_env 层实现。§4.7 的方案 6 已经证明,gym 层代码可以从 dm_env Dict spec 反推出每个 key 在 flat obs 里的 slice(字母序累加 dim),所以理论上 obs 噪声**也可以**在 gym 层按 slice 注入。我们**刻意保留在 dm_env 层**是权衡后的设计选择,不是硬性必要：(a) 在 Dict 上按 key 名注入,无需 slice 记账,也不依赖字母序假设(那个假设只有方案 6 的 override 路径才承担);(b) 与既有 2 个 ObservationRobust task 的实现零变动、bit-exact 兼容;(c) 独立的 dm_env RNG stream 已就位(§4)。action 注入放 gym 层是正确的,因为 action 一直是 flat ndarray。
2. **Per-dim 独立采样**：`numpy_rng.normal(0, std, size=action.shape)`（`robust_wrapper.py:47-49`）和 `self._rng.normal(0.0, noise_std, size=value.shape)`（`dm_env_obs_noise.py:118`）。每个 action 维度和每个 obs 标量都得到独立的噪声样本。
  **与 Robust-Gymnasium 对比**：他们的 `ant_v5.py:359` 用 `action = action + random.gauss(mu, sigma)`，使用 python 返回标量的 `random.gauss` 然后广播单一噪声值到所有维度。这是 bug（rank-1 noise 而不是 full-rank Gaussian）。OmniPiano 避免了这个 bug。
3. **两个独立的 numpy RNG** —— action noise 用 `gym.Wrapper.np_random`（由 `env.reset(seed=...)` 自动 seed），obs noise 用自己的 `np.random.default_rng(seed)` 实例并加上 offset。两个 channel 互不扰乱彼此序列。两者都可复现。
  **与 Robust-Gymnasium 对比**：他们用 python 的 `import random` 是全局状态、且不是 vec-env safe 的。
4. **Obs noise per-key 白名单过滤**（`dm_env_obs_noise.py:64`）：仅向匹配 `joints_pos / joints_vel / piano/state / piano/sustain_state` 的 key 注入。跳过 counter / goal / action observation 这类加噪没意义的 key。保留此白名单，逻辑正确。
5. **`robust_config` 模型**：每个注册的 env id 用自己的 `RobustConfig` 固定一个特定的噪声级别。**无**运行时配置切换。与 `safety_config` 的方式一致（只在 registry 配置，不在 make() 时覆盖）。

### 1.3 Phase 0 要修的 bug 和不清晰之处


| #   | 位置                                                            | 问题                                                                                                                                                  | 严重度                                            |
| --- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| 1   | RobustConfig                                                  | 没有 eval 时噪声开关 —— 无法跑"对一个噪声训练的 policy 做净 eval"或"在 eval 时扫不同噪声级别"                                                                                     | **CRITICAL**（paper-blocking）                   |
| 2   | `logger_wrapper.py:75-91` + `checkpoint_replay_eval.py`       | RobustWrapper 已 emit 每 step `ROBUST_NOISE_`*，但两个 eval CSV 都没聚合，且缺 `eval_noise_scale` 列（sweep 无法定位曲线点）。完整规格见 §0.6                                    | **CRITICAL**（reproducibility + sweep-blocking） |
| 3   | `envs/registration.py:347` 和 `multiagent/registration.py:319` | `master_seed + 31415` 是 hardcoded magic，且 MA registration 有**重复副本**；无注释、无命名常量。风险：重构一处会悄悄漏掉另一处 → 同一 master seed 下 MA 和 SA env 会得到不同的 obs-noise seed。 | **MODERATE**（可读性 + 可维护性）                       |
| 4   | `RobustConfig` docstring                                      | 有 dynamics randomization 的 TODO 占位符 —— 既然 v1 明确延期，应该删掉                                                                                              | **MINOR**（清理）                                  |
| 5   | `dm_env_obs_noise.py`                                         | 没有 `eval_mode` 开关；只要 `noise_std > 0` 就总是注入                                                                                                          | **CRITICAL**（与 #1 相同问题，但在 dm_env 层那侧）          |
| 6   | `envs/registration.py`（`make()` 签名）+ 调用点（`run_sb3_*.py` 等 SB3 模板）。**注**：`checkpoint_replay_eval.py` 不调 `make()`（走 `omnisafe.Evaluator`），无此 kwarg，不涉及 | `make()` 的 `log_split` 参数改名为 `mode`（纯 rename：值 `"train"`/`"eval"` 不变、当前行为不变、`log_dir` 独立不动、**不加第二个 flag**）。理由：Phase 0 后此 flag 兼管 eval 噪声语义，`log_split`（logging 名）控制 env 动力学会"名字撒谎"，`mode` 名副其实。详见 §12 决议 9 | **MINOR**（命名清晰；趁 Phase 0 未焊噪声行为时一并做） |


---

## 2. v1 架构概览

v1 改动后完整的 wrapper chain：

```
dm_env 层 (在 gym adapter 之前)：
    suite.load_with_task(OmniPianoTask, ...)
    → EpisodeStatisticsWrapper
    → MidiEvaluationWrapper
    → [DmEnvObsNoiseWrapper if robust_config.obs_noise_std > 0]    ← 不变
    → [ObservationActionRewardWrapper if action_reward_observation]
    → ConcatObservationWrapper
    → [FrameStackingWrapper if frame_stack > 1]
    → CanonicalSpecWrapper(clip=clip)
    → SinglePrecisionWrapper

gym 层 (在 dm_env adapter 之后)：
    DmEnvToGymnasium
    → MetricsWrapper
    → SafetyWrapper
    → RobustWrapper            ← 扩展：同时处理 reward noise
    → [SafeRecordEpisodeStatistics if mode == "eval"]   ← 扩展：记录 noise L2
```

**为什么 `RobustWrapper`（而不是新建 `RewardNoiseWrapper`）处理 reward noise**：

- Reward noise 必须在 `SafetyWrapper` 之后（因为 `SafetyWrapper` 不修改 reward —— 它只往 info 加 cost —— 所以 RobustWrapper 与 SafetyWrapper 关于 reward 的顺序在功能上没有区别，但 action noise 应该放在最靠近 env 的边界，使所有下游 wrapper 都能看到扰动后的 action；reward 注入复用同一个边界位置）
- 把所有噪声 bundle 到一个 wrapper，让 `info["robust/noise_*_l2"]` 全部集中在一处，更易于 log/审计
- 一个 wrapper = 一个 RNG seeding 路径（仍然是 2 条独立 RNG stream：action+reward 共享一条 —— 见 §4 RNG 设计）

---

## 3. Phase 0 —— bug 修复 + 基础（必须完成才能进 Phase 1）

### 目标

修复 §1.3 的 6 个问题/改进项（#6 = 决议 9 的 `log_split`→`mode` 纯 rename）。原意"不改变现有任何语义"在**训练路径**成立（action 噪声 bit-identical，§0.7 gate 验证），但有**三处有意的语义变更**（记录在案的例外）：(a) 决议 10 —— eval 默认改 matched（`eval_noise_scale=1.0`）；(b) 决议 11 —— reward eval matched（撤 force-zero）+ `obs["reward"]`=r_obs；(c) S5 —— obs-noise seed 31415→20000（obs 噪声序列有意改变）。

### Phase 0 完成状态（as-built ✅，全绿）

Phase 0 全部子阶段完成（分支 `feat/robust-phase0`）：

| 子阶段 | 内容 | test 文件 |
|---|---|---|
| S1 | RobustConfig 14 字段 + 校验 | `test_robust_config.py` |
| S2 | 噪声语义（`RobustConfig.is_channel_active`/`sample_noise`） | `test_robust_sample_noise.py` |
| S3a | `eval_noise_scale` 接线 + `log_split→mode` | `test_robust_eval_scale.py` |
| S3b | obs uniform/shift @ dm_env 层 | `test_robust_obs_dist.py` |
| S4 | reward noise + Method-6 reward-slot override | `test_robust_reward_noise.py` |
| S5 | `OBS_NOISE_SEED_OFFSET=20000` 去重 | `test_obs_noise_seed_consistency.py` |
| S6 | eval CSV robust 列（schema-locked） | `test_robust_eval_csv.py` |
| §0.7 | 2 个 v1 env equivalence（可复现 + 通道 + gaussian bit-exact） | `test_robust_v1_equivalence.py` |
| 贯穿 | Method 6 gates | `test_robust_v1_method6.py`（15/15） |

**全套 84 passed。** 语义决策见 §12 决议 9/10/11。（`examples/` SB3 模板已追踪；`examples/checkpoint_replay_eval.py` 的 schema-lock 改动在磁盘但 `tools/` 被 gitignore。）

### Phase 0 工作项

**0.1 — RobustConfig：per-distribution 独立字段**（`omnipiano/configs/__init__.py`）

**设计原则（Option C.3，2026-06-27 决议）**：3 种分布用**完全独立的字段**，字段名严格对应其数学参数。这**严格模仿 Robust-Gymnasium** 的设计（`noise_sigma` for Gaussian、`uniform_low`/`uniform_high` for Uniform、`noise_shift` for Shift），字段总数达到 14 但语义**零歧义**。

```python
import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class RobustConfig:
    """Configuration for robustness perturbations.

    Per-distribution independent fields:
      - Gaussian σ:          `*_noise_std`             (3 channels × 1 field = 3)
      - Uniform [low, high]: `*_noise_uniform_low`,
                             `*_noise_uniform_high`    (3 channels × 2 fields = 6)
      - Shift constant:      `*_noise_shift`           (3 channels × 1 field = 3)
      - Distribution knob:   `noise_dist`              (1)
      - Eval scale:          `eval_noise_scale`              (1)
    Total: 14 fields.

    At training time, RobustWrapper reads (channel, noise_dist) → picks
    the corresponding field(s). At evaluation time (via
    omnipiano.make(env_name, mode="eval")), all magnitude fields
    are multiplied by `eval_noise_scale`:
      - eval_noise_scale=1.0 (default) → matched eval (same noise as training,
        in-training protocol) — the RG-comparable robustness number
      - eval_noise_scale=0.0           → clean/nominal eval (post-training)
      - Values in between        → robustness curve sweep

    Uniform bounds use NATURAL parameters (Option 4a, §12) — NO std-matching:
      - Symmetric (default v1 registrations): user sets
        `low = -level`, `high = +level` (e.g. level ∈ {0.05, 0.10, 0.15})
      - Asymmetric (advanced users can register): user sets
        `low = -0.02, high = +0.10` to model biased sensor drift, etc.
      __post_init__ validates that `low <= high` and both are finite.
    """

    # === Gaussian σ (per channel) ===
    action_noise_std: float = 0.0
    obs_noise_std: float = 0.0
    reward_noise_std: float = 0.0

    # === Uniform [low, high] (per channel, allows asymmetric) ===
    action_noise_uniform_low: float = 0.0
    action_noise_uniform_high: float = 0.0
    obs_noise_uniform_low: float = 0.0
    obs_noise_uniform_high: float = 0.0
    reward_noise_uniform_low: float = 0.0
    reward_noise_uniform_high: float = 0.0

    # === Shift constant offset (per channel) ===
    action_noise_shift: float = 0.0
    obs_noise_shift: float = 0.0
    reward_noise_shift: float = 0.0

    # === Distribution selector ===
    noise_dist: Literal["gaussian", "uniform", "shift"] = "gaussian"

    # === Eval-time noise multiplier (applied to ALL magnitude fields) ===
    # effective eval noise = registered training noise × eval_noise_scale.
    # 0.0 = clean eval; 1.0 = same as training; >1 = stress test.
    # (Named eval_noise_scale, not eval_scale, to make explicit that it
    #  scales the *noise* and is a multiplier relative to the training
    #  level — see §0.6.2 for why it must also be logged per eval episode.)
    eval_noise_scale: float = 0.0

    def __post_init__(self):
        """Validate:
        1. Uniform bounds are ordered (low <= high) and finite.
        2. Warn if a channel has multiple distribution fields set
           simultaneously (only one will be active based on noise_dist).
        """
        for ch in ("action", "obs", "reward"):
            lo = getattr(self, f"{ch}_noise_uniform_low")
            hi = getattr(self, f"{ch}_noise_uniform_high")
            if not (np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(
                    f"RobustConfig: {ch}_noise_uniform bounds must be "
                    f"finite, got low={lo}, high={hi}"
                )
            if lo > hi:
                raise ValueError(
                    f"RobustConfig: {ch}_noise_uniform_low ({lo}) must be "
                    f"<= high ({hi})"
                )

        # Cross-field consistency check: if user sets multiple
        # distribution's fields simultaneously with a mismatched
        # noise_dist, raise early (catches accidental mis-registration).
        active_dist = self.noise_dist
        for ch in ("action", "obs", "reward"):
            std_set = getattr(self, f"{ch}_noise_std") != 0.0
            unif_set = (
                getattr(self, f"{ch}_noise_uniform_low") != 0.0
                or getattr(self, f"{ch}_noise_uniform_high") != 0.0
            )
            shift_set = getattr(self, f"{ch}_noise_shift") != 0.0

            if active_dist == "gaussian" and (unif_set or shift_set):
                raise ValueError(
                    f"RobustConfig: noise_dist='gaussian' but "
                    f"{ch}_noise_uniform_* or {ch}_noise_shift is "
                    f"nonzero. Use {ch}_noise_std instead, or set "
                    f"noise_dist to match the intended distribution."
                )
            if active_dist == "uniform" and (std_set or shift_set):
                raise ValueError(
                    f"RobustConfig: noise_dist='uniform' but "
                    f"{ch}_noise_std or {ch}_noise_shift is nonzero. "
                    f"Use {ch}_noise_uniform_low/high instead."
                )
            if active_dist == "shift" and (std_set or unif_set):
                raise ValueError(
                    f"RobustConfig: noise_dist='shift' but "
                    f"{ch}_noise_std or {ch}_noise_uniform_* is nonzero. "
                    f"Use {ch}_noise_shift instead."
                )
```

**字段数总结**：


| 类别             | 字段                                             | 数量     |
| -------------- | ---------------------------------------------- | ------ |
| Gaussian σ     | `{action,obs,reward}_noise_std`                | 3      |
| Uniform bounds | `{action,obs,reward}_noise_uniform_{low,high}` | 6      |
| Shift 常数       | `{action,obs,reward}_noise_shift`              | 3      |
| Meta           | `noise_dist`, `eval_noise_scale`               | 2      |
| **合计**         |                                                | **14** |


**为什么选 Option C.3（per-distribution 独立字段）**：

1. **数学严谨**：`_noise_std` 严格指 Gaussian σ；`_noise_uniform_low/high` 严格指均匀分布上下界；`_noise_shift` 严格指常数偏移。**任何字段名都不误命名任何分布**（尤其避免了"用 std 描述 Shift 常数"这种误导）。
2. **paper 术语无歧义**：写论文引用 "`action_noise_std=0.05` (Gaussian σ)" 时读者立刻知道这是 Gaussian 的标准差，不会误解为均匀分布或常数偏移的某个参数。
3. **对齐 RG 术语**：RG 用 `--noise-sigma`（Gaussian）、`--uniform-low`/`--uniform-high`（Uniform）、`--noise-shift`（Shift）三组独立参数。paper 里可以说 "OmniPiano's per-channel fields directly mirror Robust-Gymnasium's `--noise-sigma`, `--uniform-low/high`, `--noise-shift` parameterization."
4. **允许非对称 uniform**：`uniform_low = -0.02, uniform_high = +0.10` 是合法配置，用于建模"有偏 sensor drift" 之类的实验。v1 默认注册的 tasks 用**对称 std-matched uniform**（`low = -std·√3, high = +std·√3`），但基础设施允许高级用户自己注册非对称任务。
5. **明确的 post_init 校验**：捕获"noise_dist=uniform 但只设了 std" 这种错配，让 config 错误在启动时就 fail 而不是静默漂移。

**为什么 `noise_dist` 是全 channel 共享的标量**：保持"每个 registered env 对应一种 noise 类型"的清晰对应（与 RG `--noise-type` 一致）。Per-channel dist 组合（如"state 用 Gauss + action 用 Uniform"）v1 不支持。

**Uniform 对称性约定（Option 4a，§12 决议，as-built）**：v1 registered tasks 用**对称 uniform 的自然参数**，**不做 std-matching（√3）转换**：

```python
# level ∈ {0.05, 0.10, 0.15}（action/obs）;对称 → low = -level, high = +level
cfg.action_noise_uniform_low  = -level
cfg.action_noise_uniform_high = +level
```

3 种分布共用同一 numerical `level` 作 x 轴（但 empirical std 各不同：Gaussian=σ、Uniform=level/√3、Shift=0，见 §4.5 / §5.1.4），严格对齐 Robust-Gymnasium 的自然参数化（`--uniform-low/high`）。**非对称 uniform 是基础设施保留能力**，用户可自定义 task 注册非对称配置（例如 `uniform_low=-0.02, uniform_high=+0.10` 建模有偏 sensor drift）。

**为什么删除 `# TODO: Add dynamics randomization configs`**：dynamics randomization 已 out of scope（见 §0 / §11），不是待办;v1 代码中不留会误导的 TODO 标记。

**0.2 — `eval_noise_scale` 在 make() 中的接线**（`omnipiano/envs/registration.py`）

> **术语（决议 9）**：`make()` 的参数名是 `mode`（`log_split` 是改名前的旧名，见 §12 决议 9）；as-built 代码已统一为 `mode`。

`omnipiano.make()` 用 `mode="train"` 或 `mode="eval"` 调用。我们让 `mode` 影响 `RobustConfig` 如何被消费。**因为 Option C.3 有 3 种独立分布字段（12 个 magnitude 字段），`replace()` 需要覆盖所有 12 个**：

```python
def make(env_name, mode="train", **kwargs):
    ...
    robust_config = task_spec.robust_config or RobustConfig()

    # 根据 mode 决定 scale
    if mode == "eval":
        scale = robust_config.eval_noise_scale
    else:
        scale = 1.0

    # Scale 所有 magnitude 字段（跨 3 分布 × 3 channel = 12 字段）
    effective_robust_config = replace(
        robust_config,
        # Gaussian σ
        action_noise_std=robust_config.action_noise_std * scale,
        obs_noise_std=robust_config.obs_noise_std * scale,
        reward_noise_std=robust_config.reward_noise_std * scale,
        # Uniform bounds (scale 保持 low/high 的对称/非对称结构)
        action_noise_uniform_low=robust_config.action_noise_uniform_low * scale,
        action_noise_uniform_high=robust_config.action_noise_uniform_high * scale,
        obs_noise_uniform_low=robust_config.obs_noise_uniform_low * scale,
        obs_noise_uniform_high=robust_config.obs_noise_uniform_high * scale,
        reward_noise_uniform_low=robust_config.reward_noise_uniform_low * scale,
        reward_noise_uniform_high=robust_config.reward_noise_uniform_high * scale,
        # Shift 常数
        action_noise_shift=robust_config.action_noise_shift * scale,
        obs_noise_shift=robust_config.obs_noise_shift * scale,
        reward_noise_shift=robust_config.reward_noise_shift * scale,
    )

    # NOTE (as-built, 决议 11): reward is scaled by eval_noise_scale like
    # action/obs — it is NOT force-zeroed at eval (the earlier force-zero
    # guard + warn shown in older revisions was removed). Because
    # ObservationActionRewardWrapper feeds the noised reward into
    # obs["reward"] (a policy input), matched-eval keeps it in-distribution;
    # the true (denoised) return is a separate CSV column (ep_return_true).
    # See §12 决议 11.

    # 然后 RobustWrapper 和 DmEnvObsNoiseWrapper 消费的是
    # effective_robust_config，而不是原始的。
    ...
```

**执行点 = `make(mode="eval")` 这一个 choke point**：**经此入口构造** eval env 的驱动器（SB3 `EvalCallback`、`_final_eval`、`SafeRecordEpisodeStatistics`，以及未来的 `robust_eval_sweep.py`）自动继承 `eval_noise_scale` 缩放（**三通道对称,含 reward——决议 11,不再 force-zero**），logger 侧无需 reward 特判。**例外**：`checkpoint_replay_eval.py`（OmniSafe，用 `Evaluator.load_saved()` 重建 env）**绕过 `make()`，不自动继承**——该缺口见 §11 deferred。

**Scale 对 uniform bounds 的语义**：naive multiplication 保持"structure of the bias"。举例：


| 原始 config                          | scale=0  | scale=0.5          | scale=1          | scale=2          |
| ---------------------------------- | -------- | ------------------ | ---------------- | ---------------- |
| symmetric `low=-0.05, high=+0.05`  | `[0, 0]` | `[-0.025, +0.025]` | `[-0.05, +0.05]` | `[-0.10, +0.10]` |
| asymmetric `low=-0.02, high=+0.10` | `[0, 0]` | `[-0.01, +0.05]`   | `[-0.02, +0.10]` | `[-0.04, +0.20]` |


- scale=0 → 所有 bounds 归零 → 无噪声（clean eval，与 Gaussian/Shift 一致）
- scale 保持对称 config 仍对称、非对称 config 仍非对称（乘法保持比例）
- Uniform 的 empirical std 也自动按 scale 缩放

`replace()` 来自 `dataclasses`（廉价不可变拷贝）。原始 `robust_config`（来自 registry）保持未修改，以便 reproducibility tracing。

**已决议（§12 决议 7，2026-06-27）**：**不**向一般用户暴露公开的 `eval_noise_scale_override`（遵循与 `safety_config` 一样的"只在 registry 配置"哲学）；但**为 sweep tool 保留一个私有 kwarg `_eval_noise_scale_override=`**（带 `_` 前缀、仅 `tools/robust_eval_sweep.py` 使用），让它在**单个注册 env 上运行时覆盖 scale**，从而避免为每个 `(env × scale)` 四元组注册上百个 env。详见 §12 决议 7 与 §5.2。

**0.3 — RobustWrapper 中加 reward noise 注入**（`omnipiano/wrappers/robust_wrapper.py`）

因为 Option C.3 每分布字段独立，RobustWrapper 通过一个 `_channel_active` helper 判断某 channel 该分布是否启用：

```python
def _channel_active(self, channel: str) -> bool:
    """Return True if `channel` has nonzero noise magnitude for the
    currently-active noise_dist.
    """
    dist = self.config.noise_dist
    if dist == "gaussian":
        return getattr(self.config, f"{channel}_noise_std") != 0.0
    if dist == "uniform":
        lo = getattr(self.config, f"{channel}_noise_uniform_low")
        hi = getattr(self.config, f"{channel}_noise_uniform_high")
        return (lo != 0.0) or (hi != 0.0)
    if dist == "shift":
        return getattr(self.config, f"{channel}_noise_shift") != 0.0
    raise ValueError(f"Unknown noise_dist: {dist!r}")

def step(self, action):
    # 1. Action noise
    action_noise_l2 = 0.0
    if self._channel_active("action"):
        noise = self._sample_noise(
            self.np_random, channel="action", shape=action.shape
        )
        action_noise_l2 = float(np.linalg.norm(noise))
        action = np.clip(
            action + noise, self.action_space.low, self.action_space.high
        )

    obs, reward, terminated, truncated, info = self.env.step(action)

    # 2. Obs noise L2 read (unchanged — dm_env layer handles injection)
    obs_noise_l2 = 0.0
    if self._channel_active("obs"):
        obs_noise_l2 = self._read_obs_noise_l2()

    # 3. Reward noise —— 在 return 前对标量 reward 操作。
    reward_noise = 0.0
    if self._channel_active("reward"):
        reward_noise = self._sample_noise(
            self.np_random, channel="reward", shape=()  # scalar
        )
        reward = float(reward) + float(reward_noise)

    info[InfoKeys.ROBUST_NOISE_ACTION_L2] = action_noise_l2
    info[InfoKeys.ROBUST_NOISE_OBS_L2] = obs_noise_l2
    info[InfoKeys.ROBUST_NOISE_REWARD] = float(reward_noise)   # 新 info key

    return obs, reward, terminated, truncated, info
```

**注意**：`_sample_noise` 现在接受 `channel` 参数（不再是 `std`），内部根据 channel + dist 决定读哪个字段。见 §0.4 for 完整实现。

**Reviewer 待决问题**：reward noise 应该是 **post-cost-penalty**（即对 agent 看到的最终 reward 加噪）还是 **pre-cost-penalty**（对 env 原生 reward 加噪，在 SafetyWrapper 做任何修改之前）？SafetyWrapper 当前代码不修改 reward（仅向 info 添加 cost），所以两种今天等价。建议：**把 RobustWrapper 放在 SafetyWrapper 之后**（当前 chain 顺序），让 reward noise 加在最终 reward 上，防御性应对 SafetyWrapper 未来可能学会修改 reward 的情况。

✅ **已决定（2026-06-27）**：采纳建议，保持当前 chain 顺序（`MetricsWrapper → SafetyWrapper → RobustWrapper → [SafeRecordEpisodeStatistics]`）。reward noise 加在最外层。

**0.4 — 分布采样 `_sample_noise` helper**（RobustWrapper 新私有方法）

在 Option C.3 下，`_sample_noise` 根据 `(channel, noise_dist)` 从 RobustConfig 里读对应字段，然后从对应分布采样：

```python
def _sample_noise(self, rng, channel: str, shape):
    """Sample noise from the distribution active for `channel`.

    Field lookup convention (matches RobustConfig Option C.3):
      - noise_dist="gaussian" → read `{channel}_noise_std` as σ,
        sample from N(0, σ²) per-dim (step-level).
      - noise_dist="uniform"  → read `{channel}_noise_uniform_low` and
        `{channel}_noise_uniform_high`, sample uniformly per-dim
        (step-level). Bounds may be asymmetric (biased noise);
        v1 registered tasks use symmetric std-matched bounds.
      - noise_dist="shift"    → read `{channel}_noise_shift`, apply
        as constant broadcast to all dims (program-run-level).
        NO RNG draw; deterministic.
    """
    dist = self.config.noise_dist
    if dist == "gaussian":
        std = getattr(self.config, f"{channel}_noise_std")
        return rng.normal(0.0, std, size=shape)
    if dist == "uniform":
        lo = getattr(self.config, f"{channel}_noise_uniform_low")
        hi = getattr(self.config, f"{channel}_noise_uniform_high")
        return rng.uniform(lo, hi, size=shape)
    if dist == "shift":
        shift = getattr(self.config, f"{channel}_noise_shift")
        # Deterministic constant offset, broadcast to all dims. NO RNG
        # draw (so shift branch never perturbs Gaussian/Uniform stream
        # ordering when the same env is re-used with a different dist).
        # Matches Robust-Gymnasium ant_v5.py:361 exactly:
        #   action = action + args.noise_shift  →  our +shift constant.
        return np.full(shape, shift, dtype=float)
    raise ValueError(f"Unknown noise_dist: {dist!r}")
```

**关于分布 vs 频率的关系（重要澄清）**

**3 种分布在 v1 中的 frequency 是"分布类型自然决定"的**，不是独立可选的 knob：


| 分布           | Frequency             | 每 step 的行为                                   | RNG 参与？     |
| ------------ | --------------------- | -------------------------------------------- | ----------- |
| **Gaussian** | **step-level**        | `rng.normal(0, std)` 独立采一个新样本                | ✅ 每 step 消耗 |
| **Uniform**  | **step-level**        | `rng.uniform(-std·√3, +std·√3)` 独立采一个新样本     | ✅ 每 step 消耗 |
| **Shift**    | **program-run-level** | 恒加 `+shift_value`（常数，来自 `_noise_shift` 独立字段） | ❌ 不消耗 rng   |


这与 Robust-Gymnasium 完全一致（RG 里 `random.gauss(mu, sigma)` 和 `random.uniform(low, high)` 也是每 step 独立调用，`args.noise_shift` 也是 CLI 常数）。

**关于 v1 Shift 采用 program-run-level 语义的 3 个好处**：

1. **无 RNG 消耗**：shift 分支不动 rng 状态。同 env 若切换 dist=gauss/uniform，gauss/uniform 的采样序列与 shift 分支是否被调用完全无关。这在 debug 时消除一类干扰。
2. **paper 对照清晰**：可以直接引用 "Following Robust-Gymnasium [Gu et al. 2025] §3.2, our v1 shift channel applies a constant `+shift_value` offset each step, matching `args.noise_shift` semantics."
3. **Phase 2 单调扩展**：Phase 2 加 shift 的 episode-level 随机化时，program-run-level 的 v1 语义**保留**（作为 `noise_frequency="constant"` 的选项），episode-level 是**新加**而不是替换。见 §11 表格。

**Reviewer 决策 ✅（2026-06-27）**：

- Gaussian / Uniform = **step-level**（每 step 独立采，与 RG 一致，无需专门决策）
- Shift = **program-run-level** 恒定 `+shift_value`（来自 `_noise_shift` 独立字段，与 RG `args.noise_shift` 完全一致），Phase 2 增加 shift episode-level 变体但不替换。

**0.5 — Obs-noise RNG offset 命名常量，并去重**（`omnipiano/envs/dm_env_obs_noise.py` + 2 个调用点）

代码库当前有 **2 份 `_seed + 31415` 副本**：

- `omnipiano/envs/registration.py:347`（SA env factory）
- `omnipiano/multiagent/registration.py:319`（MA env factory）

这个重复是 bug surface：重构一个位置会悄悄漏掉另一个 → 同一 master seed 下 MA 和 SA env 会得到不同的 obs-noise seed。Phase 0 抽出单个命名常量并更新**两处**调用点都 import 它。

第 1 步 —— 在 `omnipiano/envs/dm_env_obs_noise.py`（module 级）定义常量：

```python
# omnipiano/envs/dm_env_obs_noise.py —— module level
OBS_NOISE_SEED_OFFSET = 20000
"""Offset added to the master env seed to derive the obs-noise RNG seed.

Purpose: ensure the obs-noise RNG stream is independent of the
action-noise RNG stream (which uses gym's self.np_random, seeded
directly from the master seed). Any large positive offset would do;
20000 is chosen to be far from common epoch/seed/step magnitudes so
that grep-ability is easy ("OBS_NOISE_SEED_OFFSET" / "20000" → this file).

SINGLE SOURCE OF TRUTH: both SA (omnipiano/envs/registration.py) and MA
(omnipiano/multiagent/registration.py) env factories MUST import this
constant rather than re-hardcoding the offset. Phase 0 (2026-06-27)
consolidated two prior copies (was `_seed + 31415` in each).

Setting `master_seed=0, OBS_NOISE_SEED_OFFSET=20000` makes obs RNG
deterministic across runs. Changing this offset INVALIDATES all
previously-saved noise traces — bump only with a memo.
"""
```

第 2 步 —— 更新 SA 调用点（`omnipiano/envs/registration.py:347`）：

```python
from omnipiano.envs.dm_env_obs_noise import (
    DmEnvObsNoiseWrapper,
    OBS_NOISE_SEED_OFFSET,
)
...
obs_noise_seed = (_seed + OBS_NOISE_SEED_OFFSET) if _seed is not None else None
```

第 3 步 —— 更新 MA 调用点（`omnipiano/multiagent/registration.py:319`）：

同样的 import + 同样的行替换。一模一样的模式，一模一样的值。

**Reviewer 注**：这是**纯文档化 + 去重** —— 除了 seed 值（31415 → 20000，按你早前要求故意修改）以外**无行为变化**。Equivalence test（§0.7）**有意不**覆盖这个改名，因为我们有意改变了 seed 值；Phase 0 在 commit message 里需注明：Phase 0 之前的 robust noise 轨迹在升级后**不再可复现**。

**Reviewer 待决问题**：是否要写一个单独的 `tests/test_obs_noise_seed_consistency.py`，构造 SA env 和 MA env 在同 master seed 下，断言它们的 `DmEnvObsNoiseWrapper._rng` 首个样本一致？这固化了 SA/MA 一致性这个 invariant。我的建议：**要**，约 20 LOC，维护成本低。

**0.6 — Robust eval 数据记录规格**（`omnipiano/utils/logger_wrapper.py` + `examples/checkpoint_replay_eval.py`）

> 本节由 2026-07-03 的一轮"RG logging 深度调研 + OmniPiano wrapper 数据流逐层核对"重写。核对了 RG 全仓库(结论:RG 极简,不记噪声、不记 clean-vs-perturbed、eval 只报 perturbed reward、无 CVaR/worst-case)与 OmniPiano 的 3 个 wrapper 数据源,确认了下面的"F1 天然 clean"结构性优势。

### 0.6.0 核心事实:我们的主指标 F1 天然免疫噪声(比 RG 更干净)

wrapper 数据流(已代码核对):

- dm_env 层(内→外):`task(physics) → EpisodeStats → MidiEvaluationWrapper → [DmEnvObsNoise] → OAR → ConcatObs → CanonicalSpec → SinglePrec`
- gym 层(内→外):`MetricsWrapper → SafetyWrapper → RobustWrapper → [SafeRecordEpisodeStatistics]`

两个关键位置事实:

1. `**MidiEvaluationWrapper.step` 读的是 `task.piano.activation`(physics 状态),不是 obs dict**(`robopianist/wrappers/evaluation.py:70`)。DmEnvObsNoise 只污染返回的 obs 字典数值,**不碰 physics**。所以 F1 永远从 ground-truth 按键算。(深度核对:`piano.activation` 是**实体 property**,与打包进 obs 的同名 observable 是两个不同定义 —— 详见 §0.6.5(a)。)
2. `**MetricsWrapper`(内层)在 RobustWrapper 加 reward 噪声之前**读 `task.reward_fn.reward_terms`(`metrics_wrapper.py:48`)。所以 reward 分解永远 clean。


| Channel      | physics 执行                                                                                                  | F1 / reward 分解                               |
| ------------ | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| action noise | **a_exec(噪声后)** —— `RobustWrapper` 在 `step` 里把 `action` 重赋值为 `clip(a_cmd+noise)`(robust_wrapper.py:216)再传下去 | F1 从 a_exec 的**实际按键**算 = 加噪轨迹的真实性能(clean 测量) |
| obs noise    | clean action(obs 噪声不碰 action)                                                                               | F1 从**实际 piano state** 算,不是 noised obs       |
| reward noise | clean physics                                                                                               | reward 分解在加噪**之前**算完                         |


**术语澄清**:"clean" 指**从 ground-truth physics 测量,不经噪声通道**,**不是**"不受扰动影响"。action/obs noise 下轨迹确实退化(F1 会掉),但**测量**是干净的。这把 *性能退化*(策略真的弹得更差)和 *测量污染*(reward 读数被加噪)分开了 —— RG 的 perturbed-reward 指标把两者混成一个数,我们不。

**Paper claim(可 cite)**:

> "OmniPiano reports F1 measured from ground-truth key presses, invariant to the observation/reward corruption channels — it captures the policy's *true* task performance under perturbation, separating performance degradation from measurement corruption, a distinction Robust-Gymnasium's perturbed-reward metric collapses."

### 0.6.1 记录范围决策

**只改 eval CSV(两个,schema 锁定要同步)**:

- `<split>_episode_metrics_<id>.csv`(`SafeRecordEpisodeStatistics`,SB3 eval env)
- `eval_episode_metrics_<uuid>.csv`(`examples/checkpoint_replay_eval.py`,OmniSafe post-hoc eval)

**训练 rollout CSV 不动**:OmniSafe `progress.csv` / SB3 native 由各自库原生记录;训练噪声配置由 env_id 固定、可反推,不需要 per-episode 噪声记录。要 hook 它们的 logger 成本高、收益低。

### 0.6.2 新增列

**两类列,价值定位不同**:


| 列                    | 含义                                                        | 定位                                                         |
| -------------------- | --------------------------------------------------------- | ---------------------------------------------------------- |
| `eval_noise_scale`   | 该 eval 实际跑的噪声缩放因子(0=clean, 1=training-level, sweep 用其它值)  | **科学变量 / sweep 命脉** —— 没它无法判断一行属于 robustness 曲线的哪个点。**必留** |
| `ep_noise_action_l2` | episode 内 `info[robust/noise_action_l2]`(clip 前原始噪声 L2)求和 | **开发期 tripwire**(见下)                                       |
| `ep_noise_obs_l2`    | episode 内 `info[robust/noise_obs_l2]` 求和                  | 同上                                                         |
| `ep_noise_reward`    | episode 内 `info[robust/noise_reward]` 求和(§0.3 新增 key,可为负) | 同上                                                         |


**关于后三列的诚实定位(不要误当科学变量)**:

- **不是复现所需**:固定 seed + env_id + `eval_noise_scale` 已完全决定噪声,可重建。
- **信息量低**:高维 Gaussian 的 ‖noise‖ 因 concentration of measure 紧集中在 `std·√dim` 附近,基本是 (config × episode 长度) 的近似常数。
- **记的是 clip 前原始噪声**,不是 clip 后的有效扰动。
- **唯一真价值 = silent-failure tripwire**:`eval_noise_scale` 乘法链路是 Phase 0 新写的承载性代码,最危险的失效是"eval 时噪声悄悄没生效"——F1 会看起来意外地好,无明显症状,整条 robustness 曲线作废。`ep_noise_action_l2 == 0`(当期望非零)是一眼可见的报警。给外部用户跑 sweep 时是一个 per-run 保险。

**保留决策(2026-07-03,用户)**:v1 **保留** 三列——开发/调试阶段抓静默失效非常有用。**待 pipeline 稳定后**可考虑降级(砍 obs/reward、只留 action 做 tripwire,或改为 CI test);届时若砍,需在 §0.6 记一笔并同步两个 CSV schema。

`__init__` 加 accumulator,`reset()` 重置,`step()` 从 info key 累加,episode 结束写入。与现有 `episode_return` accumulator 同模式(`logger_wrapper.py:102-110`)。`checkpoint_replay_eval.py` 的 `_CSV_HEADER` + 写入逻辑同步加同样 4 列。

**建议同时加一个 CI test**(与 tripwire 互补,不替代):断言 `eval_noise_scale=1` 时注入噪声匹配配置、`=0` 时为零——把 eval-noise 接线的正确性在 CI 钉一次。

### 0.6.3 明确**不新增**的列(核对后砍掉)

- **`ep_true_return`(clean reward 单位下的真性能)**:**不加**。原因:
  1. clean total reward = 现有 reward 分解列之和(`energy + key_press + sustain + ot_fingering + forearm`,composite reward 就是各 term 相加),**已可从现有列算出**,无需新列;
  2. 更根本:**reward channel 在 eval(固定策略)下,reward 噪声是 no-op** —— eval 时 `π(obs)→action` 不看 reward(无学习),噪声只污染返回标量、不改 action,所以轨迹与 clean env 逐字节相同,F1、ep_return 都不受影响(ep_return 只是被"事后污染",轨迹不变)。**推论:reward channel 的 robustness 完全是训练期现象**,eval 天然该 clean(eval_noise_scale=0),observed-vs-true 区分无意义。
  > ⚠️ **已被决议 11（2026-07-04）取代**：下段的"reward eval force-zero / 恒 clean / 不需 `ep_true_return`"结论**已作废**。前提"固定策略不消费 reward"在 `action_reward_observation=True`(默认)下不成立（noised reward 经 `obs["reward"]` 进入策略)。现方案:reward 与 action/obs 对称、eval matched，真实 return 用 `ep_return_true`/`ep_return_noised` 两列。详见 §12 决议 11。以下保留原文仅作历史。

  **落地结论(reward 通道,决定 2026-07-04:保留任务 + eval 禁噪)**:eval 时对 reward 加噪是**可证明的退化操作**(固定策略不消费 reward → 轨迹/F1 与 clean 逐字节相同,只污染 `ep_return` 读数),因此 **v1 在 `make(mode="eval")` 里无条件把 reward 噪声三/四字段置零**(代码强制的不变量,见 §0.2 guard),而不是"请记得设 scale=0"的约定 —— 后者有静默失效风险(有人对 reward 任务误跑 sweep → 一堆被污染的假曲线)。
  **评估协议(canonical)**:**reward-trained policy 的 eval 一律在 `OmniPiano-ClairDeLune-Clean-v0` 上跑**。单通道设计下 `R-*` 任务与 `Clean-v0` 仅差 reward 噪声配置,噪声置零后二者 byte-equivalent —— 故"在 force-zeroed 的 R 任务 env 上 eval"与"在 Clean-v0 上 eval"数值完全相同,以 **Clean-v0 为 canonical eval env** 表述最无歧义。**双重保障**:(i) 若有人仍对 reward 任务构造 eval env 并请求非零 eval 噪声(注册带噪 eval 或 sweep override),`make()` 发 `warn` 指路 Clean-v0;(ii) force-zero 兜底,即便 warning 被忽略,eval 也保证干净(不是仅提示,是真禁用)。
  **保证的锚点是 `make(mode="eval")` 这个 env 构造入口,不是任何 logger**。**经此入口构造 eval env** 的驱动器,故全部继承 force-zero:
  - SB3 **`EvalCallback`**(周期性 deterministic eval,`run_sb3_template.py:227/231`)—— 其 `mean_reward` 来自 **Monitor**,写进 `evaluations.npz` / tensorboard / best_model 选择;
  - SB3 **`_final_eval`**(手写 deterministic rollout,`run_sb3_template.py:116-134`,`ep_return` 手动累加);
  - **`SafeRecordEpisodeStatistics`**(旁挂 CSV side-effect);
  - (未来) **`tools/robust_eval_sweep.py`**(§5.2,`make(mode="eval", _eval_noise_scale_override=...)`)。

  一旦 reward 噪声在 make() 里被置零,RobustWrapper 原样透传 reward → **无论上面哪个驱动器、在 RobustWrapper 内层还是外层累加,拿到的都是 true return**。因此 `ep_return`/`mean_reward` 都是真值,**不需要 `ep_true_return` 列、也不需要 logger 做任何 reward 特判**。

  **例外(重要)**:OmniSafe 的 **`examples/checkpoint_replay_eval.py`** 用 `omnisafe.Evaluator().load_saved()` 从**训练 config 重建 env**,**绕过 `make()`**,故**不自动继承** force-zero。该缺口已在 **§11** 记为 deferred(OmniSafe 主场 safety 任务无 reward 噪声,缺口大概率为空;真需要时 reward 任务 eval 换 `Clean-v0`)。

  **未来接入新算法库的契约**:eval env 必须走 `make(mode="eval")`;满足这一条,新库用自己的 eval 循环 / logger 也自动继承该保证(SB3 `EvalCallback` 即先例)。反例即 OmniSafe replay——它绕过 make,所以不在保证范围内。
  **注意范围**:此 force-zero **仅针对 reward 通道**;`eval_noise_scale` 对 **obs/action 通道照常生效**(它们的 sweep 需要 scale>0)。**训练侧 Monitor / OmniSafe native logger 保持加噪现状**(agent 本就该看噪声 reward),训练侧真实演奏水平看 F1(clean-by-construction)。reward 通道的 robustness 因此**不画在 `eval_noise_scale` 轴上**,而是画在**训练噪声档位**轴上(P10/P30/P50 三个训练 env,checkpoint 均在 Clean-v0 上 eval),y=F1 on Clean-v0,对照 Clean-trained baseline —— 详见 §6 Reward 段。
- **per-step 全量 obs / action / noise 向量 trace(原 Tier 3 `noise_trace.npz`)**:**v1 不做**(用户 2026-07-03 确认)。固定 seed 下可重建;需要时再单独设计,不进 v1 scope。
- **a_exec(执行的噪声 action)向量**:**不记**。它的*结果*已被 F1/reward 捕获;向量本身固定 seed 可重建。

### 0.6.4 CVaR / worst-case / drop% —— 分析层,不是 logging 层

这些是 **post-hoc 从 per-episode F1/return 算的**(在 `tools/plot_robustness_curves.py`,Phase 1)。只要每 episode 记了 F1 和 return(现有列已有),分析工具就能算出 CVaR、worst-case-over-seeds、clean-vs-noised drop%。**不需要任何额外 per-step 记录**。RG 连这些都没算(只画 mean-return 曲线的 gap),我们在分析层补上即可。

### 0.6.5 数据源审计 —— 全 CSV 列 × 噪声通道正确性(2026-07-04 代码复核)

本节把"每个 CSV 记录量究竟读物理、读 obs、还是读 action"逐列钉死,给出 training/eval × channel 的正确性矩阵。**核心结论:没有任何 CSV 记录量读被加噪的 obs;唯一会"污染读数"的是 reward 通道对 `ep_return` 的直接加噪,且仅训练侧。**

*(a) F1 / precision / recall / sustain_ —— 读物理实体属性,不是 observable**

piano 里 `activation` 有**两个同名定义,别混**:

- **实体 property** `piano.activation`(`robopianist/models/piano/piano.py:245-246`)→ 直接返回 `self._activation`;后者由 `_update_key_state(physics)` 从 MuJoCo `physics.bind(self.joints).qpos`(或 `.actuators.ctrl`)算出(`piano.py:178-192`),在 `after_substep(physics)` 内更新(`piano.py:158`)。纯物理,**不进 obs**。
- **observable** `activation`(`piano.py:299-306`,`observable.Generic`)→ 打包进 obs 字典、可被 DmEnvObsNoise 加噪。

`MidiEvaluationWrapper.step` 读的是**前者(实体 property)**(`evaluation.py:70`),y_true 来自 MIDI 谱面 `task._notes`(`evaluation.py:117`)。→ F1 的 y_pred/y_true **都不经过 obs 打包/加噪路径**;外加 MidiEval 在链中位于 DmEnvObsNoise 内层(`registration.py:338` vs `:348`),双重保证。

**(b) ep_cost —— 两族约束,一族读物理、一族读执行动作 a_exec,均不读 obs**

`SafetyWrapper` 只累加(`safety_wrapper.py:40`);cost 由各 `constraint.compute_cost(env, action, obs, info)` 算(`omnipiano/safety/constraints.py`):

- **读物理族**:`HandCollisionConstraint`(`:188-194`,physics 接触检测)、`HandCollisionForceConstraint`(`:264-289`,`physics.data.contact`+`mj_contactForce`)、`TotalActuatorPowerConstraint`(`:305-312`,physics actuator power)、`InjuredJointPowerConstraint`(`:375-392`,physics force/vel 传感器)。均经 `_get_dm_internals(env)` → `composer_env.physics`(`:32-33`)。
- **读动作族**:`JointMagnitudeConstraint`(`:49-53`)、`MultiJointShared/SummedMagnitudeConstraint`(`:93-109`/`:150-167`),只用 `action` 向量。`SafetyWrapper` 在 `RobustWrapper` 内层,收到的 `action` 已是 **a_exec(加噪执行动作)**→ 读的是"物理真正执行的动作",诚实。
- **所有约束都 `del obs` 或不引用 obs** → obs 噪声永不污染 cost。**cost 在任何通道下都 clean/诚实**。

**(c) ep_return / reward 分解 —— composite = 各 term 之和,分解列恒 clean**

`CompositeReward.compute` 就是无权重求和 `sum_of_rewards += rew`(`composite_reward.py:51-56`,权重已烘焙进各 reward_fn),`reward_terms` 存的是加噪前物理值,`MetricsWrapper` 内层读取(`metrics_wrapper.py:48`)→ **6 个 reward 分解列之和 == clean episode return**,永远可算(§0.6.3 依此砍掉 `ep_true_return` 列)。`ep_return` 本身则由外层 logger 累加**流经的** reward:reward 通道加噪时它被污染,action/obs 通道不碰 reward 标量故诚实。

**(d) 正确性矩阵**


|                                                                                                              | reward 通道任务                                                                                        | action / obs 通道任务                         |
| ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| training rollout CSV `ep_return`                                                                             | **污染**(真值+噪声,≠该轨迹真 return)                                                                         | **诚实**(准确测被扰动轨迹的真 return)                 |
| eval `ep_return` / `mean_reward`(经 make() 的驱动器:SB3 `EvalCallback`+Monitor、`_final_eval`、SafeRecord CSV) | **matched**(决议 11:reward 噪声按 `eval_noise_scale` 缩放,不再 force-zero;`ep_return` 带噪,真实值见 `ep_return_true` 列 = 分解列之和) | **诚实**(sweep 各 scale 的真值;scale=0 为 clean) |
| `ep_cost` / F1 / 分解列(任意 CSV)                                                                                 | clean                                                                                              | clean(读物理 / a_exec,不读 obs)                |


"污染"= 数值 ≠ 该轨迹的真实 return(reward noise 独有);"诚实"= 准确反映一条被扰动轨迹实际发生了什么(action/obs noise 让策略真的表现更差,但读数不撒谎)。

**(e) 范围界定**:当前设计 **SafeRL 任务族与 Robust 任务族不相交**(`TaskSpec` 技术上允许 `safety_config` + `robust_config` 并存,但 v1 未注册合体任务)。推论:

- 跑 SafeRL 任务时 robust 全关 → cost/return 天然 clean;上面的污染讨论不触及 SafeRL benchmark。
- "reward 通道 `ep_return` 训练侧污染"**仅适用于 robust reward-channel 任务**,§0.6.3 的处理(eval `scale=0` + F1 干净锚 + 文档写明训练侧 `ep_rew_mean` 是加噪信号)专为它们保留。
- cost 因 (b) 在任何任务下都干净,属"双重安全"(SafeRL 无噪声 + robust 也不污染 cost)。
- 若 v2 出现 "safe + robust" 合体任务,cost 仍 clean,但 reward 通道那套 `ep_return` 处理需同时生效。

**0.7 — Equivalence test**（as-built ✅ `omnipiano/tests/test_robust_v1_equivalence.py`，4 passed）

> **as-built**：因 S5 有意改了 obs seed（31415→20000），2 个 v1 env 的 obs 噪声值不再等于 Phase 0 前；gate 改为验证**跨构建可复现**（同 master seed → 同噪声序列）+ 通道正确（action-on/obs-off 与反之）+ gaussian 采样 bit-exact（`RobustConfig.sample_noise` == `rng.normal(0, std)`，证明 14 字段扩展 / dispatch 重构没改 RNG 算术）。

**根据 [feedback_equivalence_test_when_replacing_upstream]**，当我们修改共享行为时必须验证未改动的路径仍然产生 bit-exact 一致的输出。Phase 0 改变了 `RobustConfig` 和 `RobustWrapper` 的公共接口。测试：

```python
def test_robust_v1_noise_sequences_unchanged():
    """Phase 0 changes the RobustConfig surface (adds eval_noise_scale,
    reward_noise_std, noise_dist, and per-distribution independent
    fields) but MUST NOT change the noise sequence for the 2 existing
    robust envs at the same master seed (they use Gaussian by default,
    so their action_noise_std / obs_noise_std stays wired the same way).
    """
    # Path A：重建旧 config（依次设 action_noise_std=0.01, obs_noise_std=0.01）
    # 跑 100 个 action，收集 info["robust/noise_action_l2"] 和
    # info["robust/noise_obs_l2"] 序列。
    # Path B：新 config（同 2 个 std + 新字段默认值）+ 同 seed
    # 跑 100 个 action，收集相同序列。
    # 断言 max abs diff == 0.0。
```

这能捕捉到任何"加了新字段意外改变了 RNG draw 顺序"这类回归。

---

## 4. RNG 设计（澄清）

**Stream 1 —— gym 层（RobustWrapper）**：`self.np_random`（由 `env.reset(seed=master)` 自动 seed）。用于 **action noise** 和 **reward noise**。原因：两者都是标量（action：shape=action.shape，reward：shape=()），且在 step() 中同时读取，共享一条 stream 不会有 timing 意外。Reviewer 请确认这样 OK。

**Stream 2 —— dm_env 层（DmEnvObsNoiseWrapper）**：独立的 `np.random.default_rng(master + OBS_NOISE_SEED_OFFSET)`。**仅**用于 obs noise。独立 stream 的原因：dm_env step 在不同的代码路径中被调用（在 gym adapter 之前），且 per-key 独立性要求自己一致的 draw 序列。

**为什么不用 3 个独立 stream**：需要 2 个 magic offset（obs 用 `master + 20000`，reward 用 `master + 40000` 等等）。每多一条 stream 就多一个 reproducibility 追溯面。2 个 stream 加文档化 offset 是仍然 principled 的最小方案。

**Reviewer 待决问题**：你希望 3 条 stream（一条 per channel）还是接受 2 条 stream（action+reward 共享、obs 单独）？我的建议：**v1 用 2 条 stream**（KISS）。

✅ **已决定（2026-06-27）**：2 条 stream。§5.1.0 确认 v1 每 task 为 single-channel，action 或 reward 分支同一时刻仅有一个激活 → 共享 stream 语义上完全等价于独立 stream。见 §12 Q3 决议。

**关于 shift 分支不消耗 RNG**：v1 shift 是 deterministic constant offset（见 §0.4），**不 draw rng**。所以在混合分布 sweep（如某个 env 用 shift、后续同 env 换 gauss）时，shift 分支的存在与否**不改变** stream 里 gauss/uniform 的采样顺序。这是意外的好副作用。

---

## 4.5 各 channel 中 std 的物理单位（重要 caveat）

`RobustConfig.*_noise_std` 的**数值**在 3 个 channel 上代表**不同的物理量**。这是 v1 承认但**不修**的设计瑕疵 —— Robust-Gymnasium 有完全相同的问题、他们也没修 —— 保持一致以便直接对照。

### 各 channel 单位


| Channel    | 噪声注入点看到的量                                                                                                                   | `std` 的物理单位            | 示例：`std=0.05` 的含义                                                                               |
| ---------- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------- |
| **action** | 由 `CanonicalSpecWrapper` 归一化到 canonical `[-1, 1]` 的 flat ndarray                                                            | 归一化 action 单位          | full range 2.0 的 2.5%（[-0.05, +0.05] 附近）                                                        |
| **obs**    | dm_env Dict，各 key 保持物理单位（`joints_pos` 是 radians；`piano/state` 已由 `_get_normalized_state` 归一化到 [0,1]；`joints_vel` 是 rad/s；等） | 物理单位混合（**per-key 不同**） | 关节 `joints_pos` 上是 0.05 rad ≈ 2.86°；`piano/state` 上是 [0,1] 的 5%；这两者在同一次 std=0.05 应用下**物理意义不可比** |
| **reward** | composite reward 的 scalar float，per-step 范围约 [-0.05, +3.5]                                                                  | 原生 reward 单位           | 加 0.05 → 相对 per-step reward 大约 1.4%（小）                                                          |


### 为什么不在 v1 修

- Robust-Gymnasium 完全同样的问题（`args.noise_shift = 0.005` 在 action / obs / reward 上单位不同、混用）。paper 对照时如果我们修了会造成"不能直接对齐"。
- 修法是 per-channel scaling factor 或 per-key std（Phase 1.5+，见 §11"per-dim 噪声 std" 那行的延期理由）。
- v1 的目的是先建立 pipeline，然后再优化 magnitude 语义。

### Paper 写作中如何 disclose

在 methods section 加一段类似：

> "Noise magnitudes (`*_noise_std`) are numerically consistent with the Gaussian σ across distributions on a common x-axis, but their physical meaning varies per channel: action noise is in canonical [-1, 1] action units, observation noise is in mixed physical units per key (radians for joint positions, [0, 1] for normalized piano key states), and reward noise is in raw composite-reward units. Cross-channel std values should therefore be interpreted relative to each channel's natural range, not as directly comparable magnitudes. This convention follows Robust-Gymnasium [Gu et al. 2025]."

### 实操建议：跨 channel 比较时怎么办

Paper 的 robustness curve **不跨 channel 比较绝对 std 值**：

- 每个 channel **独立**画曲线（三通道均用 x = eval_noise_scale；决议 11 后 reward 与 action/obs 对称），y = F1
- **不**画"同一 std 下 3 个 channel 的对比条形图" —— 这个图会误导读者以为 std 可比
- 若要跨 channel 定性对比"哪个 channel 更 fragile"，用 **"F1 掉 50% 时的 std 值"** 作为 metric，这样每个 channel 各自的 std scale 隐去了

---

## 4.6 边界处理与 clip 语义（与 RG 完全对齐）

`RobustWrapper` 和 `DmEnvObsNoiseWrapper` 对 3 个 channel 的 clip 行为**明确不同**，且**与 Robust-Gymnasium 完全一致**。此小节记录设计决策，避免 reviewer 追问。

### 三 channel clip 行为对比


| Channel    | OmniPiano v1                                                                                   | RG                                                     | 一致？                            |
| ---------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------ | ------------------------------ |
| **Action** | ✅ **显式 `np.clip(action+noise, low, high)`** clip 到 canonical `[-1, 1]`（`robust_wrapper.py:52`） | ❌ 不显式 clip，但 MuJoCo `ctrlrange="-1 1"` 内部 clamp        | ✅ **净效果一致**（都 bound 到 [-1, 1]） |
| **Obs**    | ❌ **不 clip**（`dm_env_obs_noise.py` grep clip 返回空）                                              | ❌ 不 clip（`observation = self._get_obs() + noise` 直接赋值） | ✅ 一致                           |
| **Reward** | ❌ **不 clip**（Phase 0 计划：`reward = float(reward) + float(reward_noise)`）                        | ❌ 不 clip（`reward = reward + args.noise_shift` 直接赋值）    | ✅ 一致                           |


**代码证据（RG）**：`ant_v4.py` 全文只有 1 处 `np.clip`（line 115，Ant 自己的 contact_forces 归一化，与 noise 无关）；obs/reward noise 3 处赋值全部无 clip。

### 每 channel 设计动机

**Action 会 clip**：

- `action_space` 是 `Box(-1, +1)`（`CanonicalSpecWrapper` 保证），clip 到边界内是**类型契约**
- 未 clip 的话，`action + noise` 可能超 [-1, 1]，MuJoCo 内部仍会 clamp（结果一样），但**其它 wrapper 可能读到 out-of-range 值**误处理
- RG 完全依赖 MuJoCo 内部 clamp，语义等同但实现路径不同
- **代价**：`std=0.15` 时，边界附近 action 约 20% 触发 clip → effective std 略低于 nominal（~10-15%）。Sweep 值 `{0.05, 0.10, 0.15}` 全部处于 "clip 影响可接受" 区间

**Obs 不 clip**：

- Obs 是"传感器观察值"语义，扰动后可能超出物理 range（如关节角超出 joint_limit + 0.1 rad）是**"传感器读错"的正确建模**
- Clip 会让 nominal std ≠ empirical std（clip 后分布不再高斯 / 均匀）
- RG 不 clip 保持 std 忠实性
- **代价**：极小（Shadow Hand joint 通常 range 约 ±1 rad，std=0.15 时超出 range 概率 <1%）

**Reward 不 clip**：

- Reward 无严格 upper/lower bound（可为大负数、大正数）
- Clip 会失真噪声分布
- RG 不 clip 保持 std 忠实性

### 对 sweep 值选择的影响（`{0.05, 0.10, 0.15}` 是 clip-friendly 的）

Action 端 clip 触发率估算（假设 policy 输出 uniform 分布在 [-0.7, +0.7]）：


| std  | 3σ 覆盖 | 边界附近 clip 触发率      | 有效 std               |
| ---- | ----- | ------------------ | -------------------- |
| 0.05 | ±0.15 | ~5%                | ≈ 0.049              |
| 0.10 | ±0.30 | ~15%               | ≈ 0.091              |
| 0.15 | ±0.45 | **~25%**           | ≈ 0.128              |
| 0.20 | ±0.60 | ~40%（如 sweep 到这个值） | ≈ 0.15（明显偏离 nominal） |
| 0.30 | ±0.90 | >70%               | 严重失真                 |


→ 我们选 `{0.05, 0.10, 0.15}` 保持在"clip 影响 <25%"，nominal std 与 effective std 的偏差 <15%，曲线含义清晰。如果 sweep 到 0.20+，需在 paper 里 flag effective std 与 nominal 的显著偏离。

Obs channel 因不 clip，nominal ≈ empirical std 严格成立。Reward 同理。

### Paper 写作 note

Paper 的 Methods 里可以简洁描述：

> "Action noise is clipped to the canonical [-1, 1] action range, matching Robust-Gymnasium's [Gu et al. 2025] implicit clamping via MuJoCo's `ctrlrange`. Observation and reward noise are unclipped, preserving the nominal-vs-empirical std correspondence. Sweep values were chosen (§Y) to keep action-clip triggering below 25%, preserving the nominal std interpretation for the action channel."

**关键 claim 可复用**："clip semantics identical to Robust-Gymnasium [Gu et al. 2025]" —— 3 个 channel 都成立。

---

## 4.7 obs["action"] / obs["reward"] 的 principled semantics 修正（方案 6）

### ✅ Step 1 Prototype 已完成 —— 15/15 gate 全绿（2026-07-03）


| 项            | 状态                                                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 分支           | `feat/robust-method6-prototype`（main 未动）                                                                                                   |
| 实现 commit    | `f8510b1`（RobustWrapper override + A1-A6）+ `f7b8a5b`（扩展 gates G1-G6, G11）                                                                  |
| 测试           | **15/15 全部通过**（A1-A6 一次通过；G1-G6+G11 一次通过），`omnipiano/tests/test_robust_v1_method6.py`                                                      |
| 生产 env smoke | `FantaisieImpromptu-ActionRobust-v0`（override bit-exact ✓）、`ClairDeLune-ObservationRobust-v0`、`ClairDeLune-CollisionSafe-v0` 全部正常构造 + step |
| 运行环境         | pianist conda env（pytest 9.0.2, numpy 2.2.6, MUJOCO_GL=egl），repo root 运行                                                                   |
| 范围           | 仅 action channel（当前 2 字段 RobustConfig, gaussian）；**reward slot override 留待 Phase 0 §0.3**（需要 `reward_noise_std` 字段先存在）                     |
| 下一步          | 等 reviewer sign-off → merge 回 main → 开 Phase 0                                                                                             |


**As-built 实现要点（比原 pseudocode 更强的三个决策）**：

1. **不重写映射公式** —— `_clean_physical` 直接 `from dm_env_wrappers._src.canonical_spec import _scale_nested_action`，配上从 chain 里 `CanonicalSpecWrapper` 缓存的**原 spec 对象**和 clip flag，输入侧只复刻 adapter 的 `np.asarray(action, dtype=np.float32)` 转换（已验证 `SinglePrecisionWrapper.step` 不动 action）。两条路径跑的是**同一段 upstream 代码**，bit-exact 是构造性保证，不是碰运气。这就是 A1/A5/A6 一次全绿的原因，也顺带消解了原设计里的 "Issue C"（a_cmd 预 clip）—— clip 在共享函数内部用同一 flag 处理，无需单独预 clip（G3 用 |a|>1.7 的越界 action 验证过）。
2. **Override 只在 action noise 实际激活时运行**（而非无条件覆写）—— 零噪声 env 返回**内层 obs 同一个对象**（A4 用对象 identity 断言，连 copy 都没发生），"对现有 baseline 零影响"是**结构性**保证而非数值巧合。副作用：A6 因此成为真正的 cross-implementation 对比（clean env 走 CanonicalSpec 原生算术，noised env 走 override 算术）。
3. **缓存 rebuild-safe** —— `reset(seed=X)` 会重建整条 dm_env chain，但 slice 布局 / physical spec 值 / clip flag 都由静态 env 配置决定，`__init__` 缓存的值对重建后的 chain 仍有效（G2 用两次不同 seed 的 rebuild 验证）。对比：`last_step_noise_l2` 依附于 wrapper **实例**，必须每次 re-walk（既有逻辑，未动）。

**重要事实澄清**：本方案**未改动 `ObservationActionRewardWrapper` 的任何一行代码**（upstream 原样）。全部改动集中在 `RobustWrapper` 内部（slice 计算 + 一次对 upstream 原函数的调用），风险面是"包含的"而非"扩散的"。

**决议时间**：2026-06-27（由用户完成深度研究后确认）；prototype 完成 2026-07-03。

**核心问题**：当 `action_reward_observation=True`（OmniPiano 默认，全部 registered env 都是），`ObservationActionRewardWrapper` 记录到 `obs["action"]` 和 `obs["reward"]` slot 里的**默认值不是 principled robust semantics 想要的**：

- `**obs["action"]`** 记录的是 **noised executed action** 的 physical 版本，泄露 noise 信息给 policy（policy 能反推出 noise）
- `**obs["reward"]`** 记录的是 **clean raw reward**（无 noise），而 policy 训练用的 return reward 是 noised —— 两处不一致

参考文献威胁模型（Wang et al. 2020 noisy-reward RL；Gu et al. 2025 Disrupted-MDP）明确要求：

- `obs["action"]` 应记录**clean commanded** (a_cmd)
- `obs["reward"]` 应记录**noised observed** (r_obs)，与 policy 训练信号一致

### 关键代码事实（已通过代码验证，2026-06-27）

**Chain 构造顺序**（`registration.py:348-380`）：

```
task
→ EpisodeStats → MidiEval → [DmEnvObsNoise]
→ ObservationActionRewardWrapper  ← INNER (记录 obs["action"] / obs["reward"])
→ ConcatObs                        ← 用 tree.flatten 字母序拼扁
→ [FrameStack if frame_stack > 1]  ← v1 默认 fs=1，不插入
→ CanonicalSpecWrapper(clip=True)   ← OUTER (canonical → physical 转换点)
→ SinglePrecision
```

**CanonicalSpecWrapper._scale_action**（`canonical_spec.py:74-88`）：

```python
def _scale_action(action, spec, clip):
    scale = spec.maximum - spec.minimum
    offset = spec.minimum
    if clip:
        action = np.clip(action, -1.0, 1.0)
    action = 0.5 * (action + 1.0)   # canonical [-1,1] → [0,1]
    action *= scale                  # → [0, scale]
    action += offset                 # → [minimum, maximum] = physical
    return action
```

**ObservationActionRewardWrapper.step**（`observation_action_reward.py:39-53`）：

```python
def step(self, action):
    timestep = self._environment.step(action)
    return self._augment_observation(action, timestep.reward, timestep)
    # ↑ 存 obs["action"] = 它收到的 action（因为在 CanonicalSpec 内层，
    #   收到的已经是 physical）
```

**tree.flatten 字母序**（实测确认）：`{'reward': 'r', 'action': 'a', 'joints_pos': 'j', ...}` 展平为 `['a', 'g', 'j', 'p', 'r']`。

### 数据流详细追踪（含方案 6 override）

```
policy 输出 a_cmd (canonical, e.g., 0.5)
     ↓ 传给 RobustWrapper.step
┌── RobustWrapper.step(action)  [gym 层] ─────────────────────────
│ PRE 阶段（仅 action_noise_std > 0 时执行）：
│   1. 保存 clean 命令: a_cmd = np.asarray(action).copy()   ← 局部变量
│   2. 加 action noise:
│        a_exec = np.clip(a_cmd + noise, low, high)
│              = 0.54  (若 noise=+0.04, canonical)
│
│   调用 self.env.step(a_exec)  ← 传下去的是 canonical 0.54
└─────────────────────────────────────────────────────────────────
     ↓ 向下经过 gym→dm_env
CanonicalSpec.step(0.54)
    → _scale_nested_action(0.54, action_spec, clip)     ← ★ 就是这个 upstream 函数
    → 0.377 rad (physical, range=[-0.698,+0.698])
    ↓
FrameStack(fs=1, skip) → ConcatObs → ObservationActionReward.step(0.377)
    → 记录 obs["action"] = 0.377   ← a_exec 的 physical 值，泄露了 noise!
    ↓
DmEnvObsNoise → MidiEval → task.step(0.377)
    → r_true (clean reward)
     ↓ 返回上行
obs["action"] = 0.377 (noised-derived)，obs["reward"] = r_true (clean)
     ↓ 上行到 gym 层
┌── RobustWrapper 回到 self.env.step 之后 ────────────────────────
│ POST 阶段（方案 6，Option B）：
│
│  [Phase 0，尚未接线] 3. 加 reward noise:
│        r_obs = r_true + reward_noise
│
│  [as-built] 4. Override obs["action"] slot（仅当 a_cmd 非空）：
│        —— 复用 upstream 函数，**不重写映射公式**：
│        a_phys = self._clean_physical(a_cmd)
│          ├ a = np.asarray(a_cmd, np.float32)      # 复刻 adapter 的 cast
│          └ _scale_nested_action(a, self._physical_action_spec,
│                                     self._canonical_clip)
│                ↑ dm_env_wrappers 自己的函数 + 从 CanonicalSpecWrapper
│                  缓存的**原 spec 对象** / clip flag
│                ↑ clip 在函数内部用同一 flag 处理 → 无需单独预 clip
│        obs = obs.copy()
│        obs[self._action_slice] = a_phys.astype(obs.dtype)
│           // 0.377 (noised leaked) → 0.349 (clean a_cmd) ✓
│           // 与 executed path 跑同一段代码 → bit-exact（非 1e-6 近似）
│
│  [Phase 0，尚未接线] 4'. Override obs["reward"] slot（复用同一 _reward_slice）：
│        obs[self._reward_slice] = r_obs
│           // r_true (clean) → r_obs (noised) ✓
│
│  5. return obs, reward, ...
│     ← 当前 prototype：reward = r_true（reward noise 未接线）
│     ← Phase 0 后：reward = r_obs
└─────────────────────────────────────────────────────────────────
     ↓
policy 下一步看到 obs：含 clean physical a_cmd（as-built）
                       ；Phase 0 后另含 noised r_obs
[Phase 0] policy 训练 signal 用 r_obs（与 obs["reward"] 一致）

注：non-robust env 下 a_cmd 为 None → 整个 override 分支不执行，
    obs 对象原样返回（连 .copy() 都不发生）→ 现有 baseline 结构性零影响。
```

### 为什么用 Option B（physical 版本的 clean a_cmd）而不是 Option A（canonical 版本）

因为 RoboPianist 传统上 `obs["action"]` slot 存 **physical 值**（continuing from CanonicalSpec 转换后 ObservationActionReward 记录的值）。如果我们改成存 canonical，会让所有现有 baseline（PPO / PPOLag / SAC 已训练好的 checkpoint）在 obs["action"] slot 上看到不同的数值分布。

Option B 的关键性质：**对 non-robust env bit-exact no-op**。

- Non-robust env 下 `a_cmd = a_exec` (no noise added)
- `CanonicalSpec.map(a_cmd) = CanonicalSpec.map(a_exec)` 同一公式相同输入
- 覆盖前后 obs 数值完全相同 → 现有 baseline 不受影响

### 我的完整理解（供 reviewer 验证）

1. `**obs["action"]` slot 当前存 physical 值** —— 这是 RoboPianist upstream 的既有行为，我们继承下来的。既有 20+ baseline 实验都是这样跑的
2. **CanonicalSpec 的 canonical→physical 映射是线性的**（`0.5*(a+1)*scale + offset`），完全 deterministic
3. **RobustWrapper 在 `__init_`_ 时缓存 dm_env 层 physical bounds**（走 wrapper unwrap 拿 action_spec），运行时只做数学运算，无 wrapper 遍历开销
4. **Slice 计算按字母序**（`tree.flatten` 行为），dm_env observation_spec Dict keys 是稳定的
5. **frame_stack=1 是 v1 唯一 supported case**，v2+ 需要额外设计

### Prototype gate 假设 —— 验证结果（2026-07-03，全部通过 ✅）


| #          | 假设                                                                           | 验证方法（as-built）                                                                                                                                          | 结果                                        |
| ---------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| A1         | ObservationActionReward 存的确实是 physical 值，且我们的 `_clean_physical` bit-exact 复现 | clean env 上 step 固定 action，`np.testing.assert_array_equal(obs[action_slice], _clean_physical(a))`；另断言 slot ≠ raw canonical（证明 rescale 真的发生了）            | ✅ **bit-exact**（比原计划的 1e-6 更强）            |
| A2         | tree.flatten 字母序稳定                                                           | 构造乱序 Dict，断言展平顺序 = 字母序                                                                                                                                  | ✅                                         |
| A3         | dm_env 层 physical action_spec 可拿到且与 live chain 一致                            | 缓存的 spec minimum/maximum 与 chain 里 `CanonicalSpecWrapper._action_spec` 逐元素相等；clip flag 相等；slice 宽度 = action_dim / 1                                     | ✅                                         |
| A4         | 对现有 env（noise=0）obs 不变                                                       | **升级为对象 identity 断言**：monkeypatch 内层 step 捕获 obs 对象，断言 RobustWrapper 返回的 `obs is inner_obs`（连 copy 都没发生 → 结构性 no-op，不依赖公式正确性）                           | ✅ **强于原计划**（identity ⊃ bit-exact）         |
| A5         | Robust env（noise>0）obs["action"] 变为 clean physical                           | 固定 action，断言 `obs[action_slice] == physical(a_cmd)` bit-exact + `info[noise_action_l2] > 0`                                                             | ✅ **bit-exact**                           |
| **A6** ★★★ | `**obs[action_slice]` 与 noise 强度无关**（cross-condition 一致性）                    | 同 100 个 fixed float64 action 依次喂 clean env 和 noised env（同 seed reset），逐 step 断言 action slot `np.array_equal`；同时统计其他 slot 分岔步数（断言 >10，证明噪声真的作用于 physics） | ✅ **100/100 step bit-exact**，其他 slot 正常分岔 |


**A6 是最强的 gate**：直接验证 override 路径的算术与 CanonicalSpec 原生算术是 bit-exact 双胞胎。一次通过的根本原因是 as-built 决策 1（复用 upstream 原函数 + 原 spec 对象，两条路径跑同一段代码）。

### 扩展 gate G1-G6, G11 —— 验证结果（2026-07-03，全部通过 ✅）

针对 reviewer 关心的"各场景下 ObservationActionRewardWrapper 配合是否出错"，追加 7 个 gate：


| #   | 覆盖的潜在 bug                                                                                                                                                                 | 结果                |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| G1  | **episode 边界**：mid-episode `reset()`（无 seed，不重建 chain）后 OAR slot 归零、继续 step，invariant 是否仍成立。断言设计为 state-independent（slot 只依赖 a_cmd），即使两 env physics 已分岔也必须成立              | ✅                 |
| G2  | **seed rebuild 缓存失效**：`reset(seed=X)` 重建整条 dm_env chain 后，`__init_`_ 缓存的 slice/spec 是否 stale（两次不同 seed 各验 5 步）                                                            | ✅ 缓存 rebuild-safe |
| G3  | **越界 caller action**（|a|≤1.7，模拟未 tanh 的 Gaussian actor）：clip 在两条路径共享的同一函数内处理，slot 仍需 bit-exact                                                                            | ✅                 |
| G4  | **float32 caller dtype**（SB3 风格；float64 OmniSafe 风格已被 A5/A6 覆盖）                                                                                                           | ✅                 |
| G5  | **SB3 DummyVecEnv**：vec 层 obs copy/stack 不破坏 override，info key 完整                                                                                                         | ✅                 |
| G6  | **action 噪声下 reward slot 一致性**：`obs[reward_slice][0] == np.float32(returned_reward)` —— 钉死"action-channel task 不产生 reward 不一致，reward override 只有 reward-channel task 才需要" | ✅                 |
| G11 | action_slice 与 reward_slice 不重叠                                                                                                                                           | ✅                 |


### 全场景覆盖矩阵（channel × 频率 的正确性论证）


| 场景                                  | obs["action"] slot                      | obs["reward"] slot                                                              | 保证方式                                               |
| ----------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------- |
| 无噪声                                 | physical(a_cmd)（a_exec=a_cmd，slot 天然正确） | r_true = r_obs                                                                  | **A4 结构性 identity**（override 分支不进入，连 copy 都没有）     |
| Obs channel                         | 同上（action 未被扰动）                         | r_true = r_obs                                                                  | override 不触发且无需触发；production smoke 验证              |
| Action gauss（step 级）                | **override → physical(a_cmd)**          | r_true = r_obs（action 噪声不产生 reward 不一致，policy 训练用的就是 noised physics 产生的 r_true） | A5/A6 + G1-G5；reward slot 由 G6 钉死                  |
| Action uniform / shift（Phase 0 才存在） | 同一条代码路径（a_cmd 在采样**前**保存，与分布/频率无关）      | 同上                                                                              | Phase 0 加 parametrized 测试复用 A5/A6 断言               |
| Reward channel（Phase 0 才存在）         | 无 action 噪声 → slot 天然正确                 | r_true ≠ r_obs → **需要 reward slot override**（Phase 0 §0.3 唯一新增逻辑）               | Phase 0 gate：`obs[reward_slice] == float32(r_obs)` |


关键结构性事实：v1 **single-channel per task**（§5.1.0）→ 每个 env 最多一个 slot 需要修，由激活的 channel 唯一决定。

### 替代架构评估（已考虑并否决，记录设计依据）

**Alt-2：在 OAR 之下插 dm_env 层 `DmEnvActionNoiseWrapper`** —— "从零设计"时的更优解（OAR 在噪声注入点之上，天然记录 a_cmd，零 override）。对当前 codebase 有三个致命伤，故否决：

1. 噪声会注入在 **physical 空间**（OAR 之下已过 CanonicalSpec）。单一 σ 对每个 actuator 物理意义不同（forearm 是米、关节是弧度），要维持 §4.5 的 canonical 噪声语义（RG 对齐）必须 per-dim rescale σ —— 把公式复刻的风险从"logged slot"转移进**实际注入的噪声本身**（错了 override 只错一个 log slot 且测试立刻抓到；错了 Alt-2 会错实际训练噪声）
2. 打破现有 2 个 registered robust env 的噪声序列 equivalence（§0.7 要求）
3. 需要 dm_env 层新 RNG stream + 新 seed offset 管理

**Alt-1（移 OAR 到 gym 层）**已在下方 "为什么用 Option B" 一节否决（坐标系 + 布局 + checkpoint 兼容性三重破坏）。

**结论**：override 方案的风险是"包含的"（slice 计算 + 一次 upstream 函数调用，均被直接 gate），替代方案的风险是"扩散的"。维持方案 6。

### 完整测试清单（as-built，`omnipiano/tests/test_robust_v1_method6.py`）

> 原计划稿的 8-test pseudocode 已被实际实现取代（commit `f8510b1` + `f7b8a5b`）。
> **实际 15 个测试，2026-07-03 全部通过**。测试代码本身即规范 —— 本节只列清单，
> 细节直接读测试文件（每个 test 有完整 docstring 说明其覆盖的失效模式）。


| Test                                                  | 类型        | 断言强度                                            |
| ----------------------------------------------------- | --------- | ----------------------------------------------- |
| `test_A1_action_slot_is_physical_and_formula_matches` | gate      | bit-exact（`assert_array_equal`）+ 非退化 sanity     |
| `test_A2_tree_flatten_alphabetical_order`             | gate      | 精确顺序                                            |
| `test_A3_layout_and_physical_spec`                    | gate      | spec 逐元素相等 + clip flag + slice 宽度               |
| `test_A4_non_robust_obs_object_untouched`             | gate      | **对象 identity**（`obs is inner_obs`）             |
| `test_A5_action_slot_clean_physical_under_noise`      | gate      | bit-exact + noise l2 > 0                        |
| `test_A6_action_slot_invariant_across_noise_levels` ★ | gate      | 100 step bit-exact + 其他 slot 分岔 sanity          |
| `test_frame_stack_gt1_raises`                         | fail-fast | `NotImplementedError`                           |
| `test_no_action_reward_obs_skips_override`            | fallback  | slices None + noise 仍生效                         |
| `test_G1_invariant_across_episode_reset`              | 扩展 gate   | 跨 reset()（无 seed）60 step bit-exact              |
| `test_G2_invariant_after_seed_rebuild`                | 扩展 gate   | 两次 seed rebuild 后缓存仍有效                          |
| `test_G3_out_of_range_caller_actions`                 | 扩展 gate   | |a|≤1.7 越界输入 bit-exact                          |
| `test_G4_float32_caller_dtype`                        | 扩展 gate   | SB3 风格 float32 caller                           |
| `test_G5_sb3_dummy_vecenv_smoke`                      | 扩展 gate   | DummyVecEnv 2×5 step + info key                 |
| `test_G6_reward_slot_consistent_under_action_noise`   | 扩展 gate   | `obs[reward_slice] == float32(returned_reward)` |
| `test_G11_slices_disjoint`                            | 防御        | slice 区间不重叠                                     |


**测试基建**：4 个 test-only env 在测试文件里 `register()`（ForElise 底座，最短曲）：
`OmniPianoTest-M6-{Clean, ActionNoise, NoAR, FrameStack}-v0`。module-scoped fixture 复用
env 实例，每个 test 自行 `reset(seed=...)` 隔离。

**Phase 0 需追加的测试**（当对应代码路径存在后）：

- dist parametrize：uniform / shift 复用 A5/A6 断言（同一 a_cmd-save/override 代码路径）
- reward-channel gate：`obs[reward_slice] == float32(r_obs)` 每 step
- eval_noise_scale=0 env 与 Clean env 全 obs bit-exact
- §0.7 noise-sequence equivalence（2 个 legacy robust env）

### Implementation strategy（prototype-first）

**Step 1: Prototype** ✅ **已完成（2026-07-03）**

- 分支：`feat/robust-method6-prototype`（commits `f8510b1` + `f7b8a5b`）
- 基于当前 codebase（未改 RobustConfig）
- `RobustWrapper.__init_`_：`_compute_override_layout()`（slice 计算 + physical spec/clip 缓存 + frame_stack/ConcatObs 守卫）
- `RobustWrapper.step`：a_cmd 采样前保存 + POST 阶段 override（仅 action noise 激活时）
- `_clean_physical()`：复用 upstream `_scale_nested_action` + 缓存 spec（bit-exact 关键决策）
- **15/15 测试通过**（A1-A6 + fail-fast/fallback + G1-G6/G11）+ 3 个生产 env smoke
- 待 reviewer sign-off 后 merge 回 main

**Step 2: 完整 Phase 0**（待启动）

- 分支：`feat/robust-phase0`
- 按 §0.1-0.7 逐项做（14 字段 RobustConfig / eval_noise_scale / reward noise / OBS_NOISE_SEED_OFFSET / CSV logging）
- 方案 6 逻辑已在 mainline（Step 1 merge 后），§0.3 只需：(a) reward slot override（`obs[reward_slice] = float32(r_obs)`，仅 reward channel 激活时）；(b) `_channel_active` / `_sample_noise` 适配 14 字段 config —— a_cmd-save/override 机制不变
- 跑 §"完整测试清单" 的 Phase 0 追加项（dist parametrize / reward gate / eval_noise_scale / §0.7 equivalence）
- 全绿才 merge

**Step 3: Phase 1（如果 Phase 0 全绿）**

- Task 注册 + eval sweep + baseline 实验

### 与 Phase 0 其他 items 的依赖关系

方案 6 已独立完成 prototype（无依赖）。Phase 0 §0.3 加 reward noise 时**同时**加 reward slot override —— 复用已验证的 slice 机制（`_reward_slice` 在 prototype 里已计算并被 A3/G6/G11 测试，只是 override 尚未接线，因为 `reward_noise_std` 字段还不存在）。

### Paper 写作可 cite 的说明

> "OmniPiano's `RobustWrapper` overrides the `action` and `reward` slots of the observation-augmented flat obs vector to expose the clean commanded action `a_cmd` (in the physical actuator range, matching RoboPianist's upstream convention) and the noised observed reward `r̃_t` to the policy, aligning with the Disrupted-MDP formalism (Robust-Gymnasium, Gu et al. 2025) and noisy-reward RL convention (Wang et al. 2020). Rather than reimplementing the canonical→physical rescaling, the wrapper invokes dm_env_wrappers' own `_scale_nested_action` with the spec object cached from the chain's `CanonicalSpecWrapper`, so the override and the executed path share one arithmetic implementation — verified bit-exact by a cross-condition gate test (identical fixed-action sequences fed to clean and noised environments yield bit-identical action slots over 100 steps). For non-robust environments the override branch never executes, leaving the observation object untouched."

---

## 5. Phase 1 —— task 注册 + eval harness

Phase 1 在 Phase 0 完成 + equivalence test 通过之后开始。

### 5.1 Task 注册矩阵

#### 5.1.0 设计原则：single-channel per task（与 RG 完全对齐）

v1 所有注册的 robust task 都是 **single-channel** —— 每个 env id 只扰动 3 个 channel 中的 1 个（action / obs / reward），其他 channel 的 `*_noise_std` 严格设为 0。

**这与 Robust-Gymnasium (RG) 的运行时语义完全对齐**（2026-06-27 audit 确认，见下）：

- RG 用**单字符串** `--noise-factor`（`robust_setting.py:22`）在 `{state, action, reward, cost, robust_force, robust_shape}` 中**选一个**
- RG 的 85 个 env 文件里所有 **320 处 `noise_factor` 引用都是 `==` 等值检查**（无 `!=`、无 `in [...]`、无组合分支）
- RG **无**独立的 `--action-noise-sigma` / `--obs-noise-sigma` / `--reward-noise-sigma` 参数 —— 只有一个全局 `--noise-sigma` 应用于当前选中的那一个 channel
- RG **无** composite / multi-channel task id（无 `RobustAntActionObs-`* 类），所有 task 命名都是 `RobustAnt-v4`、`SafetyAnt-v4` 等
- RG 的 docs（`disruptors/disruptor_overview.rst`）把 perturbation "sources" 描述为独立行，从未把 "multi-source simultaneously" 作为 feature

**为什么 OmniPiano v1 也采用 single-channel per task**：

1. **直接可比性**：每个 v1 task ↔ 一个 RG `(env_id, --noise-factor=X)` 运行时配置。paper 里可以直接引用"our 27 tasks correspond to the (env × channel × dist × level) grid of Robust-Gymnasium's runtime configurations, projected as first-class registered environments"。
2. **消融清晰**：paper reviewer 一定会问"哪个 channel 最脆？" —— single-channel task 直接给答案。multi-channel 混合注入把这个问题变成回归分析。
3. **基础设施仍支持多 channel**：`RobustConfig` 有 3 个独立的 `*_noise_std` field，未来 v2+ 若要注册 combined-channel task（如 `OmniPiano-ClairDeLune-AO-Gauss-P05-v0`，同时扰 Action + Obs），只需设两个非零 std 即可。**API 不封死，只是 v1 registration 保持 single-channel 惯例**。
4. **RNG 简化**（见 §4）：v1 每个 task 只有 1 个 channel 活跃 → action+reward 共享一条 RNG stream 完全无风险，因为 action 或 reward 分支永远只有一个被触发。

按照现有 `omnipiano/envs/__init__.py` 风格：

```python
# === Robust Task v1 —— ClairDeLune (bimanual SA) ===
# 1 曲 × 3 channel × 3 distributions × 3 noise level = 27 个 task id (+ 1 clean = 28)。
# 命名模式：OmniPiano-ClairDeLune-<C>-<Dist>-P<XX>-v0
#   <C>       ∈ {A, O, R}                  Action / Observation / Reward
#   <Dist>    ∈ {Gauss, Uniform, Shift}
#   P<XX>     level_value × 100，如 P05 = 0.05
# 示例：
#   OmniPiano-ClairDeLune-A-Gauss-P01-v0    (Action Gaussian σ=0.01)
#   OmniPiano-ClairDeLune-O-Uniform-P05-v0  (Obs Uniform bounds [-0.05, +0.05])
#   OmniPiano-ClairDeLune-R-Shift-P10-v0    (Reward Shift constant=0.10)

def _make_cfg_for_task(channel_lower: str, dist_key: str, level_value: float):
    """Given (channel, dist, level), build a single-channel RobustConfig
    with only the fields for `dist` set. eval_noise_scale defaults to 1.0
    (matched eval, all three channels INCLUDING reward — decision 11; reward
    is NOT force-zeroed, its true return is a separate CSV column).

    Each distribution's parameter is set to `level_value` DIRECTLY
    (no cross-distribution normalization — Option 4a decision):
      - Gaussian: σ = level_value
      - Uniform:  symmetric bounds `low = -level_value, high = +level_value`
                  (empirical std = level_value/√3, NOT level_value)
      - Shift:    constant = level_value

    Consequence: at the same `level_value`, the 3 distributions have
    DIFFERENT empirical noise strengths. Paper writing must clarify
    that `level_value` is each distribution's natural parameter, not
    the empirical std. See §5.1.5 for asymmetric uniform extension.
    """
    kwargs = {"noise_dist": dist_key, "eval_noise_scale": 0.0}
    if dist_key == "gaussian":
        kwargs[f"{channel_lower}_noise_std"] = level_value
    elif dist_key == "uniform":
        # Symmetric bounds; the natural parameter is `a = level_value`
        # (the half-range), NOT any std-related value.
        kwargs[f"{channel_lower}_noise_uniform_low"]  = -level_value
        kwargs[f"{channel_lower}_noise_uniform_high"] = +level_value
    elif dist_key == "shift":
        kwargs[f"{channel_lower}_noise_shift"] = level_value
    else:
        raise ValueError(f"Unknown dist: {dist_key!r}")
    return RobustConfig(**kwargs)


# Naming convention (X-2, RG-inspired compact):
#   OmniPiano-ClairDeLune-<C>-<Dist>-P<level·100>-v0
# where:
#   <C>       ∈ {A, O, R}       (Action / Observation / Reward)
#   <Dist>    ∈ {Gauss, Uniform, Shift}
#   P{XX}     numeric value × 100 (avoids `.` in env_id)
# Examples:
#   OmniPiano-ClairDeLune-A-Gauss-P05-v0   → Action Gaussian σ=0.05
#   OmniPiano-ClairDeLune-O-Uniform-P05-v0 → Obs Uniform bounds [-0.05, +0.05]
#   OmniPiano-ClairDeLune-R-Shift-P05-v0   → Reward constant shift = 0.05
# Mirrors RG paper figure label style `<algo>-<A/S/R>-<value>`, adapted:
#   - Use `O` (observation) instead of RG's `S` (state) to match our
#     `obs_noise_std` code terminology.
#   - Use `P{XX}` instead of raw `0.05` to keep env_id gym-tool-safe.

CHANNEL_LETTERS = {"Action": "A", "Obs": "O", "Reward": "R"}

# Channel-specific sweep values (all values match Robust-Gymnasium paper
# figures, ensuring direct data-point-level comparability, see §5.1.4):
#   Action / Obs: {0.05, 0.10, 0.15}
#   Reward:       {0.10, 0.30, 0.50}
#
# Rationale: OmniPiano per-step reward magnitude (~2-3) is comparable to
# RG Ant (~1-4), so RG's reward sigmas {0.1, 0.5} apply here as well.
# Action/Obs at 0.05-0.15 gives visible perturbation on joint angles
# (~2.9°-8.6°) without severe action-clipping distortion.
CHANNEL_LEVELS = {
    "Action": (0.05, 0.10, 0.15),
    "Obs":    (0.05, 0.10, 0.15),
    "Reward": (0.10, 0.30, 0.50),
}

for channel in ("Action", "Obs", "Reward"):
    channel_lower = channel.lower()
    channel_letter = CHANNEL_LETTERS[channel]
    levels = CHANNEL_LEVELS[channel]
    for dist_label, dist_key in (("Gauss", "gaussian"),
                                 ("Uniform", "uniform"),
                                 ("Shift", "shift")):
        for level_value in levels:
            level_label = f"P{int(round(level_value * 100)):02d}"  # 0.05→"P05", 0.30→"P30"
            env_id = (f"OmniPiano-ClairDeLune-"
                      f"{channel_letter}-{dist_label}-{level_label}-v0")
            cfg = _make_cfg_for_task(channel_lower, dist_key, level_value)
            register(
                id=env_id,
                base_env_name="RoboPianist-repertoire-150-ClairDeLune-v0",
                robust_config=cfg,
            )
```

### 5.1.4 为什么 channel-specific sweep（对齐 RG paper）

**Action / Obs 用 `{0.05, 0.10, 0.15}`**：

- 与 RG paper 中出现的 Action `-A-{0.05, 0.1, 0.15}` 和 State `-S-{0.05, 0.1, 0.15}` 直接对应
- 对 OmniPiano action canonical `[-1, 1]`：0.05-0.15 对应 5-15% range，clip 影响 <20%，nominal ≈ effective std
- 对 OmniPiano obs `joints_pos` (radians)：0.05-0.15 对应 2.9°-8.6° 关节角扰动，可解释、可见
- 与 RG paper 每 sweep 值都能找到对应数据点

**Reward 用 `{0.10, 0.30, 0.50}`（比 Action/Obs 大 2-10×）**：

- 与 RG paper 中出现的 Reward `-R-{0.1, 0.3, 0.5}` 直接对应
- OmniPiano per-step reward magnitude ~2-3（与 RG Ant reward ~1-4 相近）：
  - σ=0.1 → 相对 3-5%（轻扰动，起点）
  - σ=0.3 → 相对 10-15%（中扰动）
  - σ=0.5 → 相对 20-25%（强扰动）
- 与 Action/Obs 用同一 sweep（如 {0.05, 0.10, 0.15}）**语义不对称** —— reward 是 scalar，用小 sigma 几乎无效果；对齐 RG 的 reward sigma 是必要的

**Paper 写作可直接引用**：

> "Our sweep values match those reported in Robust-Gymnasium [Gu et al. 2025]: `{0.05, 0.1, 0.15}` for action and observation channels, `{0.1, 0.3, 0.5}` for reward channel. This choice ensures direct data-point comparability with the reference benchmark."

**关于 task 命名 `-A/O/R-<Dist>-P<XX>-v0` 的设计动机**：

1. `**A/O/R` 单字母 channel**：模仿 Robust-Gymnasium paper figure label（`PPO-A-0.05`、`IPPO-R-0.5` 等），让 paper 里的 legend 与 env_id 结构对齐。我们用 `O` 表示 observation 而不是 RG 的 `S`（state），保持与 code 里 `obs_noise_std` 术语一致。
2. `**P{XX}` value 后缀**：`P05` 表示 `level_value = 0.05`（除以 100 就是原值）。**跨 3 种分布共用 numerical value**（Gaussian σ、Uniform half-range a、Shift constant 数值都用同一个 P05）。避免 env_id 里含 `.`，减少解析冲突。
3. `**P{XX}` 里的数字**是**每分布自然参数值**，**不是** empirical std：
  - `A-Gauss-P05` → σ=0.05，empirical std=0.05
  - `A-Uniform-P05` → bounds [-0.05, +0.05]，empirical std = 0.05/√3 ≈ **0.029**（比 Gaussian 小）
  - `A-Shift-P05` → constant=0.05，empirical std = 0（确定性）
   Paper 需明确写明这一点：`P{XX}` 是每分布 natural parameter 的 sweep axis，**不是** empirical std 对齐。

**Paper 写作示例**：

> "We sweep the natural parameter of each distribution at three levels P01=0.01, P05=0.05, P10=0.10, encoded in the env_id suffix. The value P is Gaussian σ for Gauss tasks, Uniform half-range `a` for Uniform tasks (symmetric bounds [-a, +a]), and constant offset for Shift tasks. Note that these are per-distribution natural parameters, not empirical noise stds; the resulting empirical stds differ across distributions at the same P value (Uniform's empirical std is P/√3 ≈ 0.577·P; Shift's is deterministic 0)."

### 5.1.5 非对称 uniform 扩展（v1 infrastructure supports, tasks 未注册）

v1 registered tasks 全部用**对称** uniform (`low = -level, high = +level`)，但 `**RobustConfig` infrastructure 完整支持非对称** —— 因为 Option C.3 让 `_uniform_low` 和 `_uniform_high` 是 2 个独立字段。

**非对称 uniform 的研究场景**（v2+ 或用户自定义 task 用）：

- **Sensor 零点漂移**：`obs_noise_uniform_low=+0.02, high=+0.10` → 关节角读数 always 高估 0.02-0.10 rad
- **Actuator 单侧故障**：`action_noise_uniform_low=+0.10, high=+0.20` → 电机 always drift 正向 10-20%
- **Reward 单侧腐化**：`reward_noise_uniform_low=+0.5, high=+1.0` → reward 系统性偏高（模拟宽松评分）

**注册示例（不属于 v1 27 tasks，仅演示 infrastructure）**：

```python
register(
    id="OmniPiano-ClairDeLune-A-Uniform-Asym_02_10-v0",
    base_env_name="RoboPianist-repertoire-150-ClairDeLune-v0",
    robust_config=RobustConfig(
        noise_dist="uniform",
        action_noise_uniform_low=+0.02,      # 非对称
        action_noise_uniform_high=+0.10,     # 非对称
    ),
)
```

命名建议：`Asym_<low·100>_<high·100>` 后缀标注非对称边界（`Asym_02_10` = [+0.02, +0.10]）。也可以加 `_neg` / `_pos` 说明偏向。

**v1 default sweep 不含非对称的理由**（对齐 §11 延期理由）：

- RG paper 也未做非对称 sweep（他们默认 low=high=0.4 是退化配置，实际 examples 都用对称零均值）
- Piano-playing 的核心 robustness 挑战是"unbiased white noise 下能否弹准键"，对称即够用
- 加非对称 task 至少 double 训练成本（27 × 2 = 54 tasks），v1 paper focus 不需要
- Phase 2+ 可扩展；命名规范已保留（`Asym_XX_YY` suffix）

**已决议（详见 §12 决议 4 / 5 / 6，2026-06-27）**：

(a) **噪声级别 —— 采用 Option E1（§12 决议 4）**：**action / obs `{0.05, 0.10, 0.15}`**、**reward `{0.10, 0.30, 0.50}`**，严格对齐 Robust-Gymnasium paper 的 figure labels（可 data-point-level 直接对照），命名后缀 `P{XX}` = level×100。原提案 `{0.01, 0.05, 0.10}`（及 `{0.005,0.05,0.5}` / `{0.01,0.03,0.1}` 等替代）**已弃用**。详见 §5.1.4 / §6。

(b) **净基线 —— 在家族内显式注册 `OmniPiano-ClairDeLune-Clean-v0`（所有 std=0）**（§12 决议 5）：让 sweep 自包含、reviewer 不必跨 family 找隐式 clean baseline。

(c) **Shift 分布 —— v1 包含**（§12 决议 6）：语义为 program-run-level 恒定 `+shift_value`（与决议 2 一致），成本极低且给 paper 3-way 分布 sweep 与 RG 直接对照。v1 = **27 robust + 1 clean = 28 tasks**。

### 5.2 Eval harness —— `tools/robust_eval_sweep.py`

现有 `examples/checkpoint_replay_eval.py` 的姊妹。给定：

- 一个训练好的 ckpt（来自 27 个注册 env 之一 OR 净训练的 ckpt）
- 噪声 grid：`eval_noise_scale` 值列表（例如 `[0.0, 0.5, 1.0, 2.0, 4.0]`）

它循环（`env_id` 指 **eval env**——robustness 曲线要测的那个；ckpt 可训自它，也可能是净训 ckpt 做"对未见噪声的泛化"，见 §7）：

```python
policy  = load_policy(ckpt)      # ★ 按框架:SB3 PPO/SAC.load();robust 任务无 cost=SB3 地盘,
                                 #   不要照抄 checkpoint_replay 的 omnisafe.Evaluator(OmniSafe 专用)
channel = channel_of(env_id)     # 读 env_id 的 -A- / -O- / -R- 标记

# ★ 固定锚种子,与 scale 无关 → 各 scale 跑同一批 episode(同 MIDI/初始条件),
#   只让噪声变化,曲线不被 env-init 漂移混淆(与 checkpoint_replay 同惯例)
SEED_BASE = train_seed + EVAL_SEED_OFFSET

if channel == "R":
    # reward 任务(决议 11):与 A/O 对称,在自身 env 上 sweep eval_noise_scale。
    # canonical:直接在 Clean-v0 上单点评估;曲线横轴用**训练档位**(P10/P30/P50,由训练 env_id 决定,见 §6)
    eval_env = omnipiano.make("OmniPiano-ClairDeLune-Clean-v0", mode="eval")
    metrics  = replay(policy, eval_env, n_eps=N, seed_base=SEED_BASE)
    write_csv_row(env_id, eval_on="Clean-v0", metrics)
else:
    # action / obs 任务:在 eval env 自身上扫 scale grid
    for scale in noise_grid:                       # 例如 [0.0, 0.5, 1.0, 2.0, 4.0]
        eval_env = omnipiano.make(
            env_id, mode="eval", _eval_noise_scale_override=scale)
        metrics = replay(policy, eval_env, n_eps=N, seed_base=SEED_BASE)  # ← 同一 SEED_BASE
        write_csv_row(env_id, scale, metrics)
```

**正确性要点（伪代码已体现，直接回答"和训练是否一致"）**：
- **结构一致靠 env_id**：eval env 与训练 env 共用同一 `env_id` → task/MIDI/全部 wrapper/obs-space/action-space 全同，**唯一差别是噪声幅度被 `scale` 缩放**（`scale=1.0` 即复现训练噪声级别）。这由"env_id = 单一 canonical handle"保证。
- **固定锚种子**：所有 scale 共用 `SEED_BASE`，各 scale 跑同一批 episode，曲线只反映噪声、不混入 env-init 漂移。
- **按 channel 分流**：A/O 扫 scale grid；R 在 `Clean-v0` 上单点（详见下方 Reward 特殊处理 + §6）。**注意**：R 任务 eval **有意** clean，故与训练（带噪）不同——这是设计，不是不一致。
- **无 obs 归一化不匹配**：OmniPiano 训练/eval 都吃 raw obs（**不用 `VecNormalize`**，§11），策略可直接在 `make()` 出来的 env 上跑，无需搬运归一化统计量。
- **ckpt 按框架加载**：robust 任务是 SB3 地盘，用 `PPO/SAC.load`；`checkpoint_replay_eval.py` 的 `omnisafe.Evaluator` 是 OmniSafe 专用，**不可照抄**。

输出：列为 `(env_id, ckpt_epoch, eval_noise_scale, eval_seed, ep_return, ep_cost, ep_f1, ep_noise_*)` 的 CSV —— 喂给 robustness 曲线绘图。

> ⚠️ **已被决议 11（2026-07-04）取代**：reward 现在与 action/obs 对称——sweep tool **照常对 `-R-` 任务在其自身 env 上跑 `eval_noise_scale` grid**（不再跳过、不再换 Clean-v0）。以下原文作废，仅作历史。

**Reward 通道任务的特殊处理(决定 2026-07-04)**：对 reward-channel checkpoint(训练 env_id 含 `-R-`),sweep tool **跳过 scale grid,直接在 `OmniPiano-ClairDeLune-Clean-v0` 上做单次 eval**。因为 reward 噪声在 eval 被 `make()` 强制置零(§0.2),对 R 任务遍历 `eval_noise_scale>0` 只会得到 F1 完全相同、仅 `ep_return` 读数无意义波动的重复行;而 force-zeroed R-env 与 Clean-v0 byte-equivalent,故 canonical 做法是直接在 Clean-v0 上评估(见 §6 Reward 段)。R 任务的 robustness 曲线在**训练噪声档位**轴上跨 P10/P30/P50 画,不在此 tool 的 scale grid 上。tool 应据训练 env_id 的 channel 标记(`-A-`/`-O-`/`-R-`)自动分流:A/O 在自身 env 上跑 scale grid,R 在 Clean-v0 上跑单点。

**已决议（§12 决议 7，2026-06-27）**：采用 `make()` 上的私有 kwarg `_eval_noise_scale_override=` 做运行时覆盖，**仅供这个 sweep tool 使用，不暴露给一般用户**。替代方案（为每个 `(channel, dist, std, scale)` 四元组注册一个 env）会让 env 数膨胀到约 135 个，已否决。详见 §12 决议 7。

### 5.3 Phase 1 交付物

- 27（或 28）个注册 robust task id
- `tools/robust_eval_sweep.py`
- 每 channel 1 个 baseline 实验 = PPO + PPOLag 在 `OmniPiano-ClairDeLune-{A,O,R}-Gauss-P05-v0`，seed=0，5M 完整训练。共 6 个 run。
- 每个 algo × task 的 1 次 robustness sweep = 对每个 ckpt 在 scale ∈ {0, 0.5, 1.0, 2.0, 4.0} 上跑 `robust_eval_sweep.py`。
- 绘图脚本 `tools/plot_robustness_curves.py` 读取 sweep CSV → 每 task 一张图，每 algo 一条曲线。

---

## 6. v1 首批具体任务清单

采用 **channel-specific sweep**（全部 RG paper 对齐，见 §5.1.4）+ **X-2 命名规范**（`OmniPiano-ClairDeLune-<C>-<Dist>-P<XX>-v0`），首批任务表如下：

### Action + Obs（sweep `{0.05, 0.10, 0.15}`）


| Channel | Dist    | P05 (0.05)    | P10 (0.10)    | P15 (0.15)    |
| ------- | ------- | ------------- | ------------- | ------------- |
| **A**   | Gauss   | A-Gauss-P05   | A-Gauss-P10   | A-Gauss-P15   |
| **A**   | Uniform | A-Uniform-P05 | A-Uniform-P10 | A-Uniform-P15 |
| **A**   | Shift   | A-Shift-P05   | A-Shift-P10   | A-Shift-P15   |
| **O**   | Gauss   | O-Gauss-P05   | O-Gauss-P10   | O-Gauss-P15   |
| **O**   | Uniform | O-Uniform-P05 | O-Uniform-P10 | O-Uniform-P15 |
| **O**   | Shift   | O-Shift-P05   | O-Shift-P10   | O-Shift-P15   |


### Reward（注册档位 `{0.10, 0.30, 0.50}`；决议 11 后 eval 与 action/obs 一样按 `eval_noise_scale` sweep）


| Channel | Dist    | P10 (0.10)    | P30 (0.30)    | P50 (0.50)    |
| ------- | ------- | ------------- | ------------- | ------------- |
| **R**   | Gauss   | R-Gauss-P10   | R-Gauss-P30   | R-Gauss-P50   |
| **R**   | Uniform | R-Uniform-P10 | R-Uniform-P30 | R-Uniform-P50 |
| **R**   | Shift   | R-Shift-P10   | R-Shift-P30   | R-Shift-P50   |


> ⚠️ **已被决议 11（2026-07-04）取代**：reward 与 action/obs **完全对称**——默认 matched eval、横轴用 `eval_noise_scale`（不再是训练档位轴）、不换 Clean-v0；真实 return 用 `ep_return_true`/`ep_return_noised` 两列，F1 仍是干净 headline。详见 §12 决议 11。以下原文作废。

**Reward 通道的 eval / 曲线语义（与 action/obs 不同，决定 2026-07-04）**：reward 噪声是**训练期专属**扰动，eval 时被 `make(mode="eval")` 强制置零(§0.2 guard)，`eval_noise_scale` 对 R 任务无效（若请求非零则 warn 指路 Clean-v0）。**canonical 评估协议:每个 reward-trained checkpoint(来自 P10/P30/P50 训练 env)一律在 `OmniPiano-ClairDeLune-Clean-v0` 上 eval**（force-zeroed R-env 与 Clean-v0 byte-equivalent,以 Clean-v0 为准最无歧义）。robustness 曲线横轴是**训练噪声档位**（P10→P30→P50），纵轴是 **F1 on Clean-v0**，对照 `Clean-v0`(clean-trained) 基线 —— 衡量"训练在多脏的 reward 信号下，学出来的策略退化多少"（训练期鲁棒性）。这与 action/obs 的"部署期鲁棒性曲线"（x=`eval_noise_scale`，同一策略在不同 eval 噪声下）互补。`**tools/robust_eval_sweep.py`（§5.2）对 R 任务不做 scale grid，直接在 Clean-v0 上评估各训练档位的 checkpoint**。

### Clean baseline

- `OmniPiano-ClairDeLune-Clean-v0`（全 channel std=0）

所有 task id 完整前缀都是 `OmniPiano-ClairDeLune-…-v0`。例如：

- `OmniPiano-ClairDeLune-A-Gauss-P05-v0`（Action Gaussian σ=0.05）
- `OmniPiano-ClairDeLune-O-Uniform-P10-v0`（Obs Uniform bounds [-0.10, +0.10]）
- `OmniPiano-ClairDeLune-R-Shift-P30-v0`（Reward constant shift = 0.30）

**总计**：**Action 9 + Obs 9 + Reward 9 + Clean 1 = 28 个 registered env_id**。

**P 值语义再强调**（跨分布 numerical 一致，但 empirical std 不同）：

Action / Obs `{0.05, 0.10, 0.15}`：


| P 值 | Gaussian σ | Uniform bounds | Uniform empirical std | Shift constant |
| --- | ---------- | -------------- | --------------------- | -------------- |
| P05 | 0.05       | [-0.05, +0.05] | 0.029                 | +0.05          |
| P10 | 0.10       | [-0.10, +0.10] | 0.058                 | +0.10          |
| P15 | 0.15       | [-0.15, +0.15] | 0.087                 | +0.15          |


Reward `{0.10, 0.30, 0.50}`：


| P 值 | Gaussian σ | Uniform bounds | Uniform empirical std | Shift constant |
| --- | ---------- | -------------- | --------------------- | -------------- |
| P10 | 0.10       | [-0.10, +0.10] | 0.058                 | +0.10          |
| P30 | 0.30       | [-0.30, +0.30] | 0.173                 | +0.30          |
| P50 | 0.50       | [-0.50, +0.50] | 0.289                 | +0.50          |


---

## 7. Train/eval 协议（paper 写作）

paper 要报告的**两种协议**：


| 协议                           | 训练 env                                      | 评估 env                                                               | 测的是什么                           |
| ---------------------------- | ------------------------------------------- | -------------------------------------------------------------------- | ------------------------------- |
| **In-training robustness**   | `OmniPiano-ClairDeLune-A-Gauss-P05-v0`（噪声开） | 同 env，`eval_noise_scale=1.0`（噪声开）                                    | 给定训练时噪声，policy 是否学到鲁棒性          |
| **Post-training robustness** | `OmniPiano-ClairDeLune-A-Gauss-P05-v0`（噪声开） | 同 env，`eval_noise_scale=0.0`（净 eval）                                 | 噪声训练的 policy 在净 eval 上是否退化（有时会） |
| **对未见噪声的泛化**                 | `OmniPiano-ClairDeLune-Clean-v0`（无噪声）       | `OmniPiano-ClairDeLune-A-Gauss-P05-v0` 配 `eval_noise_scale=1.0`（噪声开） | 净训练的 policy 在测试时面对噪声能否撑住        |


**默认对应哪一行(决议 10,2026-07-04)**：`eval_noise_scale` **默认 = 1.0**,即默认 eval 就是第 1 行 **In-training / matched**——默认那个 eval 数直接是 RG 可比的鲁棒性数(RG 在训练档噪声下评估,无 clean-eval 概念)。第 2 行 post-training(clean)需**显式**设 `eval_noise_scale=0.0`。**reward 通道(决议 11)**:与 action/obs 完全对称——同样默认 matched、可 sweep、**不 force-zero**;真实 return 用 `ep_return_true` 列(F1 仍是干净 headline)。

Reviewer 至少想看前 2 种（in-training + post-training）。第 3 种（泛化）是 bonus，一旦 eval sweep 存在就是一行代码的事。

**报告指标**：F1（per-step）和 EpRet，画成 **robustness 曲线**，针对一个或多个 checkpoint。**横轴按 channel 分**：**action / obs** 用 x = `eval_noise_scale`（部署期鲁棒性,同一策略在不同 eval 噪声下）；**reward**（决议 11）：与 action/obs 一样用 x = `eval_noise_scale`（matched 默认、可 sweep）；真实 return 用 `ep_return_true` 列。~~原"训练档位轴 / Clean-v0"~~已作废。

---

## 8. 文件清单 —— 哪些被创建或修改


| 文件                                              | Phase | 状态                                                                              |
| ----------------------------------------------- | ----- | ------------------------------------------------------------------------------- |
| `omnipiano/configs/__init__.py`                 | 0     | 修改（扩展 RobustConfig）                                                             |
| `omnipiano/wrappers/robust_wrapper.py`          | 0     | 修改（加 reward noise + 分布开关）                                                       |
| `omnipiano/envs/dm_env_obs_noise.py`            | 0     | 修改（抽出 OBS_NOISE_SEED_OFFSET 常量 + 加分布支持以匹配）                                      |
| `omnipiano/envs/registration.py`                | 0     | 修改（eval_noise_scale 接线，使用来自 dm_env_obs_noise.OBS_NOISE_SEED_OFFSET 的命名常量）       |
| `omnipiano/multiagent/registration.py`          | 0     | 修改（使用同样的命名常量；去重 line 319 的 `+31415` 第二份副本）                                      |
| `omnipiano/utils/logger_wrapper.py`             | 0     | 修改（eval CSV 加 4 列：`eval_noise_scale` + `ep_noise_{action,obs,reward}`，见 §0.6.2） |
| `examples/checkpoint_replay_eval.py`               | 0     | 修改（`_CSV_HEADER` + 写入同步加同样 4 列，schema 与 logger_wrapper 锁定，见 §0.6.1）             |
| `omnipiano/utils/info_keys.py`                  | 0     | 修改（加 ROBUST_NOISE_REWARD）                                                       |
| `omnipiano/tests/test_robust_v1_equivalence.py` | 0     | 新建（bit-exact 回归测试）                                                              |
| `omnipiano/envs/__init__.py`                    | 1     | 修改（追加约 28 个 robust task 注册）                                                     |
| `tools/robust_eval_sweep.py`                    | 1     | 新建（sweep harness，约 150 LOC）                                                     |
| `tools/plot_robustness_curves.py`               | 1     | 新建（曲线绘图，约 100 LOC）                                                              |
| `omnipiano/docs/robust_task_design.md`          | （本文档） | 已存在                                                                             |


---

## 9. 测试和验证


| 测试                                                                                                                                                     | Phase | 类型                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- | ----------------------------------------------------------------------------- |
| `test_robust_v1_equivalence.py`（上文 §0.7）                                                                                                               | 0     | 在 2 个现有 robust env 上 bit-exact 回归（除掉 seed offset 从 31415 → 20000 的有意 bump 之外） |
| `test_obs_noise_seed_consistency.py`（§0.5）                                                                                                             | 0     | SA env 和 MA env 在同 master seed 下首个 obs-noise 样本必须一致                           |
| Smoke test：`omnipiano.make("OmniPiano-ClairDeLune-A-Gauss-P05-v0").reset()` 能跑                                                                         | 0/1   | smoke                                                                         |
| Smoke test：`mode="eval"` 配 eval_noise_scale=0.5 → info 上的有效 std = 原值 × 0.5                                                                        | 0     | unit                                                                          |
| Reward matched eval(§0.2,决议 11)：reward-channel 任务 `mode="eval"` 下,reward 噪声按 `eval_noise_scale` 缩放(与 action/obs 对称,**不 force-zero、不 warn**);scale=1.0→matched、0.0→clean(已在 `test_robust_eval_scale.py` 覆盖) | 0     | unit                                                                          |
| 分布合理性：symmetric std-matched uniform (`low=-0.1·√3, high=+0.1·√3`) 采 10000 样本 → empirical std ≈ 0.1 ± 1%                                                | 0     | unit                                                                          |
| Uniform 非对称支持：`uniform_low=-0.02, uniform_high=+0.10` 采 10000 样本 → mean ≈ 0.04, std ≈ 0.035                                                            | 0     | unit                                                                          |
| `__post_init__` 校验：设 `noise_dist='gaussian'` 但 `action_noise_shift=0.05` → 抛 `ValueError`                                                              | 0     | unit                                                                          |
| `__post_init__` 校验：设 `uniform_low > uniform_high` → 抛 `ValueError`                                                                                     | 0     | unit                                                                          |
| CSV：跑 1 episode eval 后，3 个新列都存在且非 NaN                                                                                                                  | 0     | smoke                                                                         |
| Sweep：`robust_eval_sweep.py` 在 5 点 grid 上完成时间 <5 min/ckpt                                                                                              | 1     | smoke                                                                         |


---

## 10. Paper 写作笔记（边做边记）

根据 `feedback_paper_writing_notes`，本节随实现进展累积实验数字和草拟 claim。初始种子：

### 我们预期会写的 claim

- "OmniPiano supports the random-perturbation subset of the disruptor taxonomy from Robust-Gymnasium [Gu et al. 2025], covering state, action, and reward channels under Gaussian / Uniform / Shift distributions, with strict separation between training-time and evaluation-time noise levels."
- "OmniPiano-Robust covers the observation, action, and reward disruptor families of Robust-Gymnasium [Gu et al. 2025]. We deliberately exclude adversarial perturbations (which require per-paper-method infrastructure not shared across baselines — LLM API, gradient access to the policy, or a separately trained adversary) and environment/dynamics randomization. The latter is out of scope by design: dynamics randomization is motivated by sim-to-real transfer, which does not apply to a fixed-repertoire pure-simulation music benchmark, and perturbing physical parameters (e.g. piano-key spring stiffness) entangles the robustness axis with the definition of correct play (the force-to-key-activation mapping the F1 metric depends on), unlike observation/action/reward noise which leaves the ground-truth F1 measurement intact."

### Phase 1 跑完后要填的数字

- 每 channel：PPO 在 `ClairDeLune` 上 std=0.05 时的 净 vs 扰 F1 跌幅
- 每 channel：PPOLag 在 `ClairDeLune` 上 std=0.05 时的 净 vs 扰 F1 跌幅
- 每 channel 的 robustness 曲线斜率（ΔF1 / Δeval_noise_scale）
- "shift" 分布的曲线是否与 Gaussian 在质上不同（YES 有意思，NO 在预期内）

### 值得记录的负结果（预期可能出现）

- 我们预期 "shift" 在与 Gaussian 同 numerical level_value 时（如 A-Shift-P05 vs A-Gauss-P05 都用 0.05）表现相似，因为 policy 原则上可学到 shift 的恒定 bias。如果显著不同，那就是个 result（注：shift 的 empirical std=0，Gaussian 的 empirical std=0.05；即使 numerical value 相同也是不同扰动，具体是 policy 应对 bias 还是 white noise 的能力对比）。
- 我们预期 std=0.10 的 reward noise（大致是一个 timestep reward 的 magnitude）会严重降低学习。如果真降，论证应当用更紧的 reward std 默认值；如果没降，那 policy 比朴素理论预测更鲁棒。

---

## 11. 为什么这些不做 / 延期（理由）

供 reviewer / paper 写作参考：


| 特性                                           | 为什么不在 v1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 何时重新审视                                                                                              |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **Dynamics randomization（out of scope，非延期）** | **已从 plan 移除（2026-07-03）**。四点理由:(1) **动机缺失**——dynamics randomization 的经典正当理由是 sim-to-real transfer(随机化因为不知真实机器人参数),OmniPiano 是纯仿真、无真实硬件目标,该动机不成立;obs/action/reward 噪声则有自洽动机(传感器/执行器噪声、reward 污染)。(2) **纠缠任务定义**——改物理参数易改变"弹对了"的定义(key spring 改变力→键激活映射 → F1 语义漂移),破坏 robustness 边界;而三 channel 噪声不碰 F1 的 ground-truth 测量。(3) **工程成本失衡**——hand+piano MJCF 数百可改参数 + MidiEval 缓存破坏风险,换薄弱收益。(4) **逐轴看无一值得**:mass/friction 边缘;actuator gain 有 positioning 故事但与 action noise 概念重叠;key-weight 是唯一 piano-native("不同键重的钢琴")但纠缠 F1 且小众。                                                                                                                | **不做**。若 reviewer 坚持要一个 dynamics 轴,`key-weight` 是唯一 piano-native 候选,但需单独评估其对 F1 语义的影响,**不预先承诺建基建**。 |
| **对抗扰动**                                     | Robust-Gymnasium 唯一的对抗实现是 LLM-prompted 的，需要 per-step API call（约 700 小时跑 5M-step 训练）。非 LLM 对抗方法（gradient FGSM、RARL、ATLA、SA-PPO）是单独 paper 的贡献 —— 每个都要修改 baseline 算法（PPO/SAC），不仅仅是 env。横跨我们的跨库（SB3 + OmniSafe）设计。                                                                                                                                                                                                                                                                                                                                                                                                                              | v2 paper 后若 reviewer 要求。                                                                            |
| **Episode 级频率（仅对 Shift 有意义）**                | v1 中 Gaussian 和 Uniform **已经是 step-level**（每 step 独立采），只有 Shift 是 program-run-level（全程恒定 `+shift_value`）。Phase 2 想加的 episode-level 特指 **shift 的 episode 级随机化**（每 reset 采一次符号或 magnitude，episode 内固定）—— 让 shift 从确定性 calibration error 变成"随机 domain 偏移"。加入它需要在 RobustConfig 加 `shift_frequency` 字段（`constant` / `episode` / 未来 `random_interval`），并在 `reset()` 里采样。**Phase 2 的 episode-level shift 是新增变体**（作为 `shift_frequency="episode"` 的选项），**不替换** v1 的 `constant` 语义（program-run-level `+shift_value`），保证 v1 baselines 可复现。对 Gaussian/Uniform 强行加 episode-level 变体（每 reset 采一次然后整 episode 固定）与 domain randomization 概念重合，独立价值不高，暂不在计划中。 | Phase 2（若 shift episode-level 有需求）。                                                                 |
| **Per-dim 噪声 std / per-channel dist**        | 给 RobustConfig 和 per-key obs filter 加组合复杂度。Scalar-std v1 已经做的是 per-dim 独立噪声（只是每 dim 上同样 std）。异质 std 是大多数 paper 不需要的精细化。**注**：Option C.3 的 Uniform bounds（low/high）已经允许**每 channel 独立控制非对称性**（用户可注册 `action_noise_uniform_low=-0.02, high=+0.10` 的任务），是"per-channel 精细化"的一种局部实现。但真正的"per-dim std"（每个 action 维度或每个 obs key 不同 std）仍需 dict-typed 字段替换 scalar，Phase 1.5+ 才做。                                                                                                                                                                                                                                                                    | Phase 1.5 如果 reviewer 明确要求。                                                                         |
| **Per-key obs σ 校准**                         | 一个 σ 应用到所有 obs key 上（`joints_pos` in rad、`joints_vel` in rad/s、`piano/state` in [0,1]）意味着不同 key 感受的相对扰动强度不同。理论上可给每 key 独立 σ 让"相对扰动强度对齐"，但：(1) **RG 完全不做这个**（全仓库无 per-key sigma、无 sigma dict），OmniPiano 保持不做保证与 RG 直接可比；(2) 校准需要对每 key 手工估计典型 range 或做统计校准，增加实验工程复杂度；(3) 现状"joint-angle 主导"是可解释的诊断信号（读者能直接把"σ=0.05 → F1 下降 20%" 归因到"关节角 2.9° 扰动"）。v1 paper 明确 disclose 这个 caveat 即可。                                                                                                                                                                                                                                                         | v2+ 如果 reviewer 要求 per-key 精细化。                                                                     |
| **Obs 归一化 wrapper**                          | 加 `VecNormalize` 或类似 wrapper 让 obs → N(0,1) 后再注 σ 会让"跨 key 相对扰动强度对齐"，但破坏与 RG paper 的 baseline 一致性（RG 直接加到 raw obs 上），也需要重跑所有现有 baseline。RG 完全不做这个，v1 保持不做。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | 明确**不计划**（除非重构整个 baseline 生态）。                                                                      |
| **N-hand SA robust**                         | 现有 wrapper 是 morphology-agnostic 的，所以实现成本为零，但实验成本（PPO × PPOLag × seed × N-morphology × 27 task）是 v1 的约 5 倍。v1 的 claim 已经足够；加入 morphology 给一个正交的轴更适合作为单独 paper 探索。                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | v1 paper 之后。                                                                                        |
| **MA Territorial robust**                    | RobustWrapper 假设单个 action ndarray；PettingZoo `ParallelEnv` 传的是 `Dict[agent_id, action]`。需要新的 `MARobustWrapper` 配 per-agent std（因为每个 agent 的 action 维度不同）。工作量大。                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Phase 2，配合任何 MA 专项 paper 扩展。                                                                        |
| **跨库统一 eval 阶段的 robust 语义（决定 2026-07-04）**   | eval 阶段的 robust 语义（`eval_noise_scale` 缩放,三通道对称——含 reward,决议 11）**保证锚在 `make(mode='eval')` 这个 env 构造入口**——任何库只要经此入口构造 eval env 就自动继承(SB3 模板即如此)。但**我们控不住每个库怎么构造/驱动它自己的 eval**:OmniSafe 用 `Evaluator.load_saved()` 从训练 config 重建 env,绕过 `make(mode='eval')`,故 Phase 0 的 eval 语义不会自动流入。**对比**:training 阶段的 robust 噪声与库无关(在 wrapper 链里、每 step 必经),完全可保证;eval 阶段则需库 opt-in 我们的入口,无法跨库强制统一。**决定:先不给 OmniSafe eval 阶段做 robust 特殊处理**——OmniSafe 主场是 safety 任务(有 cost 约束、**无 reward 噪声**),eval 天然干净;而 robust 任务(噪声、无 cost)是 SB3/CleanRL 地盘(走 `make(mode='eval')`,保证成立)。"OmniSafe × robust-eval"缺口大概率为空集。                                                    | 若将来真用 OmniSafe 训 robust 任务时再接线（让 checkpoint_replay 经 `make(mode='eval')` 构造 env,或用 `robust_eval_sweep.py`）。                              |


---

## 12. 给 reviewer 的 open questions（汇总）

**架构基础决议 ✅（2026-06-27）**：**Option C.3 —— per-distribution 独立字段**（RobustConfig 14 字段：3 std + 6 uniform_low/high + 3 shift + noise_dist + eval_noise_scale）。理由：字段名严格对应数学参数，严格模仿 Robust-Gymnasium 术语（`--noise-sigma` / `--uniform-low/high` / `--noise-shift`），允许非对称 uniform（用户可注册有偏 sensor drift 类任务），`__post_init__` 校验捕获错配。详见 §0.1。

**v1 sweep 参数化决议 ✅（2026-06-27，Option 4a）**：**Uniform 使用其自然参数**（`low` 和 `high` 独立设置），**不做 std-matching 转换**。v1 default sweep 用**对称 uniform** (`low = -level_value, high = +level_value`)。跨 3 种分布共用 numerical value `level_value ∈ {0.01, 0.05, 0.10}`，但每分布对应**不同**自然参数（Gaussian σ、Uniform half-range、Shift constant），且**empirical std 不同**（Gaussian=σ, Uniform=σ/√3, Shift=0）。非对称 uniform 由 infrastructure 支持但 v1 不默认注册，详见 §5.1.5。

**命名规范决议 ✅（2026-06-27，方案 X-2）**：env_id 采用 **RG-inspired 紧凑格式** `OmniPiano-ClairDeLune-<C>-<Dist>-P<XX>-v0`，其中 `<C> ∈ {A, O, R}`（Action/Obs/Reward，单字母模仿 RG paper figure label 的 `A/S/R`；用 `O` 代替 RG 的 `S` 以匹配我们 code 的 `obs_noise_std` 术语），`P<XX>` 是 level_value×100（避免 env_id 含 `.`）。详见 §5.1 registration loop 注释。

1. **§0.3（reward noise 放置位置）**：把 RobustWrapper 放在 SafetyWrapper 之后（当前 chain 顺序，前瞻防御）—— OK 吗？
  ✅ **已决定（2026-06-27）**：OK，保持当前顺序，reward noise 加在最外层（SafetyWrapper 之后）。理由：与 paper 标准定义"agent observes r̃ = r + ε" 一致；当前 SafetyWrapper 不修改 reward 所以与"放内部"功能等价，但未来若 SafetyWrapper 加入 reward-shaping 安全包装，外层放置保证 noise 始终扰动 agent 真正看到的 reward。
2. **§0.4（shift 分布语义）**：每 step 随机正负号 shift，而不是 episode 内常数 —— OK 吗？
  ✅ **已决定（2026-06-27）**：改为 **program-run-level 恒定 `+shift_value`**（其中 `shift_value` 来自 `RobustConfig.*_noise_shift` 独立字段，与 `_noise_std` 无关），与 Robust-Gymnasium `args.noise_shift` 完全一致。无 RNG 参与，无随机符号，每 step 每 episode 都加同一个 `+shift_value`。**注意**：Gaussian 和 Uniform 在 v1 中是 step-level（每 step 独立采），只有 shift 才需要专门指定为 program-run-level。Phase 2 加入 shift episode-level 变体时**不替换**这个 v1 语义。见 §0.4 更新后的实现。
3. **§4（RNG stream）**：2 条 stream（action+reward 共享、obs 独立）而不是 3 条 —— OK 吗？
  ✅ **已决定（2026-06-27）**：2 条 stream 足够。Audit 确认 RG 及 OmniPiano v1 每 task 均为 single-channel（见 §5.1.0），action 与 reward 分支永远只有一个被激活，共享 RNG stream 不会有 timing 干扰。Phase 2 若引入 multi-channel task，action 与 reward 分支按 `RobustWrapper.step()` 内固定顺序（先 action 后 reward）依次消费 stream，仍然 deterministic 可复现。
4. **§5.1(a)（噪声级别）**：`{0.01, 0.05, 0.10}` 级别 —— OK 还是换别的？
  ✅ **已决定（2026-06-27，Option E1）**：**channel-specific sweep 严格对齐 RG paper**：
  - Action / Obs：`{0.05, 0.10, 0.15}` —— 与 RG `-A-{0.05,0.1,0.15}` 和 `-S-{0.05,0.1,0.15}` 完全对应
  - Reward：`{0.10, 0.30, 0.50}` —— 与 RG `-R-{0.1,0.3,0.5}` 完全对应，并且与 OmniPiano per-step reward magnitude (~2-3) 匹配
   全部 sweep 值都在 RG paper 的 figure labels 里出现过，确保 paper 里可以 data-point-level 直接对照。命名后缀 `P{XX}` 依然是 level_value × 100（`P05=0.05, P30=0.30, P50=0.50`）。详见 §5.1.4 和 §6。
5. **§5.1(b)（净基线）**：在家族内显式注册 `OmniPiano-ClairDeLune-Clean-v0` —— OK 吗？
  ✅ **已决定（2026-06-27）**：**要**，家族内显式注册 `OmniPiano-ClairDeLune-Clean-v0`（所有 noise 字段都为 0 的 baseline）。理由：
  - Sweep 内自包含（robustness 曲线右端有明确 clean 参考点）
  - Reviewer 不必跨 family 找 clean baseline（不用引 `FingeringAnn` 等其他 env 作隐式 baseline）
  - 1 个 env 成本可忽略
  - 支撑 §7 表格里"对未见噪声的泛化"协议（净训 → 扰测）
   **v1 总 task 数 = 27 robust + 1 clean = 28**（已经反映在 §6 任务清单表里）。
6. **§5.1(c)（包含 Shift 分布）**：v1 包含 "Shift" 分布（27 个 task 而不是 18 个）—— OK 还是砍掉？
  ✅ **已决定（2026-06-27）**：**保留** Shift 分布，v1 = 27 task（+ 1 Clean = 28）。语义采用 **program-run-level 恒定 `+shift_value`**（与 Q2 决议一致），其中 `shift_value` 是 `RobustConfig.*_noise_shift` 字段（Option C.3 独立字段，**与 `*_noise_std` 无关**）。加入的成本极低（无 RNG，无 reset 逻辑），且给 paper 提供 3-way 分布 sweep 与 Robust-Gymnasium 的直接对照（RG `noise_shift` 直接对齐），值得纳入。**注**：v1 registered shift tasks 用**正值**（`shift = +level_value`），符合 RG paper 惯例；非对称/负值 shift 由 infrastructure 支持但 v1 不默认注册。
7. **§5.2（eval scale 覆盖路径）**：允许 sweep tool 通过 `_eval_noise_scale_override=` 私有 kwarg 在 make() 时覆盖，而不是为每个 `(env × scale)` 对注册一个 env —— OK 吗？
  ✅ **已决定（2026-06-27）**：**采用 `_eval_noise_scale_override=` 私有 kwarg** 方案（B1）。仅 `tools/robust_eval_sweep.py` 使用，不暴露给一般用户。理由：
  - **避免 env 数爆炸**：不这样做则要注册 28 × 5 scale = 140 env
  - **eval scale 是"评估行为"**：概念上不属于"env 属性"（同一策略在不同 scale 下应有不同表现）
  - `**_` 前缀清晰标示"内部 API"**：普通用户不会误用
  - **保持"train 时用 registry"的纯粹性**：训练一定从 registered env id 开始，只有 eval sweep 走覆盖路径
  - **保留了 registry-only philosophy 的 90%**：只在一处（eval sweep）打小口子，`SafeRoboPianist` 的 SB3 / OmniSafe 集成 + 复现性追溯 + 现有 safety task 传统惯例都不受影响
   实现细节（在 `omnipiano.make()` 里）：

**决议 8（2026-06-27 决定；2026-07-03 Step 1 完成 ✅）：obs["action"] / obs["reward"] principled semantics 修正（§4.7 方案 6 Option B）**

✅ **已决定并完成 prototype**：正式采纳 **方案 6 Option B（physical override）**。RobustWrapper 在 step 的 POST 阶段覆盖 obs 里 action / reward 两个 slot（action：clean commanded 的 physical 版本；reward：noised observed，Phase 0 接线）。

**Step 1 Prototype 结果（2026-07-03）**：

- 分支 `feat/robust-method6-prototype`（commits `f8510b1` + `f7b8a5b`），main 未动
- **15/15 测试全绿**（A1-A6 核心 gate + fail-fast/fallback + G1-G6/G11 扩展 gate），含 A6 cross-condition bit-exact（100 step）与 A4 对象 identity（结构性 no-op）
- 3 个生产 env smoke 正常
- As-built 关键决策：复用 upstream `_scale_nested_action` + 缓存原 spec 对象（而非重写公式）→ bit-exact 构造性成立；override 仅在 action noise 激活时运行 → 零噪声 env 连 copy 都不发生
- 完整结果与 as-built 细节见 §4.7 顶部状态表

**后续**：reviewer sign-off → merge main → Phase 0 §0.3 加 reward slot override（复用已验证的 `_reward_slice` 机制）。

**理由（简要）**：

- 修正 obs["action"] noise leak（policy 反推 noise 的漏洞）
- 修正 obs["reward"] 与训练 signal 一致性
- 严格对齐 Wang 2020 noisy-reward RL + Gu 2025 Disrupted-MDP formalism
- Option B（physical 版本）保持 RoboPianist upstream 传统（physical 值在 obs["action"]），对现有 baseline **结构性零影响**
- Prototype-first 策略已兑现（一次全绿，未触发回退路径）

**决议 9（2026-07-04 ✅，用户拍板）：`make()` 的 `log_split` 参数改名为 `mode`（纯 rename）**

✅ **已决定**：把 `omnipiano.make()` 的 `log_split` 参数改名为 `mode`，值不变（`"train"` / `"eval"`）。

**这是纯变量名替换，当前行为零变化**：`make(mode="eval")` == 旧 `make(log_split="eval")` —— 同样门控 SafeRecord 挂载（仍额外由 `log_dir` 决定）、同样作 CSV 文件名前缀；`log_dir` 独立不动；**不引入第二个 flag、不合并不拆分**。调用点（`run_sb3_*.py` 等 SB3 模板）只需把 kwarg 名 `log_split=` → `mode=`，传的字符串不变。（`checkpoint_replay_eval.py` 不调 `make()`、无此 kwarg，不涉及。）

**理由（前瞻）**：Phase 0 会让这同一个 flag 额外承担 eval 噪声语义（按 `eval_noise_scale` 缩放**三通道**噪声——含 reward，见决议 11；见 §0.2）。届时 `log_split`（logging 名）控制 env 动力学 = "名字撒谎"；`mode`（train/eval 模式）名副其实。趁现在（未焊噪声行为、调用点少）改最便宜。**明确不做**：不引入独立的第二个 flag 去"解耦 CSV 与噪声语义"—— CSV 挂载已由 `log_dir` 门控半独立（`mode="eval"` 不传 `log_dir` = eval 语义、无 CSV），再加 flag 只会制造"两个 flag 必须保持一致"的负担。

**术语说明**：as-built——代码与文档均已统一为 `mode`（`log_split` 仅在本决议作为"改名前旧名"出现）。

**决议 10（2026-07-04 ✅，用户拍板）：eval 默认改为 matched（`eval_noise_scale` 默认 1.0），对齐 RG**

✅ **已决定**：`RobustConfig.eval_noise_scale` 默认从 0.0 改为 **1.0** —— 默认 eval 在**训练档噪声下**评估（matched），给出 RG 可比的鲁棒性数。

**依据（基于 RG 代码核实）**：Robust-Gymnasium 在 disrupted env 内按每 step 传入的 `args` 施加噪声，train/eval **复用同一个扰动 env**（eval 循环见 `examples/robust_nonstationary_env/main_stationary.py`；噪声机制见 `examples/robust_state/mujoco/test.py:36-41` 的 `robust_input`），**全仓无 clean/nominal-eval 开关**。即 RG 的鲁棒性指标 = "扰动下表现"，eval 与训练同噪声。原 clean 默认（0.0）偏离 RG 方法学，且默认那个数测的是"干净表现"而非鲁棒性。

**语义**：action/obs 默认 eval 噪声档 = 训练档（同分布、不同 seed）。clean/nominal eval 需显式 `eval_noise_scale=0.0`；sweep 覆盖 `{0, 0.5, 1, 2, ...}` 画曲线。

**reward 通道**：⚠️ 原"eval 恒 force-zero、reward 任务需设 `eval_noise_scale=0.0`"**已被决议 11 取代**——reward 现与 action/obs 完全对称（matched eval、不 force-zero、可 sweep），真实 return 用 `ep_return_true` 列。详见决议 11。

**as-built**：`configs/__init__.py` 默认 1.0；`test_robust_eval_scale.py` 覆盖（default→matched、explicit-0.0→clean、reward default→warn、reward explicit-0.0→no-warn），全 59 tests green。SB3 模板的 periodic / final eval 因此默认 matched（action/obs）。

**决议 11（2026-07-04 ✅，用户拍板）：reward 与 action/obs 对称——eval matched（撤销 force-zero），真实 return 用单独 CSV 列**

**背景 / 纠错**：决议 10 及之前把 reward 在 eval **force-zero**，前提是"eval 固定策略不消费 reward"。**该前提在 `action_reward_observation=True`（OmniPiano 默认）下是错的**：方案 6 的 reward-slot override 把 noised reward 写进 `obs["reward"]`，而 `obs["reward"]` **是策略输入**。force-zero → eval 的 `obs["reward"]` 变 clean → **与训练分布 OOD 不一致**，且 reward 噪声其实经此 slot 影响策略（**非 vacuous，甚至能影响 F1**）。

✅ **已决定**：reward **不再 force-zero**，与 action/obs **完全对称**——默认 matched（`eval_noise_scale=1.0`）、可 sweep。这样 eval 的 `obs["reward"]` 保持训练同分布。

**测量**：reward matched 下 `ep_return` 会被噪声污染 → **加两列**（仅 eval CSV）：
- `ep_return_true` = **reward 分解列之和**（MetricsWrapper 读加噪前物理值；`get_reward=reward_fn.compute=Σterms` 无额外变换，已代码严格核实 —— composite_reward.py:46-56 / piano_with_shadow_hands.py:210-211），作 headline；float64 分解和比 float32 累加更精确。
- 实收（带噪）return = **现有 `ep_return` 列**（as-built 保留其含义=received/accumulated，向后兼容不改；reward 任务下即"noised"）。交叉校验：`ep_return − ep_return_true = ep_noise_reward`。（**未单独加名为 `ep_return_noised` 的列**——`ep_return` 即承担该角色。）
- action/obs 任务两列相等（reward 标量未被污染）。**F1 恒为干净 headline**（读物理）。
- **仅加到我们控制的两个 eval CSV**（`SafeRecordEpisodeStatistics` + `checkpoint_replay`，有分解列）；**training rollout**（库原生 Monitor/progress）保持 noised（有意——agent 训练信号）、无分解列 → 无 true 列（靠 F1）。跨库：真值是环境层量（info 里的 `episode_task/*_reward`），库无关；写不写取决于 logger。

**CSV 规格补充（代码核实 2026-07-04）**：
- **`checkpoint_replay_eval.py`（OmniSafe）也加 true/noised 两列——schema-locked**：其 `_CSV_HEADER` 已含全部 6 个 reward 分解列（`_INFO_KEY_BY_CSV_COL` 从 `EpisodeInfoKeys.EPISODE_TASK_*_REWARD` 读），故 `ep_return_true` = 6 列之和直接可算、`ep_return_noised` = 它累加的 `ep_return`。它虽绕过 `make()`（走 Evaluator），但 MetricsWrapper 仍在重建链里 → 分解列照样有。**注**：它是 OmniSafe/safety 任务专用，safety 无 reward 噪声 → true == noised；加这两列主要为**与 SafeRecord schema 一致**（header 明确 "mirrors SafeRecordEpisodeStatistics schema"）。真正 true≠noised 的是 SB3 reward-噪声任务。
- **`ep_cost` 恒为 true value，不被任何噪声通道污染-as-measurement**：cost 由 `SafetyWrapper`（在 `RobustWrapper` 内层）的 constraint 算，**只读 physics 或 a_exec（执行的噪声动作），从不读 obs**（constraints.py 全部 `del obs`）；reward 噪声只碰 reward 标量、与 cost 独立；`RobustWrapper` 从不改 `info[*_SAFETY_COST_*]`。OmniSafe 从 `info[STEP_SAFETY_COST_TOTAL]` 取（`run_omnisafe_template.py:191`）、replay 从 `EPISODE_SAFETY_COST_TOTAL` 读——两个 CSV 存的都是真实轨迹的真 cost。**精确区别**（同 F1）：cost 数值会随轨迹被扰动而变（诚实测量被扰动轨迹），但绝不会像 reward 噪声污染 `ep_return` 那样被"事后加噪弄脏"。

**本决议撤销 / 取代**：S3a 的 reward eval force-zero + warn（已移除）；§0.6.3 的"reward eval 恒 clean / 不需 `ep_true_return`"；§6 的"reward eval 换 Clean-v0 / 只放训练档位轴"；§5.2 的"reward 跳过 scale grid"；决议 10 的 reward 例外。→ **reward 三通道完全对称：matched 默认、`eval_noise_scale` 轴 sweep、F1 干净 headline + `ep_return_true`/`ep_return_noised` 两列**。

**edge case**：`action_reward_observation=False`（非默认）时 reward 不进 obs，eval reward 噪声才真 vacuous；标准任务 OAR=on，matched 即正确。

**as-built（S6 完成）**：`registration.py` force-zero 已移除；reward 测试改 matched。**S6 CSV 列**（两个 eval CSV,schema-locked）：`ep_return` **保留=received**（reward 任务即 noised，不改），**新增** `ep_return_true`（=分解列之和,clean headline）+ `eval_noise_scale` + `ep_noise_{action_l2,obs_l2,reward}`；追加在末尾,现有列 index 不变。交叉校验 `ep_return − ep_return_true == ep_noise_reward`。gate `test_robust_eval_csv.py`（5）；全套 80 green。checkpoint_replay 的 header 同步（tools/ 被 gitignore,改动在本地）。

每个都可独立决定；想换的请告诉我，没说的我按 my recommendation 走。

---

## 13. Reviewer sign-off

Phase 0 获批后：

- 我按 §8 的 7 处文件改动逐项实现，每个逻辑单元 1 个 commit（共 5-6 个 commit）。
- 每个 commit 后，equivalence test（§0.7）必须仍然通过。
- Phase 1 仅在 Phase 0 sign-off（commit 完成）之后开始。

Phase 1 获批后：

- 我追加注册循环（§5.1）+ 写 eval harness（§5.2），共 2-3 个 commit。
- 我在 chain 脚本里跑 §5.3 的 6 个 baseline 实验，预计总共约 24 GPU·小时。
- 结果按照最近的 figures 扁平化约定 land 到 `examples/figures_aggregate/robust_v1/`。

---

## 14. 指标 × 噪声通道 × Logger 影响矩阵（代码核实 2026-07-04）

专门回答：SB3 / OmniSafe 的四类记录（training rollout CSV、replay CSV、deterministic eval CSV、`SafeRecordEpisodeStatistics`）里的 `ep_cost` / `ep_return` / `F1` / `recall` / `precision`，是否会受 obs / action / reward robust 噪声影响。

### 14.0 术语 + 两种"影响"

- **`ep_return` == `ep_reward`**：一个 episode 内 Σ per-step reward（实收）。是的，ep_return 就是这条 episode 的总 reward。
- 必须区分两种"受影响"：
  - **污染-as-measurement**：记录的数字 **≠ 该轨迹的真实值**（尺子被弄脏）。
  - **值随扰动轨迹变化（诚实）**：噪声改变策略行为 → 轨迹真变 → 指标如实反映**被扰动轨迹**。这**不是污染**。

### 14.1 指标来源（决定是否被污染；re-verified，与文档既有结论一致）

| 指标 | 来源 | 读 obs? | 结论 |
|---|---|---|---|
| **F1 / precision / recall** | `MidiEvaluationWrapper` 读 physics（`task.piano.activation` + MIDI `task._notes`，`robopianist/wrappers/evaluation.py:70/117`） | ❌ | 永不污染（§0.6.5(a) 已验证，仍正确） |
| **ep_cost** | `SafetyWrapper` 的 constraint 读 physics 或 a_exec；`constraints.py` 全 `del obs`；reward 噪声不碰 cost；`RobustWrapper` 不改 `info[*_SAFETY_COST_*]` | ❌ | 永不污染（决议 11 CSV 规格已验证，仍正确） |
| **ep_return** | Σ 流经 wrapper 的 reward 标量；`RobustWrapper` 仅在 reward 通道对它加噪（S4） | — | **仅 reward 噪声污染** |
| reward 分解列 | `MetricsWrapper` 读加噪前 `reward_terms`（§0.6.5(c)） | ❌ | 永远 clean；其和 = clean return = `ep_return_true` |

### 14.2 核心表：指标 × 噪声通道 → 是否**污染-as-measurement**

| 指标 | action 噪声 | obs 噪声 | reward 噪声 |
|---|---|---|---|
| **F1 / precision / recall** | ❌ 不污染 | ❌ 不污染 | ❌ 不污染 |
| **ep_cost** | ❌ 不污染 | ❌ 不污染 | ❌ 不污染 |
| **ep_return** | ❌ 不污染（实收 = 被扰动轨迹真 return） | ❌ 不污染 | ✅ **污染**（= 真值 + Σ噪声） |

**唯一被污染-as-measurement 的组合：reward 噪声下的 `ep_return`。** 其余任何组合要么不受影响、要么诚实反映被扰动轨迹。

**值变化维度（非污染，补充）**：
- action / obs 噪声改变轨迹 → F1 / cost / ep_return 数值都会变（诚实，是被扰动轨迹的真值）。
- reward 噪声经 `obs["reward"]`（OAR + Method 6，`action_reward_observation=True` 默认）进入策略 → 也改变轨迹 → F1 / cost 数值会变（诚实）；ep_return 则**既随轨迹变、又被加噪污染**。

### 14.3 Logger 覆盖表：哪个 logger 有哪些指标（新验证）

| Logger | 框架 | ep_return | ep_cost | F1 / prec / recall | 分解列 | S6 列 |
|---|---|---|---|---|---|---|
| **training rollout** `train_iteration_summary.csv` | SB3 | ✓（mean±std） | ✓ | ✓ | ✓ | ✗ |
| **training rollout** `progress.csv` | OmniSafe | ✓ EpRet | ✓ EpCost | ❌（`Evaluator` 不 surface terminal info，`run_omnisafe_template.py:74-76`） | ❌ | ✗ |
| **deterministic eval** = `SafeRecordEpisodeStatistics` CSV | SB3 | ✓ | ✓ | ✓ | ✓ | ✓ |
| **replay CSV** `examples/checkpoint_replay_eval.py` | OmniSafe | ✓ | ✓ | ✓（自写 rollout 读 terminal info） | ✓ | ✓（schema-lock；safety→nominal） |

**两处新发现 / 更正**：
1. **SB3 training CSV 有分解列**（`TrainIterationSummaryCallback` 读 terminal info：`ep_f1_mean` / `ep_precision_mean` / `episode_energy_reward_mean` …）→ 更正之前"training rollout 无分解列"的说法；SB3 训练侧 clean return 也可由分解列之和恢复。
2. **OmniSafe training 无 F1/precision/recall**：`Evaluator.evaluate()` 只返回 (rewards, costs)、不 surface terminal info；F1 只在 `_final_eval` / `checkpoint_replay`（自写 rollout 读 terminal info）里才有。
3. 只有**两个 eval CSV**（SafeRecord + replay）有 S6 的 `ep_return_true` / `eval_noise_scale` / `ep_noise_*`。`SafeRecordEpisodeStatistics` **就是** deterministic eval CSV 的 writer（表中同一行）。

### 14.4 逐 logger × 逐指标的具体影响

"是否污染"由**指标来源**决定、**与 logger 无关**。所以在**任一** logger 里：

- 它记录的 **F1 / precision / recall / ep_cost 永远是真实轨迹的真值**——不被任何噪声通道污染-as-measurement；数值会随扰动轨迹变化（action/obs/reward 都可能改轨迹），那是诚实测量。
- 它记录的 **ep_return**：
  - **reward 任务** → 是 **noised（实收带噪）**，被污染-as-measurement；真值另有来源：eval CSV 的 `ep_return_true` 列 / SB3 training 的分解列之和 / F1（clean headline）。OmniSafe training 的 EpRet 是 noised 且无分解列 → 该处真值只能靠 F1（而 OmniSafe training 又无 F1，故 OmniSafe **训练**侧看真实表现要靠 replay CSV / `_final_eval`）。
  - **action / obs 任务** → 是被扰动轨迹的**真 return**（reward 标量未被污染），可直接用。

### 14.5 一句话总结

**除了"reward 噪声污染 `ep_return`"这一个组合，所有 logger 里的 `ep_cost` / `F1` / `precision` / `recall` 都恒为真实轨迹的真值（不被任何噪声污染，只会随扰动轨迹诚实变化）。`ep_return` 在 reward 任务下是实收带噪值，其真值另有来源（`ep_return_true` / 分解列之和 / F1）。** F1 是全程 noise-immune 的 headline。

---

## 15. 接入新算法库做 robust baseline —— checklist（决议 11 接库契约的落地）

未来接入新 RL 库（CleanRL / RLlib / TorchRL / Mava …）跑 robust baseline 时要做什么。

### 15.1 训练侧：无需特殊处理

`omnipiano.make(env_id, mode="train")`，正常训练即可。robust 噪声在 wrapper 链里**逐 step 自动注入（库无关）**；库只管 `step()` + 用返回的（带噪）reward 训练。matched 训练由 env 保证。

### 15.2 Eval 侧：eval env 必须经 `make(mode="eval")` 构造

- 只有经 `omnipiano.make(env_id, mode="eval")` 才**自动继承 eval 语义**（按 `eval_noise_scale` 缩放，matched 默认，决议 10/11）。
- **反例**：像 OmniSafe `checkpoint_replay` 那样用框架自己的 Evaluator 从训练 config 重建 env（**绕过 make**）→ **不继承** eval 语义（§11 deferred 记的缺口）。新库要么走 `make(mode="eval")`，要么显式复刻缩放。
- sweep：用 `make(env_id, mode="eval", _eval_noise_scale_override=scale)`（Phase 1 实现的私有 kwarg，§12 决议 7）遍历 scale；或注册 per-scale env。

### 15.3 Eval 记录 —— 两条路

**A（最省事，推荐）**：给 eval env 传 `log_dir` → make() 自动挂 `SafeRecordEpisodeStatistics` → **所有列（含 `ep_return_true` / `eval_noise_scale` / `ep_noise_*`）免费写好**，新库啥都不用做。
```python
env = omnipiano.make(env_id, mode="eval", log_dir=my_dir)   # SafeRecord 自动挂
```
**B（自写 eval CSV）**：若库要写自己的 eval CSV，必须（schema-lock 到 SafeRecord 的 24 列，便于统一画图）：
- 从 terminal info 读：`episode_task/f1`（+ precision/recall/sustain_*）、`episode_safety/cost_total` + `violations`、`episode_task/*_reward`（6 个分解列）。
- 记录：`ep_return`（实收）、**`ep_return_true` = Σ 分解列**（clean headline）、`eval_noise_scale`、(可选) `ep_noise_{action,obs,reward}` = Σ per-step `info["robust/noise_*"]`。
- 交叉校验：`ep_return − ep_return_true == ep_noise_reward`。

### 15.4 至少要加的列（回答"我理解至少需要加几个 column + ep_return_true"）

✅ 你理解对了。相对一个"只记 `ep_return`/`ep_cost`"的朴素 logger，做 robust baseline 在 **deterministic eval / ckpt eval CSV 至少要加**：

| 列 | 为什么必须 |
|---|---|
| **`ep_return_true`**（= 分解列之和） | reward 任务的真实表现；`ep_return` 带噪时的干净值。**必加** |
| **`eval_noise_scale`** | robustness 曲线定位——没它无法判断一行属于曲线哪个点。**必加** |
| `ep_noise_{action,obs,reward}` | dev tripwire，确认噪声真注入了。**建议加** |

前提：F1/precision/recall/cost/分解列 也得从 info key 读全（否则连 `ep_return_true` 都算不出）。

### 15.5 Headline + gotchas

- **headline 用 F1**（noise-immune、读物理、与库/噪声无关）；reward 任务的 return 用 `ep_return_true`。
- env **不产 `info["episode"]`**——库若依赖它需自挂 `RecordEpisodeStatistics` / `Monitor`。
- 向量化 env 下 terminal info 在 `info["final_info"]`。
- 真值（F1/cost/分解）是**环境层量**（`info` key，见 §14），库无关——写不写进 CSV 取决于库自己的 logger。**用路 A（挂 SafeRecord）最稳。**

一句话：**训练侧啥都不用改（噪声在 env 里）；eval 侧只要 (1) 经 `make(mode="eval")` 建 env、(2) 挂 `SafeRecordEpisodeStatistics`（或自写时补齐 `ep_return_true` + `eval_noise_scale` + noise 列并 schema-lock），headline 看 F1，就能产出 robust-comparable 的结果。**

