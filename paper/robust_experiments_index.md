# Robust 实验索引(实验台账)

**用途**:一处查全 robust 腿**做过什么、结果在哪、图在哪**,便于设计后续实验与整理材料。

| 文件 | 内容 |
|---|---|
| **本文件** | 实验台账:跑过哪些、路径、状态、缺口 |
| `paper/experiment_plan.md` | 待跑实验的计划与队列 |
| `paper/robust_notes.md` | 写作用:自由度设计、与 RG 对比、威胁模型、finding |
| `paper/claims.md` §C4 | paper-ready 的 claim 措辞 |
| `omnipiano/docs/robust_task_design.md` | 开发设计文档(实施细节,写作时不必读) |

**统一协议**:PPO(纯 PPO;robust 任务无 cost,不跑 PPOLag)· seed=0 · 5M env steps · n_envs=16 · γ=0.8 · eval_freq=50,000 · **最终策略 + 10 集 deterministic matched eval**(`eval_noise_scale=1.0`)。曲目 **ClairDeLune**(588 步/集)。库:**SB3**。

**语义版本**:全部为 **2026-07-21 新语义**(`obs["action"]` = 加噪后实际执行动作,硬件磨损威胁模型)。旧语义的 A 通道 run 已删除重跑。

---

## 0. 指标口径

**主指标 = `ep_return`(episode 回报),F1 为次要指标但始终记录并绘制**(2026-08-28 决定)。

用的是 **`ep_return`,即智能体实际收到的加噪回报**,不是 `ep_return_true`。理由:现实部署中无人能观测未受扰动的奖励,读者据以判断稳健性的应当是智能体真正经历的那条曲线。

> ### ⚠️ 唯一的例外:**reward-shift 任务的 `ep_return` 不能和 clean 比**
>
> reward 通道的常数偏置直接加在**被测量的那个量**上。实测(精确到小数):
>
> | 任务 | `ep_return` | `ep_return_true` | 差值 | shift × 588 |
> |---|---|---|---|---|
> | `r_shift_p50` | 2140.6 | 1846.6 | **+294.0** | **+294.0** |
> | `r_shift_n50` | 1480.7 | 1774.7 | **−294.0** | **−294.0** |
> | `r_gauss_p50` | 1692.2 | 1689.6 | +2.6 | 0(零均值) |
> | `r_uniform_p50` | 1774.1 | 1775.2 | −1.1 | 0(零均值) |
>
> 所以表里 `r_shift_p50` 的 **return +16% 完全是加上去的那个常数,不是性能提升** —— 它的
> `ep_return_true` 是 1846.6,对 clean 的 1850.6 几乎没变(−0.2%)。
>
> **零均值的 reward 噪声(gauss/uniform)不受影响**(差值 2.6 / −1.1),obs 与 action 通道
> 也不受影响(噪声不碰奖励信号)。受影响的只有 **reward × shift** 这两格,表中已标 ⚠️。

---

## 1. 已完成实验(19 个 run)

路径前缀统一为 `examples/logs/ppo_sb3_baseline_clairdelune_<key>_seed0_1/`。
数值为**最终策略 10 集 matched eval 的 mean ± std**(shift 类确定性,std 恒为 0)。

### 1.1 基线

| key | env id | **return** | Δret | F1 | ΔF1 |
|---|---|---|---|---|---|
| `clean` | `Clean-v0` | **1851 ± 0** | — | 0.626 ± 0.000 | — |
| `clean_nooar` | `Clean-NoOAR-v0` | **1745 ± 0** | −6% | 0.520 ± 0.000 | −17% |

### 1.2 单通道 × 三分布(v1 主矩阵的最高档)

| key | env id | **return** | Δret | F1 | ΔF1 |
|---|---|---|---|---|---|
| `a_gauss_p15` | `A-Gauss-P15-v0` | **1671 ± 10** | −10% | 0.267 ± 0.035 | −57% |
| `a_uniform_p15` | `A-Uniform-P15-v0` | **1802 ± 24** | −3% | 0.491 ± 0.051 | −22% |
| `a_shift_p15` | `A-Shift-P15-v0` | **1859 ± 0** | +0% | 0.615 ± 0.000 | −2% |
| `o_gauss_p15` | `O-Gauss-P15-v0` | **1832 ± 11** | −1% | 0.562 ± 0.015 | −10% |
| `o_uniform_p15` | `O-Uniform-P15-v0` | **1849 ± 17** | −0% | 0.616 ± 0.032 | −2% |
| `o_shift_p15` | `O-Shift-P15-v0` | **1832 ± 0** | −1% | 0.539 ± 0.000 | −14% |
| `r_gauss_p50` | `R-Gauss-P50-v0` | **1692 ± 12** | −9% | 0.109 ± 0.004 | −83% |
| `r_uniform_p50` | `R-Uniform-P50-v0` | **1774 ± 8** | −4% | 0.327 ± 0.029 | −48% |
| `r_shift_p50` ⚠️ | `R-Shift-P50-v0` | **2141 ± 0** | ~~+16%~~ | 0.589 ± 0.000 | −6% |

⚠️ `r_shift_p50` 的 Δret 无意义,见 §0。其 `ep_return_true` = 1846.6(−0.2%)。

### 1.3 负向 shift 对照(诊断,**不属 v1 主矩阵**)

| key | env id | **return** | Δret | F1 | ΔF1 | 用途 |
|---|---|---|---|---|---|---|
| `a_shift_n15` | `A-Shift-N15-v0` | **1698 ± 0** | −8% | 0.545 ± 0.000 | −13% | action 符号对照 |
| `o_shift_n15` | `O-Shift-N15-v0` | **1887 ± 0** | +2% | 0.677 ± 0.000 | **+8%** | obs 符号对照 + OR 分量 |
| `r_shift_n50` ⚠️ | `R-Shift-N50-v0` | **1481 ± 0** | ~~−20%~~ | 0.493 ± 0.000 | −21% | reward 符号对照 + OR 分量 |

⚠️ `r_shift_n50` 同上,`ep_return_true` = 1774.7(−4.1%)。

> **这三个 N 后缀 run 绝不能进主矩阵。** 它们和 `o_shift_p15` / `r_shift_p50` 共享
> (通道, 分布) 组合,2026-08-28 之前 `make_robust_figures.py` 用字典推导建矩阵,
> 导致它们**静默覆盖**了正向 run —— fig1 的 (Observation, Shift) 和 (Reward, Shift)
> 两格一直画的是 N 对照。现已改为显式 `MATRIX` 字面量。

### 1.4 组合通道(同分布,每通道独立 level)

| key | env id | **return** | Δret | F1 | ΔF1 | 两个分量(F1) |
|---|---|---|---|---|---|---|
| `ao_gauss_p15` | `AO-Gauss-P15-v0` | **1624 ± 13** | −12% | 0.177 ± 0.024 | −72% | A .267 / O .562 → **超可加** |
| `ar_uniform_a15_r50` | `AR-Uniform-A15-R50-v0` | **1717 ± 19** | −7% | 0.232 ± 0.031 | −63% | A .491 / R .327 → **超可加** |
| `or_shift_on15_rn50` | `OR-Shift-ON15-RN50-v0` | **1592 ± 0** | −14% | 0.672 ± 0.000 | +7% | O-N15 .677 / R-N50 .493 → **次可加** |

### 1.5 C4d-i 判定实验(2026-07-21)

切断 `obs["reward"]` 输入通路(`action_reward_observation=False`),reward 标量仍加噪 →
分离"观测通路"与"学习信号通路"。**对照基线是 Clean-NoOAR,不是 Clean。**

| key | env id | F1 | Δ vs Clean-NoOAR | 同任务 OAR 开时的 Δ |
|---|---|---|---|---|
| `r_shift_p50_nooar` | `R-Shift-P50-NoOAR-v0` | 0.472 | −9% | −6%(**伤害保留**) |
| `r_shift_n50_nooar` | `R-Shift-N50-NoOAR-v0` | **0.558** | **+7%** | −21%(**伤害消失**) |

**结论**:负向 shift 的大幅伤害是**纯观测通路效应**;正向 shift 的小幅伤害与 OAR 无关。

---

## 2. 数据与产物路径

| 类型 | 路径 | 说明 |
|---|---|---|
| **训练日志** | `examples/logs/ppo_sb3_baseline_clairdelune_<key>_seed0_1/` | 每个 run 一个目录 |
| ├ **训练 rollout** | `.../progress.csv` → `rollout/ep_rew_mean` | SB3 原生。**有 reward,无 F1** |
| ├ 周期性评测 | `.../eval_episode_metrics_*.csv` | 每 50k 步 1 集,**含 `ep_return` / `ep_f1` / precision / recall / 噪声量** |
| ├ 最终评测 | `.../eval_summary.json` | 训练结束后的评测 + 完整生效配置 |
| └ 模型 | `.../final_model.zip`, `best_model.zip` | 可用于重放评测 |
| **10 集统一评测** | `paper/data/final_eval10/<key>/` — **不在库中** | 最终策略 + 10 集 matched eval;命令见文末,可重跑 |
| **正式图** | `paper/figures/` | 见下表 |
| **绘图脚本** | `paper/make_robust_figures.py` | 可复现;编码约定写在文件头 |
| **溯源自查** | `examples/diagnostics/provenance_check.py` | γ / 完成度 / 重放三道检查 |

### 图像清单(全部 reward + F1 双指标)

| 图 | 路径 | 内容 | 数据源 | 在库 |
|---|---|---|---|---|
| Fig 2 | `fig2_learning_curves.{pdf,png}` | 上排 return(**训练 rollout 虚线 + 周期性评测实线**)、下排 F1(仅评测) | `examples/logs/` | ✅ |
| Fig 1 | `fig1_main_reward_f1.{pdf,png}` | 上排 return、下排 F1;3 通道 × 3 分布 + clean 参考线 | `final_eval10/` | ❌ |
| Fig 3 | `fig3_compound.{pdf,png}` | 组合扰动 vs 正确方向的分量,双指标 | `final_eval10/` | ❌ |
| Fig 4 | `fig4_conservatism.{pdf,png}` | precision/recall 哑铃图(F1 的分解,单指标) | `final_eval10/` | ❌ |

> **F1 行没有训练 rollout 曲线**:SB3 和 OmniSafe 在采样阶段都不计算 F1,
> 训练遥测里只有 reward。这是已知缺口,不是遗漏。

> **Fig 1 / 3 / 4 及其数据不在库中(2026-08-29 移除)。** 三张图画的都是**最终策略**
> 的 10 集 matched eval,数据源 `paper/data/final_eval10/`。是否在论文中报告最终策略
> 的评测与对比,由写 robust 章节的同学决定 —— 本台账不预设结论。
>
> **数字不受影响**:本文件 §2/§3 的表格值来自那次评测,已逐条记录在此;`robust_notes.md`
> §5 同。删的是 CSV,不是数字。
>
> **需要时如何取回**:15 个 run 各跑一次文末命令,数据落回 `paper/data/final_eval10/<key>/`,
> 再跑 `make_robust_figures.py` 即自动重新生成三张图(脚本已做存在性判断,数据缺失时
> 只画 Fig 2 并打印跳过原因)。或直接从 git 历史取:`git show 8255ef2:paper/data/final_eval10/...`。

---

## 3. 主要发现速查

按主指标(return)重新排序;F1 作为佐证。

1. **通道敏感度(return)**:Action ≈ Reward ≫ Observation。最高档高斯下
   return −10% / −9% / −1%,F1 −57% / −83% / −10%。
2. **F1 的跌幅比 return 大一个量级** —— 这本身是个 finding:
   return 只掉 9% 而 F1 掉 83%(`r_gauss_p50`)。**没有 noise-immune 指标就看不见这种破坏**,
   是 C2 最强的案例,也正是 F1 必须继续报告的理由。
3. **分布难度排序**:action/reward 上 Shift > Uniform > Gauss;**observation 通道反转**
   (Shift 最差 0.539 < Gauss 0.562 < Uniform 0.616)。
4. **保守化**:precision 恒在 0.92–0.99,recall 崩塌;F1 的跌幅几乎全来自 recall。
5. **恒定偏置符号决定成败,且有益方向因通道而异**:action ±0.15 → −2% / −13%;
   **obs ±0.15 → −14% / +8%(负向高于 clean)**;reward ±0.50 → −6% / −21%(按 F1)。
6. **组合律因噪声族而异**:随机噪声**超可加**,恒定偏置**次可加**。
7. **恒定 reward 偏置的伤害来自观测通路**:切断 `obs["reward"]` 后负向的 −21% 完全消失。
8. **评测方差**(10 集):clean / shift 恒为 0;随机噪声 **return CV 0.46–1.34%,F1 CV 2.6–13.5%**。
   → 按主指标 return 看,**1 集已足够**;F1 的方差大一个量级,多集主要是为它服务的。

---

## 4. 已知缺口 / 后续实验候选

| 缺口 | 说明 | 成本 |
|---|---|---|
| **单 seed** | 全部 seed=0。**当前最大的统计弱点** | 每条 claim ×2 seed |
| **仅最高档** | 只有 A/O 的 P15、R 的 P50;中间档 P05/P10/P30 未跑 | 12 run |
| **无 robustness 曲线** | scale sweep {0,0.5,1,2,4} 未跑 | **不需重训**,重放即可 |
| **训练 rollout 无 F1** | SB3/OmniSafe 都不在采样阶段算 F1 | 需改 wrapper |
| **混分布任务未注册** | 能力已实现(2026-07-21),命名方案待定 | 注册免费 |
| **N-hand × robust 未开始** | 形态阶梯与 robust 的交叉 | 见 `robust_notes.md` |

**最便宜的高价值下一步**:robustness 曲线(纯重放,零训练)。

---

## 5. 常用命令

```bash
# 训练一个 robust 任务
MUJOCO_GL=egl python examples/run_sb3_baseline.py --algo ppo \
    --env OmniPiano-ClairDeLune-A-Gauss-P15-v0 --seed 0

# 最终策略 + 10 集 matched eval(本表数字的来源)
MUJOCO_GL=egl python examples/robust_eval_sweep.py \
    --ckpt examples/logs/ppo_sb3_baseline_clairdelune_a_gauss_p15_seed0_1/final_model.zip \
    --env OmniPiano-ClairDeLune-A-Gauss-P15-v0 --algo ppo \
    --scales 1.0 --num-eval-eps 10 --train-seed 0 --out-dir paper/data/final_eval10/a_gauss_p15

# 重画全部 robust 图(同时打印 return/F1 汇总表)
python paper/make_robust_figures.py

# 数据溯源自查
python examples/diagnostics/provenance_check.py
```

**用户侧设置 eval 噪声倍率**(2026-07-21 起为公开参数):
```python
env = omnipiano.make(robust_id, mode="eval", eval_noise_scale=2.0, log_dir=d)
```
