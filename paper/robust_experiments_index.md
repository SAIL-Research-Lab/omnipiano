# Robust 实验索引(实验台账)

**用途**:一处查全 robust 腿**做过什么、结果在哪、图在哪**,便于设计后续实验与整理材料。
**与其他文档的分工**:

| 文件 | 内容 |
|---|---|
| **本文件** | 实验台账:跑过哪些、路径、状态、缺口 |
| `paper/robust_notes.md` | 写作用:自由度设计、与 RG 对比、威胁模型、结果表、finding |
| `paper/claims.md` §C4 | paper-ready 的 claim 措辞 |
| `omnipiano/docs/robust_task_design.md` | 开发设计文档(实施细节,写作时不必读) |

**统一协议**:PPO(纯 PPO;robust 任务无 cost,不跑 PPOLag)· seed=0 · 5M env steps · n_envs=16 · γ=0.8 · eval_freq=50,000 · **最终策略 + 10 集 deterministic matched eval**(`eval_noise_scale=1.0`)。曲目 **ClairDeLune**(588 步/集)。库:**SB3**。

**语义版本**:全部为 **2026-07-21 新语义**(`obs["action"]` = 加噪后实际执行动作,硬件磨损威胁模型)。旧语义的 A 通道 run 已删除重跑。

---

## 1. 已完成实验(19 个 run)

路径前缀统一为 `examples/logs/ppo_sb3_baseline_clairdelune_<key>_seed0_1/`。
F1 为**最终策略 10 集 matched eval 均值 ± std**(shift 类确定性,std 恒为 0)。

### 1.1 基线

| key | env id | F1 | 说明 |
|---|---|---|---|
| `clean` | `Clean-v0` | **0.626 ± 0.000** | 全表参照 |
| `clean_nooar` | `Clean-NoOAR-v0` | 0.520 ± 0.000 | OAR 消融(obs 1176→1130) |

### 1.2 单通道 × 三分布(v1 主矩阵的最高档)

| key | env id | F1 | ΔF1 |
|---|---|---|---|
| `a_gauss_p15` | `A-Gauss-P15-v0` | 0.267 ± 0.035 | −57% |
| `a_uniform_p15` | `A-Uniform-P15-v0` | 0.491 ± 0.051 | −22% |
| `a_shift_p15` | `A-Shift-P15-v0` | 0.615 ± 0.000 | −2% |
| `o_gauss_p15` | `O-Gauss-P15-v0` | 0.562 ± 0.015 | −10% |
| `o_uniform_p15` | `O-Uniform-P15-v0` | 0.616 ± 0.032 | −2% |
| `o_shift_p15` | `O-Shift-P15-v0` | 0.539 ± 0.000 | −14% |
| `r_gauss_p50` | `R-Gauss-P50-v0` | 0.109 ± 0.004 | −83% |
| `r_uniform_p50` | `R-Uniform-P50-v0` | 0.327 ± 0.029 | −48% |
| `r_shift_p50` | `R-Shift-P50-v0` | 0.589 ± 0.000 | −6% |

### 1.3 负向 shift 对照(诊断,不属 v1 矩阵)

| key | env id | F1 | 用途 |
|---|---|---|---|
| `a_shift_n15` | `A-Shift-N15-v0` | 0.545 ± 0.000 | action 符号对照 |
| `o_shift_n15` | `O-Shift-N15-v0` | **0.677 ± 0.000** | obs 符号对照 **(高于 clean)** + OR 分解分量 |
| `r_shift_n50` | `R-Shift-N50-v0` | 0.493 ± 0.000 | reward 符号对照 + OR 分解分量 |

### 1.4 组合通道(同分布,每通道独立 level)

| key | env id | F1 | 两个分量 |
|---|---|---|---|
| `ao_gauss_p15` | `AO-Gauss-P15-v0` | 0.177 ± 0.024 | A .267 / O .562 → **超可加** |
| `ar_uniform_a15_r50` | `AR-Uniform-A15-R50-v0` | 0.232 ± 0.031 | A .491 / R .327 → **超可加** |
| `or_shift_on15_rn50` | `OR-Shift-ON15-RN50-v0` | 0.672 ± 0.000 | O-N15 .677 / R-N50 .493 → **次可加** |

### 1.5 进行中

| key | env id | 目的 |
|---|---|---|
| `r_shift_p50_nooar` | `R-Shift-P50-NoOAR-v0` | **C4d-i 判定实验**:切断 OAR 观测通路,看 reward shift 的效应是否消失 |
| `r_shift_n50_nooar` | `R-Shift-N50-NoOAR-v0` | 同上(负向臂) |

> 判读:对照基线是 **Clean-NoOAR 0.520**(不是 Clean 0.626)。效应消失 ⇒ reward 噪声主要经**观测通路**起作用;效应保留 ⇒ 是 **critic 学习动态**。

---

## 2. 数据与产物路径

| 类型 | 路径 | 说明 |
|---|---|---|
| **训练日志** | `examples/logs/ppo_sb3_baseline_clairdelune_<key>_seed0_1/` | 每个 run 一个目录 |
| ├ 训练遥测 | `.../progress.csv` | SB3 原生(`rollout/ep_rew_mean` 等)**无 F1** |
| ├ 周期性评测 | `.../eval_episode_metrics_*.csv` | 每 50k 步一集,**含 F1/precision/recall/cost/reward 分解/噪声量** |
| ├ 最终评测 | `.../eval_summary.json` | 训练结束后的评测 + 完整生效配置 |
| └ 模型 | `.../final_model.zip`, `best_model.zip` | 可用于重放评测 |
| **10 集统一评测**(图的数据源) | `paper/data/final_eval10/<key>/` | 最终策略 + 10 集 matched eval,`ep_f1`/`ep_return` 等 24 列 |
| **正式图** | `paper/figures/fig{1,2,3,4}_*.{pdf,png}` | 见下表 |
| **绘图脚本** | `paper/make_robust_figures.py` | 可复现;编码约定写在文件头 |
| **等价性 gate 工具** | `omnipiano/tests/tools_robust_fingerprint.py` | 重构前后比对噪声指纹,证明"无需重跑" |

### 图像清单

| 图 | 路径 | 内容 | 支撑 |
|---|---|---|---|
| Fig 1 | `paper/figures/fig1_main_f1.pdf` | 3 通道 × 3 分布 F1(mean±std)+ clean 线 | C4a / C4b |
| Fig 2 | `paper/figures/fig2_learning_curves.pdf` | 三分面学习曲线 | 退化贯穿训练 |
| Fig 3 | `paper/figures/fig3_compound.pdf` | 组合扰动 vs **正确方向**的分量 | **C4f 超可加 vs 次可加** |
| Fig 4 | `paper/figures/fig4_conservatism.pdf` | precision/recall 哑铃图 | C4c |

> 另有 N-hand 诊断图(非 robust):`figW_winterwind_nhand.*`、`figK_kiev_nhand.*`,脚本 `paper/make_{winterwind,kiev}_figures.py`。

---

## 3. 主要发现速查(详述见 `robust_notes.md` §5)

1. **分布难度排序**:action/reward 上 Shift > Uniform > Gauss;**observation 通道反转**(Shift 最差 0.539 < Gauss 0.562 < Uniform 0.616)。
2. **通道敏感度**:Action ≈ Reward ≫ Observation(最高档高斯:−57% / −83% / −10%)。
3. **保守化**:precision 恒在 0.92–0.99,recall 崩塌;F1 的跌幅几乎全来自 recall。
4. **reward 噪声隐形破坏学习**:R-Gauss-P50 的 F1 只剩 0.109(−83%),实收 return 却只低 9%。**没有 noise-immune 指标就看不见**(C2 最强案例)。
5. **恒定偏置符号决定成败,且有益方向因通道而异**:action ±0.15 → −2% / −13%;**obs ±0.15 → −14% / +8%(负向高于 clean)**;reward ±0.50 → −6% / −21%。
6. **组合律因噪声族而异**:随机噪声**超可加**(组合低于两个分量),恒定偏置**次可加**(组合≈较好分量,有害分量被掩盖)。
7. **⚠️ 待解释**:episode 定长 588 步 ⇒ 恒定 reward 偏置在 MDP 层应策略不变,实测却两向都显著伤害 → 判定实验进行中(§1.5)。
8. **shift 类任务评测方差恒为 0**(常数偏置不抽随机数 + 初始姿态确定),随机噪声任务 std 0.004–0.051 → 这正当化了统一的 10 集协议。

---

## 4. 已知缺口 / 后续实验候选

| 缺口 | 说明 | 成本 |
|---|---|---|
| **单 seed** | 全部 seed=0。**当前最大的统计弱点**,claim 5/6 尤其需要多 seed | 每条 claim ×2 seed ≈ 2 run/条 |
| **仅最高档** | 只有 A/O 的 P15、R 的 P50;**中间档 P05/P10/P30 未跑** | 12 run 可填满难度-F1 曲线 |
| **无 robustness 曲线** | 只有 matched(scale=1.0);scale sweep {0,0.5,1,2,4} 未跑 | **不需重训**,`robust_eval_sweep.py` 重放即可 |
| **威胁模型对照不完整** | 新旧语义对比仅 shift 通道严格可比(两侧确定性);gauss/uniform 的旧值是单集口径 | 需重训旧语义才能扩展 |
| **混分布任务未注册** | 每通道不同分布的能力已实现(2026-07-21),但一个任务都没注册 | 注册免费;命名方案待定 |
| **N-hand × robust 未开始** | 形态阶梯与 robust 的交叉;`ep_noise_action_l2` 可报出各形态实际噪声量 | 见 `robust_notes.md` 讨论 |

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

# robustness 曲线(同一策略扫多个 scale,无需重训)
MUJOCO_GL=egl python examples/robust_eval_sweep.py --ckpt <final_model.zip> \
    --env <robust id> --algo ppo --scales 0.0 0.5 1.0 2.0 4.0 --num-eval-eps 10

# 重画全部 robust 图
python paper/make_robust_figures.py
```

**用户侧设置 eval 噪声倍率**(2026-07-21 起为公开参数,无需注册新任务):
```python
env = omnipiano.make(robust_id, mode="eval", eval_noise_scale=2.0, log_dir=d)
```
