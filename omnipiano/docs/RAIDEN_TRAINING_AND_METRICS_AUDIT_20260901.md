# OmniPiano RAIDEN 训练与指标审计总结

> 状态：实验状态与质量审计快照，不替代冻结的训练协议
>
> 数据截止：2026-09-01 23:40 JST
>
> 代码快照：`20260828_marl_robust_v1_protocol_1_1`
>
> 快照 SHA-256：`ad0645182101902c645db316cd13ef5db67b8fc1e49e0907cbae97355c698b83`
>
> 环境协议：1.1；当前正式指标：Metrics-v2
>
> 权威数据源：RAIDEN 本地 immutable artifacts、submission manifests 与 SGE accounting；W&B 仅作为镜像和可视化入口

## 0. 一页结论

截至本快照，四组被跟踪的实验共 166 个 run：112 个完成、9 个运行、45 个排队，未发现已知训练失败。Etude-12、Song Sweep OT 和 MARL-Core 已完成当前已提交矩阵；Robustness-Core 尚未完成。

当前结果不是“全部不能用”，而应分层处理：

- 模型权重、训练轨迹、同任务的 team return、Metrics-v2 `note_event_f1` 和 `key_time_micro_f1` 基本可用。
- contact-force attribution 存在已经确认的评测后处理 bug。它污染手/agent 归属和协作指标，但不影响训练、模型权重、team return 或全局 note-event F1。
- 部分 collision-force trace 含 non-finite 值，相关 collision 指标不能用于正式结论。
- SB3 PPO 的 loss 是有限值，但更新幅度长期异常偏大；它可以保留为 `SB3-default PPO` baseline，不能描述为“健康收敛的 PPO”。SAC 暂未显示同类系统性数值异常。
- IPPO/MAPPO 没有记录 learner loss、KL、entropy、gradient norm 等核心诊断，因此目前只能审计表现曲线，不能审计优化器健康度。
- reward 与 F1 的整体正相关很大程度上受共同训练进度驱动；局部增量相关弱、存在 `F1=0` 但 reward 较高、算法排序翻转和跨曲长度混淆。这不是单纯代码 bug，而是 OmniPiano benchmark 应量化的核心科学问题。
- 现有 paper 文档中“episode return 为主指标”的口径与本次证据存在冲突。建议 leaderboard 以独立的音乐表现指标为 headline，return 作为训练信号和同任务辅助诊断；在团队统一口径前，不应使用 raw return 做跨曲排名或 reward-noise 的 headline 结论。

一句话定位建议：

> OmniPiano 不只测试智能体能否获得高 RL return，而是系统检验高 return 是否真正转化为音乐正确性、多手协作、物理效率与扰动下的鲁棒性。

## 1. 本次审计回答什么

本文件回答五个问题：

1. RAIDEN 上已经完成、正在运行和排队的实验是什么；
2. 当前 loss、reward、F1 和 artifact 是否存在训练级数据损坏；
3. 哪些异常是工程 bug，哪些是 benchmark 应研究的 reward–performance mismatch；
4. 哪些结果可直接保留、哪些只需重评、哪些必须补跑；
5. 在论文和 leaderboard 中应该怎样使用这些指标。

本文件是带时间戳的审计快照。动态任务状态以后应从 manifests、`qstat`、`qacct` 和本地 artifact 重新生成，不能把本节数字当成永久台账。

## 2. 实验矩阵与实时状态

| 矩阵 | 当前设计 | 完成 | 运行 | 排队 | 已知失败 | 目前可用于什么 |
|---|---:|---:|---:|---:|---:|---|
| Etude-12 | 12 曲 × PPO/SAC × 2 seeds = 48 | 48 | 0 | 0 | 0 | 两 seed pilot；Reward–F1 审计 |
| Song Sweep OT | 11 env × PPO/SAC × seed 0 = 22 | 22 | 0 | 0 | 0 | 单 seed、legacy metric 的探索性结果 |
| MARL-Core | 4 曲 × Mono PPO/IPPO/MAPPO × 3 seeds = 36 | 36 | 0 | 0 | 0 | 三算法主矩阵；协作归属指标需修复后重算 |
| Robustness-Core | 10 task × PPO/SAC × 3 seeds = 60 | 6 | 9 | 45 | 0 | 仍在训练，不能生成最终 robustness claim |
| **合计** | **166** | **112** | **9** | **45** | **0** | — |

额外说明：

- MARL 最后完成的 run 为 GreatKiev MAPPO seed 2，job `18467696`，2026-09-01 23:07 JST 结束，`failed=0`、`exit_status=0`。
- Robustness 的 9 个运行中任务和 45 个排队任务会继续变化。
- 两个长时 W&B mirror job 和 hold 状态的 post-processing job 不计入 166 个训练 run。
- “完成”表示训练 job 和预期 artifact 已完成，不等于所有指标都已经论文可用。

### 2.1 各矩阵的统计资格

| 矩阵 | 当前资格 | 尚缺内容 |
|---|---|---|
| Etude-12 | pilot | 正式 3-seed 口径还缺 seed 2，共 24 个 PPO/SAC run |
| Song Sweep OT | descriptive/exploratory | 仅 seed 0；需切换到 Metrics-v2 并增加 seeds 后才能正式比较 |
| MARL-Core | 训练矩阵完整 | final eval 目前每 seed 仅 1 个 deterministic episode；协作 attribution 需修复 |
| Robustness-Core | incomplete | 60-run 训练矩阵和后续 10 episodes/scale sweep 均需完成 |

## 3. 数据完整性和 provenance

### 3.1 已通过的检查

- 已完成正式任务的训练标量未发现系统性 OOM、`Traceback`、`Killed`、segfault 或 loss/reward/F1 数值爆炸。
- 已检查任务的 SGE accounting 为正常退出，artifact gate、预期 checkpoint、evaluation 和 trace 文件存在。
- CSV/JSON 中出现的 `NaN` 并不都代表训练崩溃。例如没有 true-positive 时，onset error 或 work-per-correct-event 在数学上未定义；strict JSON 应将这种条件未定义的可选量序列化为 `null`，按有效条件样本聚合，并同时报告有效 `n` 和 null rate，而不是填 0。required headline metric 或异常产生的 non-finite 必须计入 failure/missingness，不能静默排除。
- 冻结快照 hash 已写入 submission manifest，可以把每个结果追溯到同一份代码与协议。

### 3.2 本地权威结果根目录

```text
/data/giil/zhangzy/omnipiano/runs/etude12_reward_f1_v4
/data/giil/zhangzy/omnipiano/runs/song_sweep_ot_raiden
/data/giil/zhangzy/omnipiano/runs/marl_core_v1
/data/giil/zhangzy/omnipiano/runs/robust_core_v1
```

对应 submission manifests：

```text
/data/giil/zhangzy/omnipiano/runs/etude12_reward_f1_v4/submission_manifests/etude12_20260828_061212_pid2866220.tsv
/data/giil/zhangzy/omnipiano/runs/song_sweep_ot_raiden/submission_manifests/song_sweep_ot_20260830_172252_pid3976415.tsv
/data/giil/zhangzy/omnipiano/runs/marl_core_v1/submission_manifests/marl_core_20260828_231051_pid768952.tsv
/data/giil/zhangzy/omnipiano/runs/robust_core_v1/submission_manifests/robust_core_20260828_231112_pid770306.tsv
```

### 3.3 W&B 的角色

W&B 用于在线看曲线、团队共享和后续汇总，但不是唯一事实来源。Core uploader 曾对 MARL ForElise monolithic PPO seed 2 发生大量初始化 timeout；本地 artifact 完整，不应因 W&B 缺 run 就判定训练丢失。最终表格应由本地 artifacts 生成，再检查 W&B run 数量、logical ID、snapshot hash 和 final step 是否一致。

任何文档、脚本、日志或 commit 都不得写入 W&B API key。

## 4. Loss 与优化健康审计

### 4.1 SB3 PPO：有限，但更新幅度异常

PPO 的 train loss、value loss 和 reward/F1 曲线没有显示数值爆炸；真正的问题是 policy update 过大：

| 数据组 | `clip_fraction` 观察 | `approx_kl` 观察 |
|---|---:|---:|
| Etude-12 PPO | 中位约 0.862；final 约 0.878–0.907 | 中位约 0.597；final 约 0.676–1.116；峰值约 3.09 |
| Song Sweep PPO | 中位约 0.851；final 约 0.868–0.911 | 中位约 0.531；final 约 0.611–1.182；峰值约 5.03 |
| MARL monolithic PPO | 最后更新约 0.918–0.940 | 最后更新约 1.19–1.69 |
| 当前 Robust PPO | 约 0.813–0.874 | clean run 峰值约 1.42、1.63、5.09；final 约 0.594–0.657 |

如此高的 clip fraction 表明大部分样本都落在 clipping 区域，高 KL 表明更新远离旧策略。结论应写成：

- 这些 run 可保留为可复现的 `SB3-default PPO baseline`；
- 不能仅凭 loss 有限就声称 PPO 正常收敛；
- 需要增加 calibrated PPO 对照，优先测试更低 learning rate、更少 update epochs、合适的 batch size 和 `target_kl`；
- 不覆盖或删除现有默认 PPO，避免事后改变 baseline。

### 4.2 SAC：当前未见同类系统性异常

已审计 SAC 的 critic loss 通常处于约 0.01–0.02，最大值低于 1。少量 `ent_coef_loss` 瞬时尖峰在下一次更新恢复，未伴随持续 reward/F1 崩塌。目前可以把 SAC 视作数值稳定 baseline，但这不等于它在所有歌曲上一定优于 PPO。

### 4.3 IPPO/MAPPO：不是“loss 正常”，而是“loss 没有被记录”

当前 `training_progress.jsonl` 主要记录：

- environment steps；
- wall time；
- RLlib all-agent diagnostic return；
- episode length。

它没有记录 policy loss、value loss、entropy、KL、clip fraction、gradient norm、explained variance。因此，IPPO/MAPPO 曲线有限且任务完成，只能说明训练流程没有显式崩溃，不能证明 learner 优化健康。

此外，`rllib_all_agent_return_mean_diagnostic` 会把广播给多个 agent 的 shared team reward 再求和，通常约为真实 `team_return` 的 1.88–1.96 倍。这个字段只能做 RLlib 内部诊断，不能与 monolithic PPO return 放在同一坐标轴比较。

## 5. Reward–F1 alignment 的实证结论

### 5.1 哪个 F1 可以比较

Metrics-v2 `note_event_f1` 是独立于 reward 的评测量：按音高精确匹配、预测与目标事件一对一匹配，并使用 50 ms onset tolerance。它衡量“是否在正确时间弹出了正确音符”，目前未受到 contact-attribution bug 影响。

legacy `ep_f1`/frame-macro F1 来自 50 ms frame 上的 MIDI wrapper，容易受到休止帧和 empty-frame 语义影响。它只能用于历史兼容诊断，不能与 Metrics-v2 `note_event_f1` 合并成一张表。

特别地，现有 robust 历史文档把 2026-07 旧 run 的 `ep_f1` 描述为 note-level F1。按其代码 provenance，这些值应暂时标记为 **legacy frame-macro F1**，直到用保存的 checkpoint 按 Metrics-v2 重评。诸如 clean `0.626`、R-Gauss `0.109` 的历史值不能直接解释为当前的 note-event F1。

### 5.2 相关性结果

| 数据组 | 原始 reward–F1 相关 | 增量/局部相关 | 额外现象 |
|---|---|---|---|
| Etude-12 | 每 run Spearman 约 0.921–0.992 | delta Spearman 约 0.021–0.697 | 37/48 runs 出现 zero-event-F1 checkpoint；2/12 歌曲发生 PPO/SAC 排序翻转 |
| Song Sweep OT | 每 run reward–legacy-F1 Spearman 约 0.942–0.997 | delta Spearman 约 0.086–0.849 | 71/2360 checkpoints 的 legacy frame-macro F1 为 0；4/11 env 发生 reward vs legacy-F1 算法排序翻转 |
| MARL-Core | 同一歌曲 checkpoint Pearson 约 0.509–0.902 | — | 跨曲 raw return–F1 为 −0.180；按 episode step 归一化后为 +0.741 |

Etude-12 中共有 477/4800 个 checkpoint 的 `note_event_f1=0`，此时 `return_per_step` 最高仍约 2.901。最终 reward 中 sustain 和 forearm shaping 合计占比中位数约 48.8%，48 个 final 中有 46 个 sustain F1 为 1。这说明 policy 可以先学会持续踏板、姿态或其他 shaping 项，却没有匹配到任何正确 note event。

### 5.3 为什么 raw correlation 高仍然不够

训练步数同时推动 reward 与 F1 上升，会制造很高的 raw correlation。更严格的问题是：在相邻 checkpoint 之间，reward 的增加是否对应 F1 的增加；不同算法按 reward 排名时，是否也按 F1 排名；高 reward 是否排除了不弹正确音的 idle solution。

当前证据给出的答案是否定的：

- delta correlation 可以接近 0；
- 多个 checkpoint 在 F1 为 0 时仍获得明显正 reward；
- PPO/SAC 在部分歌曲上的 reward 排名和 F1 排名相反；
- raw episode return 受曲长影响，跨歌曲直接平均会混入 episode length；
- reward-noise/shift 可以直接改变观测到的 return，而不对应真实音乐表现改变。

因此，reward 可以继续作为训练信号和同任务内的辅助诊断，但不能单独作为跨歌曲 leaderboard 的主分数。

### 5.4 `F1=0` 应怎样解释

`note_event_f1=0` 本身不自动代表指标程序出错。它表示该 evaluation episode 中没有满足音高、一对一匹配和 onset tolerance 的 true positive。若 policy 只维持踏板、姿态、近似按键或发生时序偏移，reward 仍可能大于 0。

排查顺序应是：

1. 确认使用的是 Metrics-v2 还是 legacy F1；
2. 检查 target event 数、predicted event 数、TP/FP/FN；
3. 检查 onset tolerance、episode 截断和 MIDI offset；
4. 查看 event trace 和视频；
5. 再判断是合理的失败样本、reward loophole，还是 evaluator bug。

## 6. 已确认的评测实现缺陷

### 6.1 P0：contact-force attribution threshold 为 0

冻结快照中的 event attribution 使用了等价于以下逻辑的判定：

```python
np.flatnonzero(impulse > 0.0)
```

正式评测 metadata 也记录了 `contact_force_threshold_n=0.0`。这会把浮点残余和 denormal 级别的“接触力”误判为真实接触。

在当时已完成的 35 个 MARL final trace 中：

- actual events：2015；
- attributed events：1728；
- attributed events 中有 581 个最大力低于 `1e-6 N`，占 attributed events 的 33.6%，占全部 actual events 的 28.8%；
- 最小值约为 `6.4e-314 N`，显然不能代表物理接触。

受污染的字段包括：

- per-hand/per-agent correct-event attribution；
- contribution share；
- effective active agents；
- idle-agent rate；
- unattributed/duplicate attribution rate；
- per-hand/per-agent work-per-correct-event。

不受影响的字段包括：

- 模型和 checkpoint；
- 训练 reward；
- team return；
- 全局 note-event F1；
- key-time micro F1；
- 不依赖 attribution 的全局 `episode_physical/work_per_correct_event`。

结论：这是评测后处理/指标归属 bug，不是训练全部失效。已有 `episode_trace.npz` 的 run 可在确定非零 threshold 后离线重算，通常无需重训 policy。上面的 2015/1728/581 统计只覆盖先完成的 35 个 run；第 36 个 run 已经完成但尚未纳入该次统计，正式重算必须覆盖全部 36 个。

### 6.2 P0：collision force non-finite

以下 MARL jobs 的 collision-force integral/max 在原始 trace 中出现 non-finite，strict JSON 后成为 `null`：

```text
18467672  18467677  18467685  18467688  18467692
```

这些 run 的 collision force 指标不能用于正式结论；在查清 trace 来源并加入 finite-value gate 前，collision count/rate 也应谨慎使用。例如不能据当前值直接声称某个 GreatKiev policy 存在特定碰撞率。

### 6.3 人工 contact 抽查尚未完成

系统已经生成 420 条 contact checklist 样本，但人工标注数为 0。当前不能写“contact attribution 已被人工验证”。人工抽查应同时查看 event、接触位置/力、对应 hand/agent 和视频帧，用于选择阈值并估计误归属率。

## 7. 哪些是 bug，哪些是 benchmark 的科学问题

| 现象 | 分类 | 处理方式 |
|---|---|---|
| `impulse > 0` 导致虚假 contact attribution | 工程 bug | 修 threshold、加测试、离线重算 |
| collision force non-finite | 工程 bug | 修 trace/聚合 finite gate，重评受影响 run |
| IPPO/MAPPO 缺 learner metrics | observability 缺口 | 增加日志，跑小规模验证 |
| W&B 缺 run 但本地 artifact 完整 | 同步/展示问题 | 以本地为准，重传并做 hash 对账 |
| PPO 高 clip fraction/KL | baseline 优化问题 | 保留默认 baseline，增加 calibrated PPO |
| reward 高但 F1 为 0 | 可能是 reward loophole/任务失败 | 用独立指标和 trace 判断，不自动当 evaluator bug |
| raw reward 与 F1 排序不一致 | benchmark 科学问题 | 正式报告 alignment、ranking agreement 和 failure rate |
| reward shift 改变 received return | threat-model/测量问题 | 同时报 `true_return` 与 F1，禁止用 polluted return 做 headline |

OmniPiano 的价值不是强迫 reward 与 F1 完全相同。如果直接把 reward 设置成最终 F1，再用 F1 证明 reward 有效，会形成循环论证。更有价值的设计是保留 reward-independent scorecard，比较不同 reward、算法和协作机制能否真正提升音乐表现。

## 8. 当前结果的使用决策

| 结果/产物 | 决策 | 理由或动作 |
|---|---|---|
| model/checkpoint | 保留 | attribution bug 不影响训练 |
| training curve 与 source hash | 保留 | 可用于复现和优化诊断 |
| 同任务内 team return | 保留为辅助量 | 不跨曲直接比较；reward noise 下区分 received/true |
| Metrics-v2 note-event/key-time F1 | 保留 | 当前主要独立音乐表现指标 |
| SB3 PPO loss/KL/clip | 保留 | 用于说明默认 baseline 的更新异常 |
| SAC loss | 保留 | 当前数值审计正常 |
| MARL contact/agent contribution 指标 | 离线重算 | 先确定非零 threshold 与 sensitivity |
| 受影响 collision 指标 | 重评 | 修复 non-finite gate 后重新生成 |
| MARL final uncertainty | 增加 evaluation | 每 seed 目前仅 1 episode，不足以估计 episode-level 方差 |
| Etude-12 正式统计 | 补 24 个 run | 增加 PPO/SAC seed 2 |
| Song Sweep 正式统计 | 重评并补 seeds | 当前是 single-seed legacy metric |
| Robustness 正式结论 | 等矩阵与 sweep 完成 | 当前仅 6/60 完成 |

## 9. 建议冻结的 benchmark scorecard

不建议把多个维度压成一个未经验证的 “OmniScore”。主表应并列展示以下维度。

### 9.1 音乐正确性（headline）

- Metrics-v2 `note_event_precision/recall/F1`；
- `key_time_micro_precision/recall/f1`；
- onset/offset timing error；
- F1 learning AUC、达到固定 F1 阈值所需 steps；
- zero-event-F1 checkpoint rate 和 high-reward/zero-F1 rate。

### 9.2 Reward alignment（诊断）

- 同一 run 的 raw、delta 和 detrended reward–F1 correlation；
- 算法按 reward 与按 F1 的 ranking agreement；
- return per step 或按目标 event 归一化的 return；
- reward component decomposition；
- idle-solution incidence。

### 9.3 Multi-agent coordination（修复 attribution 后）

- team F1；
- per-agent contribution share；
- effective active agents；
- idle-agent rate；
- duplicate/unattributed note rate；
- load balance 和 partition violation；
- collision 与 per-agent work-per-correct-event。

### 9.4 Robustness

- F1–noise-scale AUC；
- clean-to-matched F1 drop；
- worst-scale F1；
- precision/recall degradation；
- `received_return` 与 `true_return` 的差值；
- paired 10-episode evaluation 与 3-seed uncertainty。

### 9.5 物理效率与安全

- actuator work/energy；
- collision count、force integral/max；
- safety constraint violation rate/cost；
- 每个正确 note event 的能耗或接触代价。

## 10. Reward 设计建议

不要覆盖现有 reward。把它冻结并命名为 `LegacyDenseReward-v1`，从而保证历史结果可复现。另行注册和消融一个尚未实现的 `AlignedDenseReward-v2` 候选：

- 移除或居中 idle 状态下的正 reward 基线；
- sustain reward 只在乐谱需要踏板的区间生效；
- 降低 forearm/fingering shaping 相对 event correctness 的权重；
- 对 note-event TP 给 credit，对 FP、FN 和 timing error 给明确 penalty；
- 按目标 event 数或 episode length 做尺度归一化；
- 保持 evaluation Metrics-v2 完全独立，不把 evaluator 直接复制成 reward。

Reward-v1/v2 应在相同算法、歌曲、seed、预算和 evaluation cadence 下对照，并同时报告：最终 F1、sample efficiency、alignment、idle failure、训练稳定性和 reward sparsity。这样才能回答“更对齐的 reward 是否真正更容易学习”，而不是只证明新 reward 数值更像 F1。

## 11. 下一步执行优先级

### P0：先修评测正确性与文档口径

1. 不打断正在运行的 immutable Robustness jobs；让训练按冻结协议完成。
2. 为 contact attribution 选择物理上有意义的非零 threshold，做 threshold-sensitivity curve。
3. 完成 420 条 checklist 中有代表性的人工抽查并记录 annotator agreement。
4. 为 collision trace 添加 finite gate、异常来源日志和 synthetic regression test。
5. 使用保存的 episode traces 离线重算 36 个 MARL run 的 attribution/coordination scorecard；缺 trace 时重放 checkpoint，而不是重训。
6. 统一 paper metric policy：解决“return primary”与本次审计建议之间的冲突；修正历史 robust `ep_f1` 被称为 note-level F1 的口径错误。

### P1：补优化可观察性与公平 baseline

1. 给 IPPO/MAPPO 写出 policy/value loss、entropy、KL、clip fraction、grad norm、explained variance 和真实 team return。
2. 把 RLlib 的 all-agent summed return 明确重命名为 diagnostic，禁止参与跨算法主图。
3. 用少量代表歌曲做 calibrated PPO pilot；通过后再决定是否扩展到正式矩阵。
4. 在下一协议版本或额外 held-out evaluation 中，把 MARL final eval 从 1 episode 提升为多 episode paired evaluation；当前冻结 MARL 协议本身规定的是 1 episode。

### P2：完成正式矩阵

1. 补 Etude-12 seed 2 的 24 个 PPO/SAC run。
2. 完成 Robustness-Core 60/60，并执行 `eval_noise_scale={0,0.5,1,2,4}`、每 scale 10 episodes 的 sweep。
3. 对 Song Sweep 用 Metrics-v2 重评 checkpoint，并至少补到 3 seeds 后再做 paper claim。
4. 重新生成 strict aggregate、plots、tables 与 W&B final mirror。

### P3：Benchmark ablation

1. 实现并注册 `AlignedDenseReward-v2`，与 v1 做公平对照。
2. 扩展曲目难度、形态手数和 coordination territory 轴。
3. 再考虑是否需要单一综合分数；在维度权重没有 reviewer 可辩护依据前，保持多维 scorecard。

## 12. Paper claim guardrails

当前证据支持以下表述：

- OmniPiano 揭示了 high RL return 不一定意味着正确的音乐事件表现；
- 独立于 reward/noise channel 的音乐指标对 robustness 评测是必要的；
- 默认 PPO 在该高维任务上呈现有限 loss 但异常大的 policy updates；
- 当前训练和模型 artifact 总体完整，主要问题集中在评测归属、日志可观察性和指标口径。

当前证据不支持以下表述：

- “MAPPO 已经学会更好的合作”——contact attribution 尚未修复；
- “PPO 已健康收敛”——clip fraction/KL 不支持；
- “raw episode return 可以跨歌曲排名”——曲长混淆且出现跨曲负相关；
- “历史 robust F1 就是 Metrics-v2 note-event F1”——其 provenance 指向 legacy frame-macro F1；
- “single-seed Song Sweep 或 two-seed Etude 已是正式 benchmark 结果”；
- “Robustness-Core 已完成”——截至快照仅 6/60 完成；
- “contact attribution 已人工验证”——420 条 checklist 尚未标注。

### 12.1 与现有 paper 文档的待解决冲突

当前 [`paper/claims.md`](../../paper/claims.md) 和 [`paper/robust_experiments_index.md`](../../paper/robust_experiments_index.md) 记录了 2026-08-28 的决定：episode return 为 primary、F1 为 secondary。本次审计发现：

- raw return 可在 F1 为 0 时保持较高；
- reward/F1 存在算法排序翻转；
- raw return 不能跨曲比较；
- reward shift 会直接污染 received return；
- 历史 robust F1 的 metric family 标注也需修正。

因此这两份 paper 文档在团队重新冻结 metric policy 前应视为“历史写作口径”，不能单独作为当前 leaderboard 定义。最低安全共识是：任何主结论都必须同时报告 reward-independent musical metrics；raw return 禁止用于跨曲 leaderboard，reward-noise 结论必须同时给出 `true_return` 和 F1。

## 13. 修复与正式发布的验收条件

在把当前结果称为 paper-ready 之前，至少满足：

- contact threshold 有单位、有 sensitivity 分析，并通过人工抽查；
- collision/energy 等物理 trace 全部 finite，未定义量以 `null` 表示而非静默填 0；
- Metrics-v2 名称、版本和 tolerance 写入每个 evaluation artifact；
- legacy F1 与 Metrics-v2 绝不混表；
- 正式比较至少 3 seeds，并有多 episode final evaluation；
- 算法比较使用相同任务、budget、eval cadence 和 metric version；
- aggregate 脚本严格校验 exact matrix，不允许缺 run 静默算平均；
- W&B logical ID、step、snapshot hash 与本地 artifact 对账；
- 所有 headline 数字均能从冻结 artifact 一键再生成。

## 14. 相关仓库文档

- [`robust_task_design.md`](robust_task_design.md)：robust task 的实现与 threat-model 设计；
- [`paper/robust_notes.md`](../../paper/robust_notes.md)：历史 robust 写作笔记与单-seed 证据；
- [`paper/claims.md`](../../paper/claims.md)：当前 paper claim 台账，但 metric policy 需按 §12.1 重新统一；
- [`paper/robust_experiments_index.md`](../../paper/robust_experiments_index.md)：历史 robust 实验索引，其 F1 family 与 primary-metric 表述需修订。

## 15. 推荐的最终报告层级

正式论文和 benchmark release 建议按以下顺序组织结果：

1. **Performance**：每首曲、每算法的 Metrics-v2 event/key-time F1 与 timing error；
2. **Learning**：F1 AUC、sample efficiency、reward curve 和优化诊断；
3. **Alignment**：reward–F1 raw/delta correlation、ranking flips、idle failures；
4. **Coordination**：修复后的多 agent attribution、负载与碰撞；
5. **Robustness**：noise-scale performance curve、worst case、received/true return；
6. **Efficiency/Safety**：能耗、碰撞、constraint cost；
7. **Reproducibility**：seed、snapshot hash、manifest、checkpoint 和 W&B mirror。

这样形成的 benchmark 交付物不是“一张 reward 曲线”，而是一套可以区分会刷 shaping reward、会弹对音符、会协调多手、能抵抗扰动并满足物理约束的完整测评流程。
