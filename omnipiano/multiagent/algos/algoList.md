# Repo 清单 + 精确 clone 命令

| 算法 | 首选 repo | 为什么是它 | 取什么 | 层 |
|---|---|---|---|---|
| **HAPPO** | `github.com/PKU-MARL/HARL` | HAPPO 作者自己的后续官方实现(HARL, JMLR 2024)。README 明说含 HAPPO/HATRPO/HAA2C/HADDPG/HATD3/HAD3QN/**HASAC**,且"sequential update scheme"是其核心特征 | HAPPO actor 的 surrogate 数学 | **V2** |
| HAPPO(交叉核对) | `github.com/morning9393/HAPPO-HATRPO` | ICLR 2022 原论文代码 | 同一个 surrogate,双重核对 | V2 |
| **A2PO** | `github.com/xihuai18/A2PO-ICLR2023` | ICLR 2023 论文给出的官方实现；仓库声明仍在重构，因此固定 commit 后作为公式和控制流 oracle | PreOPC、semi-greedy 顺序、joint-ratio clipping、near-linear adaptive clip | **V2** |
| **MAT** | `github.com/PKU-MARL/Multi-Agent-Transformer` | 官方(论文正文给的就是这个链接) | `mat/algorithms/mat/algorithm/` 下的 transformer;`mat/algorithms/mat/mat_trainer.py` | **V1 + V2** |
| **FACMAC** | `github.com/oxwhirl/facmac` | 官方,含 FACMAC / FACMAC-nonmonotonic / MADDPG / COMIX / COVDN。基于 PyMARL | `src/modules/mixers/` 的 QMIX mixer;FACMAC learner 作 V3 | **V1 + V3** |
| **MASAC** | `github.com/PKU-MARL/HARL`(**HASAC**) | HASAC = heterogeneous-agent SAC,**和 HAPPO 同一个 repo**。一次 clone 覆盖两个算法 | HASAC actor + critic | **V2 + V3** |
| MASAC(中心化 critic 参考) | `github.com/marlbenchmark/off-policy` | MAPPO 团队的 off-policy 姊妹 repo(MADDPG/MATD3 等),README 里有链接 | critic 构造方式 | V2 |
| **MAPPO 回溯预言机** | `github.com/marlbenchmark/on-policy` | 官方 MAPPO(Yu et al., "Surprising Effectiveness of PPO")。**你的那些 trick 就出自这里** | `onpolicy/algorithms/` 下的 MAPPO 更新 | **V2** |

**顺便一个交叉核对源:`github.com/jidiai/GRF_MARL`** 在一个框架里同时实现了 IPPO / MAPPO / HAPPO / A2PO / MAT —— 想看"同一套代码风格下 IPPO 和 HAPPO 差什么"时非常省时间。
