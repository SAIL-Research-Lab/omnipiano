# paper/ — OmniPiano 论文工作目录

论文与代码同库演进,每个版本经 git 可追溯。

## 约定

- **语言**:论文正文/摘要/claims 均为英文(投稿语言);讨论注释可用中文。
- **文件结构**:
  - `claims.md` — 核心 claims 清单(paper-ready 措辞 + 证据指针 + 状态标记)。写作的单一事实来源;各节草稿从这里取 claim,不另行发明。
  - `abstract.md` — abstract 草稿(带版本号 v0/v1/…)+ 决策日志(venue、长度、语气)。
  - 后续:`outline.md`、`sections/` 按需增加。
- **同步纪律**(双会话协作:一边修代码、一边写论文):
  - 实验数字**只从设计文档引用**(`robust_task_design.md` §10、`static_partition_design.md`、README demo 数字),不在 paper/ 里手编;数字变更先改设计文档再改这里。
  - 写作决策(venue、claims 取舍)落在本目录 + memory,代码会话不必读 paper/ 全文。
- **状态标记**:每条 claim 带 `[verified]`(有实测数字/测试锁定)、`[pending]`(实验排队中)、`[design]`(纯设计性质、无需实验)。abstract 只允许引用 `[verified]` 和 `[design]` 的 claim;`[pending]` 的先用占位措辞。
