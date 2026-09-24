# Baseline and scope audit — 2026-09-19

- Repository: `SAIL-Research-Lab/omnipiano`; target branch: `safety`.
- Main baseline: `8a1545aba66a6baf03c5c905a8bf5bee80a2fe2a`.
- Uploaded archive: `omnipiano-main (2).zip`.
- Archive SHA256: `3fb329d4119b940bff35a6b09fc3bacd915ddf1bc70756e196145a78b414f4c6`.
- All 173 tracked paths match the archive after CRLF-to-LF normalization; no extra archive files.
- 31 of those paths differ from the previous local working project. The uploaded
  archive / current main is the baseline, not the old local tree.

## Relevant differences and resolutions

| Main component | Difference from the previous local safety work | Resolution confined to safety/ |
|---|---|---|
| configs | No SafetyConfig cost_limit or BenchmarkEnvConfig energy_penalty_coef | Budget in Task; safety-local dataclass exposes upstream energy parameter |
| SafetyWrapper | No deepcopy, reset hooks or appended action-history observations | New costs are immutable/stateless; no action_slew migration |
| registration | Different factory implementation | Use its existing register/make APIs without patching them |
| task/reward | Main implementation changed | Use main's implementation; preserve selections, not historical binary equivalence |
| legacy constraints | Main contains older public API used by envs/__init__.py | Leave constraints.py byte-for-byte unchanged; add semantics.py |
| training/replay examples | Different defaults and missing local protocol additions | Self-contained safety run/train/cmdp/evaluate/plot modules |
| old main/ablation scripts | Local-only scripts not in this baseline | Rebuild under safety/; preserve 120 + 60 + 63 independent cells |

Changed baseline paths compared with the old local project:

```text
.gitignore
.readthedocs.yaml
README.md
docs/_static/custom.css
docs/_static/images/framework_v1.png
docs/conf.py
docs/index.rst
docs/introduction/overview.md
docs/introduction/quick_start.md
docs/requirements-doc.txt
examples/checkpoint_replay_eval.py
examples/render_checkpoint.py
examples/run_omnisafe_template.py
examples/run_sb3_sac_template.py
examples/run_sb3_tqc_template.py
omnipiano/configs/__init__.py
omnipiano/docs/robust_task_design.md
omnipiano/envs/__init__.py
omnipiano/envs/registration.py
omnipiano/safety/constraints.py
omnipiano/tasks/omni_piano_task.py
omnipiano/tests/test_protocol_wiring.py
omnipiano/tests/test_robust_reward_noise.py
omnipiano/wrappers/robust_wrapper.py
omnipiano/wrappers/safety_wrapper.py
paper/README.md
paper/claims.md
paper/figures/fig2_learning_curves.png
paper/make_robust_figures.py
paper/robust_experiments_index.md
paper/robust_notes.md
```

The previous website importer and handoff packages are separate deliverables and
are not copied into this experiment-only publication. No credentials, private logs,
PIG data, models, generated figures or local paths are required in the commit.

The publication diff must contain only `omnipiano/safety/` paths. No force-push,
main-branch update, unrelated deletion or automatic merge is part of this change.
