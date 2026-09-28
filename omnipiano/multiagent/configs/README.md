# MARL configurations

This directory has three intentionally separate roles:

- `marl_train_config_default.json` is the schema-v1 default used by the CLI.
- `marl_task_example.json` is the schema-v2 authoring template used by the
  compiler and tests.
- `scho_winterwind_5x5x3/` is the frozen paper reproduction suite: five
  algorithms, five SCHO settings and three seeds (75 actual training runs).

Additional single-experiment configurations live in `examples/`; they are not
members of the frozen 5 x 5 x 3 comparison.

Generation, validation, environment-audit, and queue code lives beside the
MARL implementation in `../reproduction/`, not among the JSON files.

Do not edit JSON under `scho_winterwind_5x5x3/runs/` in place. Its manifest and
checksum file intentionally make configuration drift fail fast.
