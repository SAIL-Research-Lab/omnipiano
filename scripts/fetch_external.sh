#!/usr/bin/env bash
# Fetch reference MARL implementations, pin them, and record provenance.
#
#   bash scripts/fetch_external.sh list          # what would be fetched
#   bash scripts/fetch_external.sh clone harl    # clone one upstream (gitignored)
#   bash scripts/fetch_external.sh clone all
#   bash scripts/fetch_external.sh locate harl   # find the files we care about
#   bash scripts/fetch_external.sh vendor harl   # copy leaf files + PROVENANCE
#
# Two-stage on purpose. `clone` pulls the WHOLE repo into external/_upstream/,
# which is gitignored: those repos pin their own gym/sacred versions and must
# never enter our import path. `vendor` copies only the leaf files that are pure
# torch, into a tracked directory, and records a sha256 for every one so nobody
# can quietly edit reference code and still call it reference code.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

UPSTREAM=external/_upstream
VENDOR=omnipiano/multiagent/external

# name|url|pin  -- pin is a tag/branch; `vendor` records the resolved SHA.
# Pinning to a branch is a deliberate first step: you cannot know the right
# commit before you have read the code. `vendor` freezes whatever you resolved.
REPOS=(
  # HAPPO oracle moved here from HARL: this is the ICLR 2022 paper's own code,
  # i.e. the reference we actually cite, AND it is MIT. HARL is unlicensed.
  # Default branch is `master` (it is a fork of cyanrain7/TRPO-in-MARL).
  "happo_iclr22|https://github.com/morning9393/HAPPO-HATRPO.git|master"
  "on_policy|https://github.com/marlbenchmark/on-policy.git|main"
  # Unlicensed upstreams: cloned for READING and for V3 (run in their own conda
  # env) only. `vendor` refuses them, by design.
  "harl|https://github.com/PKU-MARL/HARL.git|main"
  "mat|https://github.com/PKU-MARL/Multi-Agent-Transformer.git|main"
  "facmac|https://github.com/oxwhirl/facmac.git|main"
)

# What we are looking for in each repo. These are SEARCH PATTERNS, not paths:
# upstream layouts drift, so `locate` finds them and prints what it found.
declare -A WANT=(
  [happo_iclr22]="happo trpo ppo_trainer r_actor_critic valuenorm util.py"
  [on_policy]="r_mappo.py r_actor_critic popart valuenorm util.py"
  [harl]="happo hasac v_critic soft_twin_continuous_q_critic on_policy_base"
  [mat]="ma_transformer transformer_policy mat_trainer"
  [facmac]="qmix.py vdn.py facmac_learner maddpg_learner"
)

_row () { IFS='|' read -r n u p <<<"$1"; echo "$n" "$u" "$p"; }

case "${1:-list}" in
list)
  printf '%-14s %-58s %s\n' NAME URL PIN
  for r in "${REPOS[@]}"; do printf '%-14s %-58s %s\n' $(_row "$r"); done
  echo
  echo "upstream (gitignored): $UPSTREAM"
  echo "vendored   (tracked):  $VENDOR"
  ;;

clone)
  mkdir -p "$UPSTREAM"
  for r in "${REPOS[@]}"; do
    read -r n u p <<<"$(_row "$r")"
    [[ "${2:-all}" == "all" || "${2}" == "$n" ]] || continue
    if [[ -d "$UPSTREAM/$n/.git" ]]; then
      echo "[skip] $n already cloned"; continue
    fi
    echo "[clone] $n <- $u ($p)"
    # --depth 1 keeps it small; we record the resolved SHA, which is what
    # provenance actually needs. Drop --depth if you want to bisect upstream.
    git clone --depth 1 --branch "$p" "$u" "$UPSTREAM/$n" \
      || echo "[warn] $n failed (branch '$p' may not exist; try master)"
  done
  ;;

locate)
  n=${2:?usage: locate <name>}
  d=$UPSTREAM/$n
  [[ -d $d ]] || { echo "[abort] not cloned: $d"; exit 1; }
  echo "== $n @ $(git -C "$d" rev-parse HEAD)"
  echo "-- LICENSE / COPYING (NO LICENCE == all rights reserved == cannot vendor):"
  # -maxdepth is a GLOBAL option and must precede all tests, or find warns that
  # it applies to tests written before it too. Grouping the -o with \( \) is also
  # required, otherwise -maxdepth binds to only one branch.
  found_lic=$(find "$d" -maxdepth 2 \( -iname 'LICENSE*' -o -iname 'COPYING*' \) -print)
  if [[ -z "$found_lic" ]]; then
    echo "   *** NONE FOUND -- vendoring is BLOCKED for $n ***"
  else
    echo "$found_lic" | sed 's/^/   /'
    echo "   first line of each:"
    while read -r L; do echo "     $(basename "$L"): $(head -1 "$L")"; done <<<"$found_lic"
  fi
  find "$d" -maxdepth 2 -iname 'LICENSE*' -o -maxdepth 2 -iname 'COPYING*' | sed 's/^/   /'
  echo "-- files matching what we want:"
  for pat in ${WANT[$n]}; do
    find "$d" -name "*${pat}*" -name '*.py' -not -path '*/.git/*' | sed "s/^/   [$pat] /"
  done
  echo
  echo "NEXT: read them, then list the exact leaf files in VENDOR_FILES below"
  echo "      and run: bash scripts/fetch_external.sh vendor $n"
  ;;

vendor)
  n=${2:?usage: vendor <name>}
  d=$UPSTREAM/$n
  [[ -d $d ]] || { echo "[abort] not cloned: $d"; exit 1; }
  # Fill this in AFTER `locate` -- one upstream-relative path per line.
  # Leave empty to be told so, rather than silently vendoring nothing.
  VENDOR_FILES=$(cat "external/vendor_manifest/$n.txt" 2>/dev/null || true)
  if [[ -z "$VENDOR_FILES" ]]; then
    echo "[abort] no manifest: external/vendor_manifest/$n.txt"
    echo "        Create it with one upstream-relative .py path per line."
    exit 1
  fi
  # The whole point of provenance is that a reviewer can check it. Copying
  # unlicensed code in would make the artifact unredistributable, which is a
  # much more expensive mistake to discover at camera-ready than here.
  if ! find "$d" -maxdepth 2 \( -iname 'LICENSE*' -o -iname 'COPYING*' \) \
       | grep -q .; then
    echo "[abort] $n has no LICENSE/COPYING at its root."
    echo "        Absent a licence, default copyright applies: we may clone and"
    echo "        run it, but copying files into a repo we publish is"
    echo "        redistribution. Options:"
    echo "          (a) file an issue upstream asking for a licence;"
    echo "          (b) use a licensed equivalent (HAPPO -> happo_iclr22, MIT);"
    echo "          (c) keep it in _upstream/ only and run it via V3."
    exit 1
  fi
  sha=$(git -C "$d" rev-parse HEAD)
  out=$VENDOR/$n; mkdir -p "$out"
  : > "$out/.filelist"
  for f in $VENDOR_FILES; do
    [[ -f "$d/$f" ]] || { echo "[abort] missing upstream file: $f"; exit 1; }
    dst="$out/$(basename "$f")"
    cp "$d/$f" "$dst"
    echo "$f  ->  $(basename "$f")  $(sha256sum "$dst" | cut -d' ' -f1)" >> "$out/.filelist"
    echo "[vendor] $f"
  done
  for L in "$d"/LICENSE* "$d"/COPYING*; do
    [[ -f "$L" ]] && cp "$L" "$out/UPSTREAM_$(basename "$L")" && echo "[vendor] $(basename "$L")"
  done
  python - "$n" "$sha" "$out" <<'PY'
import hashlib, json, pathlib, sys
name, sha, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
url = {"harl": "https://github.com/PKU-MARL/HARL",
       "mat": "https://github.com/PKU-MARL/Multi-Agent-Transformer",
       "facmac": "https://github.com/oxwhirl/facmac",
       "on_policy": "https://github.com/marlbenchmark/on-policy",
       "happo_iclr22": "https://github.com/morning9393/HAPPO-HATRPO"}[name]
files = {}
for p in sorted(out.glob("*")):
    if p.name.startswith(".") or p.name == "PROVENANCE.json":
        continue
    files[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
(out / "PROVENANCE.json").write_text(json.dumps({
    "name": name, "upstream_url": url, "upstream_commit": sha,
    "vendored_verbatim": True,
    "policy": ("Files in this directory are VERBATIM upstream copies and are "
               "NEVER edited. Any adaptation lives in _adapter.py, which we "
               "wrote. provenance_test.py fails if any sha256 below changes, so "
               "'unmodified reference implementation' is a checkable claim."),
    "sha256": files,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"[ok] wrote {out}/PROVENANCE.json ({len(files)} files @ {sha[:12]})")
PY
  ;;

*) echo "usage: $0 {list|clone|locate|vendor} [name]"; exit 1 ;;
esac
