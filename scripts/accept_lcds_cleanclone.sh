#!/usr/bin/env bash
set -euo pipefail
: "${CC_CLEAN_CLONE:?Set CC_CLEAN_CLONE to a new absolute clone directory}"
if [[ "$CC_CLEAN_CLONE" != /* || -e "$CC_CLEAN_CLONE" ]]; then
  echo "Refusing a relative or existing clone target: $CC_CLEAN_CLONE" >&2
  exit 2
fi
# Some remote shells inherit ignored SIGCHLD; Git/LFS needs normal waitpid.
git_safe() {
  python3 -c 'import os,signal,sys; signal.signal(signal.SIGCHLD,signal.SIG_DFL); os.execvp("git", ["git", *sys.argv[1:]])' "$@"
}
export GIT_LFS_SKIP_SMUDGE=1 GIT_PAGER=cat
git_safe -c http.version=HTTP/1.1 clone --depth 1 --single-branch --branch codex/reproducible-release \
  https://github.com/kong-johnny/CausalChapter.git "$CC_CLEAN_CLONE"
cd "$CC_CLEAN_CLONE"
git_safe lfs install --local
git_safe lfs pull --include="checkpoints/lcds_longformer_reconstruction/best.pt,data/avlecture/lcds_visual_features.tar.gz"
git_safe rev-parse HEAD
python3 scripts/verify_artifacts.py --profile lcds
python3 scripts/prepare_eval_features.py \
  --archive data/avlecture/lcds_visual_features.tar.gz --destination data/lcds
export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=""
python3 scripts/smoke_lcds_release.py --output outputs/lcds_release_smoke
echo "GITHUB_CLONE_LCDS_ACCEPTANCE_COMPLETE; selective weights, no NAS data/weight links"
