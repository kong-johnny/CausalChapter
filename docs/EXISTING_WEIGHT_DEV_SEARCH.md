# Existing-weight and decoding search: predicted chapters only

The user authorizes broad improvement attempts, starting with inexpensive
hyperparameter changes and combinations of existing checkpoints. The hard
constraint is **no ground-truth segmentation as generation input**.

## Data and protocol locks

- The first stage uses the first eight videos of the original dev split,
  identically for every candidate. Subset scores are screening results, not
  publishable final benchmark scores. Promising candidates need full-dev
  confirmation before any later test evaluation.
- Boundaries and midpoint chapter timelines come from the existing LCDS
  predictions. All predicted chapters are retained. There is no GT-based
  chapter filtering or oracle generation.
- Annotations are evaluation references only. Runtime guards reject oracle
  scope, the old GT-filtered construction, and test input paths in this search.
  Four dependency-free guard tests pass.
- CIDEr/METEOR lexical Hungarian matching is fixed for the entire search to
  audit the original caption-matching definition. This choice is declared
  before any candidate run; the earlier temporal-matching score remains on
  record. SODA always uses all predicted events and actual reference times.
  This profile is not silently interchangeable with either earlier profile.

## Stage 1 grid

Two existing AVLecture-domain generators are evaluated with the same Qwen
3B base: the recovered cross-attention/Q-Former model and the existing
text-only model. Each receives four settings:

| Setting | Beams | Repetition penalty | CSSE top-k |
| --- | ---: | ---: | ---: |
| Baseline | 3 | 1.005 | 3 |
| Wider beam | 5 | 1.005 | 3 |
| More support | 3 | 1.005 | 5 |
| More repetition control | 3 | 1.05 | 3 |

Other settings, retrieval backend, input features, and seed are held fixed.
`scripts/sweep_existing_generators_dev.py` records the exact dev and boundary
hashes, settings, per-run PIDs/logs, and results. It stops on a failed candidate
and does not silently retry. No new checkpoint training is needed for this
stage. The existing stronger checkpoints remain preserved.

## Superseded diagnostic

The historical paper-code-profile comparison was explicitly stopped before
completion because its old data builder GT-filters predicted chapters before
generation. It must not be restarted under the user's new no-GT constraint.
This is a deliberate cancellation, not a crashed model or a completed result.

After stage 1, inspect actual dev cases, confirm promising combinations on all
35 dev videos, then expand to additional existing segmenters or boundary
thresholds if warranted. No fabricated success criterion or predetermined
score can replace the measured result. Formal release/main integration stays
on hold until the comparable target and clone acceptance are satisfied.
