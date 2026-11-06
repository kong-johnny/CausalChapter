# Clean train-only generator experiment

Status (2026-11-03): epoch1 was selected on dev and accepted by the user for
release. Epoch2 completed but regressed on dev; epoch3 was stopped at the user's
request. The released weights are the preserved epoch1 snapshot. Full70 test
evaluation completed. Raw records are in `outputs/clean_generator_20261103/`.

## Why a separate entry point

The historical training builder mixes train and dev, and its training loop
uses test for checkpoint selection. The old path remains for audit, not as a
source for new clean-holdout claims. `generate/train_clean.py` does not import
`generate/train.py` and cannot warm-start from an old adapter.

The first controlled configuration uses a fresh Qwen2.5-3B base with new
rank-32 LoRA, text-only fusion, up to three planned epochs, learning rate 2e-5, accumulation
4, 4096 training tokens, 8192 inference tokens, beam 3, repetition 1.005,
CSSE top-3, hashing candidate retrieval, seed 42. These are a predefined clean
baseline, not parameters selected from the quarantined historical dev sweep.
It retains two-pass generation and KL removal-intervention CSSE. Visual input
remains in the LCDS segmenter; a controlled visual-generator variant can be
tested later without changing the split or metric protocol.

## Input and label isolation

1. Original 245 train and 35 dev videos must have disjoint canonical IDs and
   normalized full transcripts. IDs ending `__predaug` are canonicalized for
   overlap checks. This does not rule out every near-duplicate.
2. The frozen LCDS model predicts **train** boundaries without using training
   boundary labels during inference. Actual frozen dev boundary/timeline
   artifacts are supplied separately. Both artifacts must exactly cover their
   split, have matching boundary indices, contiguous positive intervals, and
   the complete video duration. The segmenter itself was supervised on train;
   these train predictions are not out-of-fold predictions, which is disclosed.
3. Generation and CSSE receive **every predicted chapter**, including unmatched
   chapters, and no reference-title fields. Reference changes cannot change
   their segment text, spans, or timestamps. No GT boundary builder is used as
   a generation input. Few-shot examples are disabled.
4. During gradient training only, each predicted chapter can receive a teacher
   title from its highest time-IoU training reference (fixed minimum IoU 0.3).
   This is training-label assignment, not GT-based generation filtering.
   Pass-2 CSSE generates all chapters before loss-target assignment. Both
   passes train only on train examples; dev never enters either loss dataset.
5. Every epoch evaluates all predicted dev chapters. CIDEr is the primary
   selection metric, with SODA_c only a tiebreak; all metrics remain recorded.
   Lexical caption matching is explicit and fixed. SODA uses all predictions
   and real GT event times. This profile is not interchangeable with the
   earlier temporal-caption score or the legacy GT-prefiltered profile.
6. No test annotation/transcript path or oracle generation is used by this
   entry point. Test reuse, if any after a separately frozen dev decision,
   must be disclosed and cannot select this model.

## Commands and artifacts

After activating the intended environment, set `CC_CLEAN_GENERATOR_ROOT` to
a new absolute task directory, `CC_LCDS_CHECKPOINT` to the frozen LCDS release
bundle, `CC_FULL_VIS_CACHE`, `CC_BASE_MODEL` to an official complete base,
`CC_LCDS_DEV_BOUNDARIES`, and `CC_LCDS_DEV_TIMES`. Run:

```bash
bash scripts/run_clean_generator.sh
```

The script first predicts train chapters, then performs a one-step smoke of
each gradient pass with full CSSE generation for one train video, then starts
a **fresh** formal training instance. Smoke weights are never used as the
formal initialization and no smoke checkpoint is saved. Use a detached PID
and a unique log on AIS; inspect existing processes before starting.

The independent CLI `python -m generate.train_clean --help` documents explicit
arguments; `--audit-only` validates split and artifact structure without ML
imports. Input/base-file SHA-256 hashes, package versions, settings, and
initialization policy are saved. Base hashes establish byte identity, not
official provenance by themselves; official-source verification is still a
release requirement. The sole best checkpoint is transactionally replaced;
only this trainer's exact superseded `previous_best` directory is removed.
Interrupted checkpoint transactions require inspection, not blind overwrite.
There is no claim of exact optimizer-state resume.

Local validation: six pure-Python generation-input tests, three trainer-input
audit tests, and four split-audit tests pass; CLI help and syntax compilation
pass. These and four usage guards also passed on AIS. The real GPU smoke
completed one update per pass (losses 1.96763 and 4.41859), generated all
predicted chapters of the smoke train video, did not score dev, and saved no
checkpoint. The formal run reinitialized from the base afterward; two complete
epochs and their dev evaluations are retained in the experiment records.

The frozen segmenter generated 1,403 train chapters over 245 videos; 1,371
receive loss targets under the fixed training-IoU rule. An initial pre-model
failure exposed an empty trailing transcript chapter: the old segment builder
dropped it, breaking the complete timeline. Strict construction now preserves
empty predicted chapters, with a regression test; legacy construction keeps
its original behavior. Prepared predictions were reused after the failed
parent exited, but no generator checkpoint/optimizer state was resumed.
Recovery logs preserve the original failure.
