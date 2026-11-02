# Reproducibility audit (2026-11-02)

This document distinguishes recovered artifacts, validated behavior, and
remaining release blockers. It must not be read as a successful reproduction
of every paper table.

## Verified artifacts

- The released generation LoRA is byte-identical to the historical best
  checkpoint: SHA-256
  `080b4a95fc9f486ebe0bc61faee32c5acbe141a0a28944ba5fd33bbba4a65df8`.
- The Qwen boundary adapter and classifier have been recovered; their hashes
  are in `MANIFEST.sha256`.
- Replaying all 70 videos from the historical probability artifact reproduces
  the committed boundary JSONL byte for byte:
  `a6c0b0b54bb186acb95bc5d9bcb4942880f767473922def770ef8f1451ad4ffa`.
- Decoding excludes the final utterance before probability-based suppression,
  uses threshold 0.35, and suppresses peaks less than 6 utterances apart.
- All currently released artifact checksums pass locally.

## Different segmentation model families

The Qwen boundary checkpoint is an end-token classifier. It is not the
historical Longformer + CLIP causal model. These models must not be labeled as
the same paper experiment.

The recovered Qwen postprocessing log reports F1=50.64, F1@30=73.57,
BS@30=99.08, and tIoU=81.18 using the legacy evaluator. The legacy F1@30 uses
utterance-index tolerance, not seconds. The BS implementation can change with
the availability or failure of the optional `segeval` backend, and its fallback
is a separate formula. Those scores cannot be compared across evaluator
versions without recording which implementation was used.

## Historical table discrepancy

Recovered experiment notes contain this row, with explicit column headings:

| Model | F1 | BS@30 | F1@30 | mIoU |
| --- | ---: | ---: | ---: | ---: |
| causal video + text | 56.72 | 72.97 | 67.38 | 69.95 |

The manuscript instead associates F1@30 with 72.97 and reports BS@30=63.02
elsewhere. Reproducing that manuscript row requires the original prediction
artifact and metric implementation, or an evidence-backed correction; swapping
column names or tuning on test labels is not a valid resolution.

## Measured PPU reruns

The isolated Python 3.12 / Torch 2.9.0 PPU environment with Transformers
4.57.1 and PEFT 0.15.2 completed neural boundary inference on all 70 test
videos. Two reruns produced identical boundary files, SHA-256
`5b560e90959b6cbaa45bba8685ab827285a835d9fe6c223f91d56ee0c04b0981`.
They differ from the historical reference on eight utterance labels across
six videos. Legacy scores are F1=50.64, F1@30=73.40, BS@30=99.07,
tIoU=81.38. Exact probability replay is therefore not evidence of bitwise
neural inference reproducibility across hardware.

Generation evaluated all 70 videos with the historical fixed boundary file:
CIDEr=108.89, METEOR=19.89, SODA_c=21.82, versus historical saved scores
111.19 / 20.18 / 21.66. All 301 generated titles were nonempty.
Raw results are saved under `outputs/verified_ppu_20261102/`; this was not
an end-to-end run using the newly recomputed boundary file.
Of the 301 titles, 240 exactly match the historical strings and 61 differ.
For example, “Pion Decay: Chiral vs. Helicity States” becomes “Pion Decay
and Chiral vs. Helicity States”, whereas “Spinor projection and gauge-coupling
unification” becomes “Reduction of vector–axial coupling via spinor
projection”. The former is largely wording drift; the latter changes topic
emphasis. Thus aggregate metric proximity alone does not establish identical
behavior. These test comparisons are diagnostic only and are not used to
select the new segmentation model or tune its hyperparameters.

The historical generation visual cache covers only two of the 70 test
videos; its loader substitutes zero visual features for missing videos.
The packaged historical cache preserves this behavior rather than silently
substituting a different experiment. A separate complete cache containing
173,730 utterance features has now been recovered from OSS for the new
Longformer reconstruction.

The exact historical causal checkpoint was not located in the inspected
backups. The new LCDS implementation passed eleven unit tests and a real
video encoder/head forward-backward smoke test. Training from the pinned
official Longformer initialization has started on train/dev only. This is a
new manuscript reconstruction, not a recovered historical checkpoint; its
quality must be assessed separately from the historical paper run.

## New LCDS development run (not final test)

Ten epochs completed successfully. The selected epoch 10 checkpoint gives
internal-label F1=70.88 and time-domain F1@30s=77.89 on 35 development
videos using the initial threshold 0.35 / gap 8 / utterance-end decoding.
No test split was loaded by training. Only the best checkpoint is retained.

Development cases expose a localization issue: even perfect processed labels
placed at utterance ends have 6.75-second mean error against annotated chapter
times, with only 34.81% within five seconds. Some positive utterances span
the actual transition rather than ending at it. A midpoint decoder does not
consult chapter annotations at inference and reduces this alignment-only
mean error to 3.59 seconds on dev.

An explicit 399-candidate **dev-only** search selects threshold 0.75, gap 8,
and midpoint localization. It yields internal F1=70.63, F1@5s=56.51,
F1@10s=75.84, F1@30s=78.07, count MAE=0.89, and matched MAE@30s=3.63s.
At the same threshold/gap, end localization gives F1@5s=22.30 and
matched MAE@30s=7.53s. This is a changed, explicitly recorded decoder—not
an improvement in the underlying utterance-label classifier. The evaluation
entry point exports actual chapter intervals for the selected anchor so the
reported localization must also be used by downstream inference. The frozen
final test is reported below; new-LCDS end-to-end generation remains pending.

## Frozen final test and clone run

The frozen development reproduction gate passed before test execution, with
the checkpoint and decoder hashes recorded in
`outputs/acceptance_20261102/final_test_frozen.json`. One new-LCDS final test
completed on 70 videos: internal P/R/F1=63.71/55.18/59.14, F1@5/10/30s=
47.31/63.80/69.18, count MAE=1.34, matched MAE@30s=4.60s. These results
are lower than dev and are not used to select another threshold/checkpoint.
They are a new model result, not confirmation of the historical 56.72 row.

The independent GitHub clone also completed its Qwen boundary-to-title run:
CIDEr=107.02, METEOR=19.75, SODA_c=21.44, with 300 nonempty generated
titles. These are **legacy-protocol** scores, not strict end-to-end scores.

## Generation evaluation protocol limitation

Inspection of the legacy data builder found that it Hungarian-matches
predicted segments against annotations *before generation*, drops unmatched
segments, and places reference title events at predicted timestamps. It also
lexically matches generated titles to references for caption scoring. Thus
its historical SODA_c does not independently measure true annotated boundary
timing and must not be advertised as strict end-to-end temporal accuracy.
The preserved historical profile remains available for audit, with this
limitation explicit.

The new opt-in `CC_EVAL_PROTOCOL=strict` path retains all predicted chapters,
uses the actual frozen midpoint decoder intervals, and supplies every real
annotated GT chapter and timestamp to SODA. Caption matching uses temporal
IoU rather than reference-title wording. CIDEr/METEOR still describe matched
caption pairs only; they are not standalone detection-recall metrics. Saved
details include all predicted and reference events. Oracle and new-LCDS
predicted-boundary generation are now being evaluated under this separate
protocol with the complete visual cache. These scores are not directly
comparable to the legacy profile.

## Remaining acceptance checks

- Validate the final release code after its explicit semantic-backend and
  seed settings, and run generation using recomputed neural boundaries.
- Complete train/dev-only LCDS training and case analysis; freeze the selected
  configuration before evaluating the final test split.
- Package the complete visual features required by the new reconstruction,
  separately from the historical two-video generation profile.
- Verify a fresh clone, LFS downloads, a clean environment, and the complete
  evaluation command before declaring the release reproducible.

Cross-device bitwise equality after retraining is a separate requirement from
re-evaluating fixed checkpoint bytes. A release should state hardware and
numeric tolerances supported by actual repeated runs.

## Remote shell diagnostic

The AIS `prun` session produced Git LFS filter/index errors even after all
downloaded bytes passed SHA-256 validation. Direct `git-lfs clean` succeeded;
resetting inherited `SIGCHLD` handling to its default before executing Git
also made the same filtered `git diff --quiet` succeed. This isolates at least
that failure to remote process handling, not corrupted release weights.
For this diagnostic in the affected remote session:

```bash
python3 -c 'import os, signal; signal.signal(signal.SIGCHLD, signal.SIG_DFL); os.execvp("git", ["git", "lfs", "pull"])'
```

This is an environment-specific workaround, not a reason to bypass artifact
hash validation or to disable LFS for ordinary clones.

## Annotation-free usability smoke test

The packaged `segment.infer_lcds` entry point completed a real development
video on CPU using only example ID, transcript, utterance timestamps, and
visual features. No chapter label or annotation fields were passed. Its
decoded boundaries exactly match the frozen dev output; maximum probability
difference versus the PPU run is 0.00004183. It exported contiguous positive
chapter intervals. The saved receipt is
`outputs/acceptance_20261102/release_smoke/acceptance.json`.

This establishes one-video inference usability, not full CPU benchmark speed,
upstream-PyTorch clean-install acceptance, or fresh-GitHub-download validation.
Those distinctions remain explicit. The current user priority is publishing
this usable **AVL-domain** checkpoint, without waiting for a better-than-paper
claim. For other data domains, retrain both segmentation and generation.
