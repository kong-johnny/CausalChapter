# Release acceptance: better-than-paper prediction pipeline

## Current user priority: usable AVL checkpoint release

The latest user request is to validate the existing F1=59.14 checkpoint and,
if usable, publish it on GitHub. **Outperforming the paper is no longer a
prerequisite for publishing this usable AVL version.** The earlier
better-than-paper target below remains a separate research goal, not a reason
to withhold a functioning checkpoint. Clearly label the measured protocol and
limits; validate annotation-free inference, positive contiguous chapter
intervals, artifacts and download instructions before declaring it usable.
Release both modules explicitly as AVL-domain weights. Recommend retraining
both modules for migration to another dataset/domain. Unstarted architecture
ablation work is deferred while this release is validated and published.

The user requests results above the paper followed by integration into the
repository. This is a target and a merge gate, not an already-achieved result
or a guarantee that experiments will attain a predetermined score.

The user subsequently authorizes architecture and hyperparameter adaptation,
including dataset-specific settings, as long as the implementation broadly
retains the paper's mechanisms. Bit-for-bit or layer-for-layer duplication is
not required. Preserve LCDS local dependency interventions and CSSE
cross-segment support selection; disclose changed backbones, fusion, losses,
training settings, and computational budgets. Architecture adaptation is not
permission to alter the evaluation to obtain a preferred outcome.

## Comparable targets

The current manuscript contains different configurations:

| Paper configuration | CIDEr | METEOR | SODA_c | F1 | F1@30 | tIoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3B full ablation | 104.59 | 17.87 | 12.73 | 56.72 | — | 69.95 |
| Main-table Qwen | 116.88 | — | 13.80 | — | 72.97 | 69.95 |

Do not quietly substitute the smaller ablation target for the main-table
target. Backbone sizes, data, boundary alignment, endpoint convention,
tolerance units, and metric implementations must be specified before a
like-for-like improvement claim. Metric-column discrepancies in the recovered
notes remain documented in `REPRODUCIBILITY_AUDIT.md`.

## Gates before main integration

1. Use predicted boundaries from the trained segmenter as input to generation.
   Oracle boundaries are diagnostic only and do not satisfy acceptance. Do not
   schedule additional oracle runs as the primary improvement workflow.
2. Compare the final candidate with the appropriate paper configuration under
   the same documented evaluation protocol. Report every target metric, not
   only improvements. Do not compare legacy generation scores with strict
   scores as if they used the same evaluation.
3. Do not remove unmatched predicted chapters or replace real reference times
   with predicted times in a strict end-to-end claim. If the exact historical
   evaluation cannot be recovered, say comparability is unresolved and report
   a corrected baseline under the same new protocol; do not declare a win
   against an incompatible published number.
4. Train and select on train/dev only. The existing frozen final-test result
   stays on record. Do not optimize thresholds, select checkpoints, or adjust
   evaluation rules to improve that test result. Disclose any later reuse of
   the published benchmark; do not portray reused test data as unseen.
5. Ship the exact weights, frozen configuration, tokenizer, required processed
   features, licenses, checksums, and commands. A new GitHub clone must download
   and validate them, and run annotation-free segmentation and the complete
   predicted-boundary generation pipeline. Document supported hardware and
   numerical tolerances; do not promise universal bitwise equality.
6. Once these checks and the better-than-paper comparison genuinely pass,
   integrate the reviewed release branch into the user's repository main
   without overwriting unrelated changes. Preserve the experiment audit trail.
   Until then, keep work on `codex/reproducible-release` and label it unfinished.

Current status: **not accepted**. New LCDS test internal F1=59.14 is not by
itself proof of outperforming paper F1=56.72, because the boundary conventions
are not established as identical. Strict predicted-boundary generation and
fresh-clone acceptance of the new LCDS weights are still pending.
