# Clean generator epoch 1: predicted-boundary dev review

This is a preliminary development review, not a test result or a paper-win claim.
Source: `metrics_epoch1.json` and `eval_details_dev_pred_epoch1.jsonl` downloaded
from the active run on 2026-11-03. The unchanged run continues into epoch 2.

## Complete dev result

- 35 videos, 169 predictions, zero empty titles, 170 reference events.
- CIDEr 99.6889, METEOR 20.8240, SODA_c 18.0033.
- Strict events and real timestamps; lexical caption matching, 154 pairs.
- Chapter counts: 10 videos under-segmented, 15 equal-count, 10 over-segmented.
  Equal counts do not establish correct boundaries. Four predicted chapters
  exceed 1,800 seconds; this alone does not establish truncation.
- Epoch-1 mean losses: pass1 1.68299, pass2 1.28519. Loss is not quality proof.

## Inspected cases and limited inferences

1. `mit126@@MIT2_003SCF11_lec13_300k`: two predictions versus four GT chapters.
   The first predicted interval is 0–3407.22s; reference changes occur at
   435s and 1965s before the next reference boundary at 3413s. Its title,
   "Rigid-body angular momentum and the general parallel-axis theorem",
   emphasizes angular momentum/inertia but does not name the reference's
   tipping-box problem. This is a concrete merged-chapter/content-coverage
   failure, not evidence that beam search alone will recover missing chapters.
2. `mit126@@MIT2_003SCF11_lec19_300k`: four predictions versus six references.
   The final 2396.28–4475.34s interval merges damping behavior and experimental
   estimation. The title names damping classification and logarithmic decrement,
   but one title cannot restore the missing separate event in SODA.
3. `mit159@@MITRES18-009F15_3_300k`: nine predictions versus eight references.
   An extra split at 501.90s falls inside the first 0–1253s reference chapter.
   The first two titles both describe first-order ODE visualization. Later
   "Eigenvalues and Eigenvectors for Second-Order Systems" omits the more
   specific companion-matrix conversion in the corresponding temporal region.
   Title specificity and segmentation are separate candidate bottlenecks.
4. `mit097@@MIT6_042JS15_17_ipod`: three predictions versus three references,
   boundaries 682.68/1479.13s versus 688/1485s. Titles correctly name DAGs,
   scheduling, and Dilworth's lemma, but compress reference details such as
   antichains and the class-size application. This is a useful correctly
   segmented control for later title-specificity comparisons.

Lexical matches can cross temporal regions (visible in the ODE case); do not
interpret those caption pairs as proof of temporally correct chapters. SODA
continues to use all predictions and true reference times.

## Next controlled comparisons, not yet executed

- Finish the predefined three epochs and compare these same dev cases.
- For merged chapters, examine local LCDS scores around missed dev transitions;
  compare thresholds without assuming lower is always better. The separate
  fine dev sweep found threshold .74/gap8/middle F1@30s 78.5185, versus
  .75's 78.0669. Do not silently replace the active run's .75 artifacts.
- For correctly segmented controls, compare coverage/specificity across epochs
  before proposing beam or prompt changes. Inspect token truncation explicitly
  before attributing long-chapter failures to context limits.
- CSSE support IDs and KL scores are present, but this review has not established
  that support selection caused any title error. Do not claim that diagnosis
  without a fixed-input support ablation.

No test input was used for this review. No active training parameters changed.
