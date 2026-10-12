# Run report: budget_check

- Prompt: `baseline_v0` (sha `8097ebde279ec2d8`), output format `number`
- Model and settings (unset = API default): {'model': 'google/gemini-3.1-pro-preview', 'thinking_level': 'low', 'backend': 'openrouter'}
- Model version(s) the API reported, with call counts: {'google/gemini-3.1-pro-preview @ Google': 4}
- Cost the API reported: $0.05 over 4 calls (about $0.012 per call)
- Pairs: 0 gated pairs (0 in the gate view), 0 runs in total

## Gate view (gated pairs; no annotator pairs in this run)

| Measure | Result | Interval / note |
|---|---|---|
| Wrong drawings accepted (a pair counts if ANY run accepted it) | 0/0 | exact 95% CI ; true rate below n/a with 95% confidence |
| Correct drawings sent back (a pair counts if ANY run rejected it) | 0/0 | exact 95% CI  |
| Wrong accepted, per run | n/a | cluster bootstrap over pairs |
| Correct rejected, per run | n/a | cluster bootstrap over pairs |
| 'Cannot verify' (2), per run | n/a |  |
| Malformed output, per run | n/a | format compliance |
| Blocked or empty response, per run | n/a | counted as a non-accept, kept in the denominator |
| Strict accuracy, per run (2, malformed and blocked count as wrong) | n/a |  |
| Accuracy when a decision was given (0 or 1) | n/a | coverage n/a |
| Stability (same decision across repeated runs) | n/a mean agreement | 0/0 pairs unanimous |

How many wrong drawings it takes to certify a false-accept rate (95% confidence, zero false accepts allowed): below 10.0%: 29, below 5.0%: 59, below 2.0%: 149. This run has 0.

## Confusion matrix (runs, gated pairs)

| expected \ model output | 0 (reject) | 1 (accept) | 2 (cannot verify) | malformed | blocked / empty | API error |
|---|---|---|---|---|---|---|
| should reject | 0 | 0 | 0 | 0 | 0 | 0 |
| should accept | 0 | 0 | 0 | 0 | 0 | 0 |

## Production rule: how many calls per drawing, and how many must say 1

No answered runs.

## Corrected drawings that gold labelling says still have an issue

None of the 0 corrected drawings has a recorded issue.

## By pair category

| group | pairs | runs | wrong accepted (pairs) | correct rejected (pairs) | unsure runs | strict accuracy |
|---|---|---|---|---|---|---|
| identity | 4 | 4 | - | 0/4 | 0.0% | 100.0% |

## Pairs that were not fully correct

None.

**How to read this.** Decision 1 = accept, 0 = reject, 2 = cannot verify. Only an exact 1 is an accept. Zero false accepts on a small sample bounds the true rate but does not prove it is zero; the one-sided bound above shows how far the sample can certify.
