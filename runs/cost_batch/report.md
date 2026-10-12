# Run report: cost_batch

- Prompt: `baseline_v0` (sha `8097ebde279ec2d8`), output format `number`
- Model and settings (unset = API default): {'model': 'google/gemini-3.1-pro-preview:batch', 'thinking_level': 'low', 'backend': 'openrouter'}
- Pairs: 1 gated pairs (1 in the gate view), 1 runs in total
- **WARNING: 1 runs failed with API errors and are excluded below. Re-run to complete them.**

## Gate view (annotator pairs: each molecule's wrong and corrected drawing vs its reference)

Cross-molecule pairs are trivially easy negatives, so they are left out of the bound below and listed under 'By pair category'.

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

## Confusion matrix (runs, annotator pairs)

| expected \ model output | 0 (reject) | 1 (accept) | 2 (cannot verify) | malformed | blocked / empty | API error |
|---|---|---|---|---|---|---|
| should reject | 0 | 0 | 0 | 0 | 0 | 1 |
| should accept | 0 | 0 | 0 | 0 | 0 | 0 |

## Production rule: how many calls per drawing, and how many must say 1

No answered runs.

## By pair category

| group | pairs | runs | wrong accepted (pairs) | correct rejected (pairs) | unsure runs | strict accuracy |
|---|---|---|---|---|---|---|
| annotated_wrong | 1 | 1 | - | - | n/a | n/a |

## By split

| group | pairs | runs | wrong accepted (pairs) | correct rejected (pairs) | unsure runs | strict accuracy |
|---|---|---|---|---|---|---|
| unassigned | 1 | 1 | - | - | n/a | n/a |

## Pairs that were not fully correct

| pair | expected | decisions per run | outcomes | first reason given |
|---|---|---|---|---|
| BZ2256__wrong | 0 | - | errorx1 |  |

**How to read this.** Decision 1 = accept, 0 = reject, 2 = cannot verify. Only an exact 1 is an accept. Zero false accepts on a small sample bounds the true rate but does not prove it is zero; the one-sided bound above shows how far the sample can certify.
