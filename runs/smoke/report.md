# Run report: smoke

- Prompt: `baseline_v0` (sha `8097ebde279ec2d8`), output format `number`
- Model and settings (unset = API default): {'model': 'google/gemini-3.1-pro-preview', 'backend': 'openrouter'}
- Model version(s) the API reported, with call counts: {'google/gemini-3.1-pro-preview @ Google': 1}
- Pairs: 1 gated pairs (1 in the gate view), 1 runs in total

## Gate view (annotator pairs: each molecule's wrong and corrected drawing vs its reference)

Cross-molecule pairs are trivially easy negatives, so they are left out of the bound below and listed under 'By pair category'.

| Measure | Result | Interval / note |
|---|---|---|
| Wrong drawings accepted (a pair counts if ANY run accepted it) | 0/1 | exact 95% CI [0.0%, 97.5%]; true rate below 95.0% with 95% confidence |
| Correct drawings sent back (a pair counts if ANY run rejected it) | 0/0 | exact 95% CI  |
| Wrong accepted, per run | 0/1 = 0.0% [0.0%, 0.0%] | cluster bootstrap over pairs |
| Correct rejected, per run | n/a | cluster bootstrap over pairs |
| 'Cannot verify' (2), per run | 0/1 = 0.0% [0.0%, 0.0%] |  |
| Malformed output, per run | 0/1 = 0.0% [0.0%, 0.0%] | format compliance |
| Blocked or empty response, per run | 0/1 = 0.0% [0.0%, 0.0%] | counted as a non-accept, kept in the denominator |
| Strict accuracy, per run (2, malformed and blocked count as wrong) | 1/1 = 100.0% [100.0%, 100.0%] |  |
| Accuracy when a decision was given (0 or 1) | 100.0% | coverage 100.0% |
| Stability (same decision across repeated runs) | n/a mean agreement | 0/0 pairs unanimous |

How many wrong drawings it takes to certify a false-accept rate (95% confidence, zero false accepts allowed): below 10.0%: 29, below 5.0%: 59, below 2.0%: 149. This run has 1.

## Confusion matrix (runs, annotator pairs)

| expected \ model output | 0 (reject) | 1 (accept) | 2 (cannot verify) | malformed | blocked / empty | API error |
|---|---|---|---|---|---|---|
| should reject | 1 | 0 | 0 | 0 | 0 | 0 |
| should accept | 0 | 0 | 0 | 0 | 0 | 0 |

## Production rule: how many calls per drawing, and how many must say 1

This run made 1 call(s) per pair. A rule only applies if production makes the same number of calls per drawing; each extra call costs money and time. Errors, 2, malformed and blocked answers all count as 'not 1'.

| Rule | Wrong drawings accepted | Correct drawings sent back | Calls per drawing |
|---|---|---|---|
| 1 call per drawing (each run on its own) | 0/1 runs | 0/0 runs | 1 |

## By pair category

| group | pairs | runs | wrong accepted (pairs) | correct rejected (pairs) | unsure runs | strict accuracy |
|---|---|---|---|---|---|---|
| annotated_wrong | 1 | 1 | 0/1 | - | 0.0% | 100.0% |

## By split

| group | pairs | runs | wrong accepted (pairs) | correct rejected (pairs) | unsure runs | strict accuracy |
|---|---|---|---|---|---|---|
| unassigned | 1 | 1 | 0/1 | - | 0.0% | 100.0% |

## Pairs that were not fully correct

None.

**How to read this.** Decision 1 = accept, 0 = reject, 2 = cannot verify. Only an exact 1 is an accept. Zero false accepts on a small sample bounds the true rate but does not prove it is zero; the one-sided bound above shows how far the sample can certify.
