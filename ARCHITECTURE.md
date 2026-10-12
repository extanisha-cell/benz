# Architecture: chemical-structure verification pipeline

Goal: one system prompt that compares Image A (typeset reference) with Image B (annotator's handwritten
copy) on **Gemini 3.1** and outputs `1` accept / `0` reject / `2` cannot verify, with **zero false accepts**.
This repo does not ship the prompt to a platform. It is the harness that *measures* how accurate a prompt is.

## 1. The whole plan on one page

```
 STAGE 1: DATA                STAGE 2: GOLD LABELS             STAGE 3: FREEZE
 ───────────────              ─────────────────────            ───────────────
 samples CSV (17 molecules)   blind copies (neutral names)     split  ->  data/splits.json
   │  Drive / S3 links          │                                 12 dev + 5 held-out
   ▼                            ▼                                 (held-out is locked)
 prep (download.py)           12 blind transcriber batches
   │  magic-byte check          │  (2 independent passes)
   ▼                            ▼
 data/images/<mol>/           6 adjudicators -> by_molecule/<id>.json
   ref.png                      │
   wrong.jpg                    ▼
   correct.jpg                6 verifiers (refuters) -> verification/<id>.json
   + manifest.json              │
   (signed URLs stripped)       ▼
                              gold merge -> errors.json, transcriptions.json, review_table.md
                              gold csv   -> gold_labels.csv (one row per molecule)
                                                │
                                                └─ you read review_table.md before trusting it

 STAGE 4: EVALUATE A PROMPT                                  STAGE 5: DECIDE
 ──────────────────────────                                  ───────────────
 prompts/<candidate>.txt ─┐                                  compare runs side by side
                          ▼                                  │
 pairs.py builds labelled pairs ──► runner.py ──► Gemini 3.1 Pro ▼
   annotated_wrong    expect 0     │   N rounds   via gemini.py or   pick prompt + policy for "2"
                                   │   --budget-usd cap  openrouter.py
   annotated_correct  expect 1     │                          │
   cross_molecule     expect 0     ▼                          ▼
   identity           diagnostic   runs/<name>/outputs.jsonl  final test on held-out
                                   │  (resumable)             (--allow-heldout, logged)
                                   ▼
                      parse.py (fail-closed: only exact "1" = accept)
                                   ▼
                      metrics.py + stats.py ──► report.py ──► report.md
                        false-accept count        exact binomial,
                        false-reject rate         cluster bootstrap
                        "2" share

 PERCEPTION PROBE (separate track, same runner)
 perception_probe.txt ─► Gemini transcribes ONE image to a fixed schema
                      ─► perception.py scores each field vs gold transcriptions
                      => tells you "misread the drawing" apart from "misjudged the pair"
```

## 2. Module map (`src/chemverify/`)

```
                        run.py  (python -I run.py <command>)
                           │
                        cli.py  ── commands: status, models, prep, split, pairs, eval,
                           │                 probe, score-perception, report, compare,
                           │                 gold {prepare-blind|merge|csv|validate}, test
        ┌──────────┬───────┼────────┬────────────┬────────────┬───────────┐
        ▼          ▼       ▼        ▼            ▼            ▼           ▼
   download.py  manifest.py  splits.py  pairs.py  runner.py  perception.py  gold.py
        │          │                       │         │  │         │            │
   imageinfo.py    │                       │         │  │         │            │
   (sniff format,  │                       │         │  │         │            ▼
    no decoding)   │                       │         │  │         │       taxonomy.py
        └────┬─────┘                       │         │  │         │       (error categories)
             ▼                             │         │  │         │
          paths.py  ◄──────────────────────┘         │  │         │
        (ROOT, DATA, IMAGES, GOLD, RUNS, ROLES)      │  │         │
                                                     │  └─► parse.py (output parsers)
                                                     │  └─► prompts.py (load + hash prompt)
                                                     ▼
                                                 gemini.py  (REST client, retries,
                                                             rate limit, key redaction)
                                                    ▲
                                                 openrouter.py  (subclass: same retries and
                                                                 redaction, OpenRouter's API)

   metrics.py ─► stats.py        report.py reads metrics output
   perception.py ─► stats.py
```

| Module | Job |
|---|---|
| [cli.py](src/chemverify/cli.py) | Every command; wires the modules together |
| [gemini.py](src/chemverify/gemini.py) | Google `generateContent` over REST, stdlib only; shared client with retries, rate limiting and key redaction; key lookup (key file, then environment) |
| [openrouter.py](src/chemverify/openrouter.py) | Same client interface for OpenRouter's chat-completions API (`google/gemini-3.1-pro-preview`); the backend is chosen from the key prefix |
| [runner.py](src/chemverify/runner.py) | Runs prompt over tasks, writes `outputs.jsonl`, resumes, refuses to mix prompts or settings, `--max-calls` guard |
| [pairs.py](src/chemverify/pairs.py) | Builds labelled pairs and their expected answer |
| [parse.py](src/chemverify/parse.py) | Strict parser: line 1 must be exactly `0`, `1` or `2` |
| [metrics.py](src/chemverify/metrics.py) | Fail-closed scoring; a wrong drawing is falsely accepted if **any** run accepted it |
| [stats.py](src/chemverify/stats.py) | Clopper-Pearson, Wilson, cluster bootstrap over pairs |
| [report.py](src/chemverify/report.py) | Markdown report and run comparison |
| [perception.py](src/chemverify/perception.py) | Single-image transcription schema and field-by-field scoring |
| [gold.py](src/chemverify/gold.py) | Blind copies, schema validation, merge; recomputes verification status from evidence |
| [splits.py](src/chemverify/splits.py) | Frozen dev / held-out split, never overwritten |
| [download.py](src/chemverify/download.py), [manifest.py](src/chemverify/manifest.py), [imageinfo.py](src/chemverify/imageinfo.py) | Allow-listed hosts, magic-byte format check, signed URLs stripped before saving |

## 3. Safety rules built into the design

| Rule | Where it is enforced |
|---|---|
| Only an exact `1` is an accept; `2`, malformed output, blocked or empty responses and API errors are not | parse.py, metrics.py |
| Held-out molecules are untouchable while tuning | splits.py, `eval` needs `--allow-heldout`, access logged to `runs/_heldout_access.log`; `report` and `compare` drop held-out pairs before computing any number (`splits.drop_heldout`) |
| Downloaded files are untrusted | download.py keeps image bytes only, nothing executed or imported |
| The API key never enters the project or OneDrive | read from `C:\Users\anysh\.chemverify.env` (outside OneDrive) or an environment variable, sent only in the auth header, redacted from every error |
| A run cannot silently change prompt or settings | runner.py checks the prompt hash and settings in `meta.json` |
| Gold labels are not trusted blindly | blind transcription, adjudication, then an independent refuter, then `gold merge` |

## 4. Known weaknesses

Found in a design review on 2026-10-11. The first four were code flaws and are fixed. The rest are limits
of the plan: code can measure or reduce them but not remove them, and the table says which is which.

| # | Weakness | Status |
|---|---|---|
| 1 | The false-accept bound counted trivially easy cross-molecule pairs, so it looked tighter than the real task supports (8.4% instead of 16.2% on 17 molecules) | **Fixed.** The gate view uses annotator pairs only ([metrics.py](src/chemverify/metrics.py) `headline`); cross-molecule shows as a separate sanity row |
| 2 | Held-out results leaked through totals and `compare`; the README baseline ran on `--split all --allow-heldout` | **Fixed.** Held-out pairs are dropped before any number is computed; baseline commands now use `--split dev` |
| 3 | Gold-flagged issues in "correct" drawings never reached the metrics | **Fixed.** The report lists them; an accept of a flagged drawing is called out as a possible false accept |
| 4 | Blocked and empty responses were dropped from the denominators like transient errors | **Fixed.** They are their own outcome, count as non-accepts and stay in every denominator |
| 5 | Too few wrong drawings to certify "no wrong accepted": 12 dev drawings only show the rate is below about 22%, and the held-out test is 5 drawings once | **Quantified, not fixable in code.** Each report now says how many wrong drawings certify 10%, 5% and 2% (29, 59, 149). Only more labelled wrong drawings close the gap; until then the safeguard has to come from how the prompt is used in production |
| 6 | The harness scored "any of N runs accepted", but production at one call per drawing behaves like the per-run rate; unanimity is stricter and sends more correct drawings back | **Mitigated.** The report simulates "at least k of N calls say 1" for every k, beside the one-call rate ([metrics.py](src/chemverify/metrics.py) `vote_rules`). **Still your decision:** which rule production uses and how many calls it can afford |
| 7 | The platform model and settings are unconfirmed (Pro vs Flash-Lite, thinking level, image resolution), and `gemini-3.1-pro-preview` is a preview alias | **Mitigated.** Reports list the model versions the API returned and warn on a mixed run; `compare` shows each run's settings; the README has a checklist of what to ask the platform owner. **Partly closed:** the platform runs Gemini 3.1 Pro (confirmed by the user 2026-10-11), which matches the harness default family. **Still open:** the exact model ID (the harness uses the `gemini-3.1-pro-preview` alias), thinking level, image resolution, and calls per drawing |
| 7b | The only key available is an OpenRouter key, so calls reach Gemini 3.1 Pro through OpenRouter rather than Google's own API: OpenRouter picks the provider, thinking is set through `reasoning.effort`, and there is no media-resolution setting | **Mitigated, not removed.** Each record stores the provider; `--route-to` pins one; the report warns on mixed versions. A Google AI Studio key would remove the difference. Worth a side-by-side run if the platform calls Google directly |
| 7c | Cost: one default-thinking call was $0.22 and 3 minutes (18,000 reasoning tokens); a 180-call baseline would be about $40 | **Mitigated.** Measured on one pair: low $0.014 / 10 s, medium $0.029 / 56 s, same verdict. `--budget-usd` caps a run (new calls stop at the budget; overshoot at most `--workers - 1` calls) and each report prints the cost. **Still open:** which thinking level the platform uses, since a cheap run at a different level measures a different setup |
| 8 | Error localisation and perception gold come from agents, not people; `review_table.md` is the only human check | **Open, left alone on purpose.** The only code fix is a mandatory human sign-off, which adds the review effort you want to avoid. `gold merge` already lists the rows that need a human look; read those before trusting `split --stratify-gold` or perception scores |
| 9 | Black-ink compliance is a pixel property the model has to judge by eye; the harness never decodes pixels | **Not done.** Decoding JPEG needs an image library and the harness is standard-library only (Pillow is not installed). The perception probe already scores how well the model reads ink colour against gold, so the risk is measured, not removed. A code-side check belongs on the platform |

### Found in the second review (after the gold labels and the OpenRouter backend existed)

| # | Weakness | Status |
|---|---|---|
| 10 | No way to correct an expected label. `pairs.py` always expects `1` for a corrected drawing, but gold says molecule_294's "corrected" drawing is not correct (imidazole where the reference has a 1,2,4-triazole). The report calls it out; the headline counts still treat a reject of it as a false reject | **Open, needs your decision.** Fix would be a small overrides file (`pair -> expected 0, or excluded, with a reason`) applied when pairs are built |
| 11 | The perception probe ignores the split: it runs on all 17 molecules and its scores are not hidden for held-out ones, so it shows how well Gemini reads the held-out drawings before the final test | **Open.** Fix: probe only dev molecules unless `--allow-heldout`, and drop held-out images from `score-perception` unless `--show-heldout` |
| 12 | No acceptance criteria are written down anywhere. The harness measures, but nothing says what result means "good enough", which invites moving the goalposts after seeing the numbers | **Open, needs your decision.** Candidate criteria belong in this file before the first baseline. With only 5 held-out molecules a clean result can only rule out a clearly bad prompt (zero false accepts out of 5 means the true rate is under about 45%), so the criteria have to say what else production does to cover the gap |
| 13 | The stratified split puts every rare error type in the held-out set (simulated: dev is 9 of 12 ring-position errors, and missing-atom, extra-group, wrong-label and connectivity examples are mostly held-out), and it would put molecule_195, a molecule flagged for review, in the held-out set. The prompt cannot be tuned on types it never sees in dev, and held-out labels should be the most solid ones | **Open.** Options: keep flagged molecules out of the held-out candidates, or require at least one of each error type in dev. Decide after you review the flagged rows |
| 14 | The same OpenRouter key is still stored in the Windows user variable `GEMINI_API_KEY` (73 characters) as well as in the key file. Any program on this account that reads `GEMINI_API_KEY` would send an OpenRouter key to Google | **Open, your system setting.** Safe to remove, since the key file wins: `[Environment]::SetEnvironmentVariable("GEMINI_API_KEY", $null, "User")` |
| 15 | The perception probe is a required measurement but is not in the $2 plan: the 120-call baseline is about $1.70 and a one-run probe on 51 images is about another $0.50 | **Open.** Budget about $2.50, or probe only the 12 dev molecules (about $0.35) |
| 16 | Housekeeping: `runs/` holds five experiment folders, and `GeminiClient`, `GeminiResult` and `GeminiError` are reused for OpenRouter under their old names | **Minor.** Move the experiments aside; rename when convenient |

## 5. Where the project stands (from the files on disk, 2026-10-11)

| Stage | State |
|---|---|
| Harness code and 9 test files | written; `python -I run.py test` passes (151 tests) |
| Stage 1: images | downloaded: 17 molecules, 51 images, `manifest.json` present |
| Stage 2: gold labels | **Done, awaiting your review.** 12 blind transcription batches (two independent passes), 6 adjudicators, 6 independent verifiers; all 17 molecules adjudicated and verified, merged into `errors.json`, `transcriptions.json`, [review_table.md](data/gold/review_table.md) and [gold_labels.csv](data/gold/gold_labels.csv). All labels were written by Claude agents, not people. 6 molecules are flagged for a human look. 3 of the 17 corrected drawings have a recorded issue: molecule_294 (high: imidazole instead of the reference's 1,2,4-triazole) and molecule_130 and molecule_16 (low-severity legibility notes). 6 of the 17 wrong drawings also have notation errors on top of the main mistake |
| Stage 3: split | `data/splits.json` does **not** exist yet; `split --stratify-gold` is meant to run after `gold merge` |
| Stage 4: evaluation | experiments only, all through OpenRouter and all run before any split existed: `runs/smoke`, `cost_low`, `cost_med` (the same pair, BZ2256__wrong, at three thinking levels: same verdict `0`), `cost_batch` (a failed attempt: the half-price batch model is refused), `budget_check` (4 reference-against-itself pairs, all accepted). No prompt has been scored on the labelled set yet |
| Prompts | `baseline_v0.txt`, `baseline_auditor_verbatim.txt`, `perception_probe.txt` |

Next steps in order: read the flagged rows of the review table (especially molecule_294, whose "corrected"
drawing has an imidazole where the reference has a 1,2,4-triazole), freeze the split with
`split --stratify-gold`, run the baseline on `--split dev` with `--budget-usd`, then iterate on the dev set.
