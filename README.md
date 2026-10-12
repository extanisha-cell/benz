# Chemical-structure verification: prompt harness

Scores system prompts that decide whether a handwritten reproduction (Image B) shows the same
compound as a reference structure (Image A). It is built to run against **Gemini 3.1 Pro**, through
Google's Gemini API or through OpenRouter, and to measure how accurate a prompt really is before it goes
onto your platform.

Standard library only, no installs. Always run it in isolated mode:

```
python -I run.py <command> --help
```

## The output the prompt must produce

```
1            <- line 1: exactly one digit
Same compound; all labels and positions match.   <- line 2: one short reason
```

`1` = equivalent and guideline-compliant (accept), `0` = differs or breaks a guideline (reject),
`2` = cannot verify. Only an exact `1` is ever counted as an accept (fail-closed); malformed output,
`2` and API errors are never accepts. Whether `2` goes to review or is rejected is a policy choice made
after measuring how often it happens.

## Setup

1. Put your key in **`C:\Users\anysh\.chemverify.env`**, one line: `OPENROUTER_API_KEY=<key>` for an
   OpenRouter key (`sk-or-...`), or `GEMINI_API_KEY=<key>` for a Google AI Studio key (39 characters,
   `AIza...`). That file is outside the project on purpose: the project lives in OneDrive, which would
   sync a key to the cloud. Never copy it into the project and never paste it into chat. The harness looks
   in this order: the key file, then the `OPENROUTER_API_KEY` / `GEMINI_API_KEY` / `GOOGLE_API_KEY`
   environment variables (including the Windows user environment, so VS Code need not be restarted). A key
   in the file wins, so a stale variable never blocks it. The key goes only in the request's auth header
   and is redacted from every error message. `status` shows which backend the key selects and warns if the
   key does not look like one for it.
2. Check: `python -I run.py status`, then `python -I run.py models` (lists the models your key can use).
3. Images go to the Gemini API. Use a key whose data-use terms you accept (free-tier keys usually let
   Google use submitted content; paid keys do not).

## Workflow

| Step | Command |
|---|---|
| Unit tests | `python -I run.py test` |
| Download images (once) | `python -I run.py prep --raw-csv <csv outside this folder>` |
| Gold labels (see below), then | `python -I run.py gold merge` |
| Freeze dev / held-out split | `python -I run.py split --stratify-gold` |
| Smoke test, nothing sent | `python -I run.py eval --prompt prompts/baseline_v0.txt --name smoke --split dev --dry-run` |
| One real call | `python -I run.py eval --prompt prompts/baseline_v0.txt --name smoke --split dev --limit 1 --runs 1` |
| Baseline | `python -I run.py eval --prompt prompts/baseline_v0.txt --name base_v0 --split dev --runs 5` |
| Your original prompt as-is | `python -I run.py eval --prompt prompts/baseline_auditor_verbatim.txt --format auditor_json --name base_orig --split dev --runs 5` |
| Develop on the dev set | `python -I run.py eval --prompt prompts/<candidate>.txt --name cand1 --split dev --runs 5` |
| Compare runs | `python -I run.py compare --run runs/base_v0 --run runs/cand1` |
| Final test, once | `python -I run.py eval --prompt prompts/<final>.txt --name final --split heldout --allow-heldout --runs 5`, then `report --run runs/final --show-heldout` |
| Perception probe | `python -I run.py probe --prompt prompts/perception_probe.txt --name probe1` then `python -I run.py score-perception --run runs/probe1` |

Every call is saved to `runs/<name>/outputs.jsonl`. Re-running the same command resumes: finished calls
are skipped and failed ones are retried. A run folder refuses to mix results from a different prompt or
settings. `--max-calls` (default 500) stops an accidental large run before it starts.

## What is measured

* **Pairs.** Each molecule gives two labelled pairs: reference vs the wrong drawing (expected `0`) and
  reference vs the corrected drawing (expected `1`). `cross_molecule` pairs (reference vs a different
  molecule's drawing, expected `0`) are extra negatives. `identity` pairs (reference vs itself) are
  diagnostic only: a coloured reference can fail the black-ink rule.
* **False accept** = a wrong drawing that any run accepted. This is the number that must be zero.
  Intervals are exact binomial over pairs; run-level rates use a bootstrap over pairs, because
  repeated runs of one pair are not independent.
* **The gate view counts annotator pairs only.** Cross-molecule pairs are easy negatives, so they are
  shown as a separate sanity row and under "By pair category", never inside the false-accept bound.
* **Small samples.** With 17 wrong drawings, zero false accepts only shows the true rate is probably
  below about 16% (95% one-sided); with the 12 dev drawings it is about 22%. The report prints that
  bound next to the count.
* **How many wrong drawings would be enough.** The report prints how many wrong drawings must all be
  rejected to certify a false-accept rate below 10%, 5% and 2% (29, 59 and 149 at 95% confidence).
  Only more labelled wrong drawings move that number; the prompt cannot.
* **Production rule.** The report simulates "accept only if at least k of N calls say 1" from the N
  runs it already has, next to the one-call-per-drawing rate. Run with `--runs N` where N is the
  number of calls production will really make per drawing; each extra call costs N times the money and
  time, and the stricter rules send more correct drawings back.
* **Blocked or empty responses** (safety block, empty text) are counted as non-accepts and stay in
  every denominator. Only transient API errors are excluded, and the report warns about both.
* **Corrected drawings the gold agents flagged.** The gold files record anything still off in a
  corrected drawing. Those pairs keep their expected `1`, but the report lists them: an accept of a
  flagged drawing is a possible false accept, and a reject may be justified.
* **Perception.** A separate probe asks the model to transcribe each single image to a fixed schema
  (labels, abbreviations, R-labels, ring sizes, bond counts, stereo marks, ink colour, legibility),
  scored field by field against the gold transcriptions. This separates misreading from misjudging.

## Held-out discipline

`split` freezes 5 molecules as held-out. `eval` refuses to touch them without `--allow-heldout`, and
every access is logged to `runs/_heldout_access.log`. `report` and `compare` drop held-out pairs
before computing anything, so no total, rate or `metrics.json` entry reveals how they did, unless
`--show-heldout` is given (logged). Use that only for the final test. The raw `outputs.jsonl` of a run
that included held-out molecules still holds their answers, so do not run tuning evaluations on them.

## Gold labels

The CSV says which drawing is wrong, not what is wrong. Labelling agents produce that:

1. `gold prepare-blind` copies every image to a neutral name so a transcriber cannot tell roles apart.
2. Two blind agents transcribe each image independently.
3. An adjudicator compares them, reads all three drawings of a molecule and writes
   `data/gold/by_molecule/<id>.json` (final transcriptions and the errors in the wrong drawing).
4. A different agent tries to refute it and writes `data/gold/verification/<id>.json`.
5. `gold merge` validates both, recomputes the verification status from the evidence, and writes
   `data/gold/errors.json`, `transcriptions.json` and **`review_table.md`**, which you should read.
6. `gold csv` writes `data/gold/gold_labels.csv`, one row per molecule (primary error category, every
   error with its severity, issues still in the corrected drawing, confidence, whether a human should
   look, verification status). It opens in Excel and can be re-run any time.

`gold validate --kind transcriptions|adjudication|verification --file ...` lets an agent check its own file.
In the first run, 12 blind transcribers (two passes, 9 images each, one molecule at most per batch),
6 adjudicators and 6 verifiers were Claude agents; no person has checked the labels yet.

## Layout

```
run.py                      entry point
src/chemverify/             the harness (gemini.py = API client, runner.py, metrics.py, perception.py, gold.py ...)
prompts/                    baseline_v0.txt, baseline_auditor_verbatim.txt, perception_probe.txt
tests/                      unit tests (fake transport, no network)
data/                       images, manifest.json, samples.csv (signed URL query strings removed), gold/
runs/                       one folder per evaluation run
```

## Gemini settings

Defaults mirror the API: model `gemini-3.1-pro-preview`, and temperature, thinking level and image
resolution are **not sent** unless you pass `--temperature`, `--thinking-level`, `--media-resolution`.
Google recommends leaving temperature at 1.0 for Gemini 3. Whatever you test with must match what your
platform sends; each run records its exact settings in `runs/<name>/meta.json`. Images are sent exactly
as downloaded, which is what a platform passing image URLs through would give the model.

### Backends: Google directly, or through OpenRouter

The harness picks the backend from the key: `sk-or-...` means OpenRouter (model
`google/gemini-3.1-pro-preview`), anything else means Google's own API (`gemini-3.1-pro-preview`).
Override with `--backend gemini|openrouter` and `--model <id>`. Both give Gemini 3.1 Pro, but OpenRouter
sits in between, so check these before treating a number as "what the platform would do":

* **Provider.** OpenRouter picks the provider unless you pin one with `--route-to <slug>` (copy the slug
  from the model's page; routing then never falls back to another). Each record stores the provider the
  response named (`model_version`, e.g. `google/gemini-3.1-pro-preview @ Google`), and the report warns if
  a run mixed versions or providers.
* **Settings.** Thinking level goes through `reasoning.effort`. There is no media-resolution setting, so
  `--media-resolution` is refused. Unset settings are not sent, and `require_parameters` is switched on
  whenever one is, so a provider that cannot honour it fails loudly instead of ignoring it.
* **Your images leave your machine twice**: to openrouter.ai, then to the provider. `--data-collection deny`
  restricts routing to providers that do not keep or train on data; without it your OpenRouter account
  setting decides. The run prints which one applies.
* **Cost and time are driven almost entirely by thinking tokens** (billed as output, $12 per million; the
  images and prompt are about $0.005). Measured once each on the same pair (`BZ2256__wrong`; one sample per
  level, and it varies by pair, so treat these as rough):

  | `--thinking-level` | Reasoning tokens | Cost per call | Time |
  |---|---|---|---|
  | not set (API default, behaves like high) | 18,256 | $0.225 | 177 s |
  | `medium` | 1,923 | $0.029 | 56 s |
  | `low` | 710 | $0.014 | 10 s |

  All three gave the same verdict and the same reason on that pair. `google/gemini-3.1-pro-preview:batch`
  (half price) is refused by the chat endpoint, so it is not usable here.
* **Keeping a run to a budget.** `--budget-usd 2` stops starting new calls once $2 has been spent in that
  invocation (OpenRouter reports each call's cost; Google's own API does not, so the flag needs the
  OpenRouter backend). Calls already in flight still finish, so the total can overshoot by up to
  `--workers - 1` calls (a live check with a $0.02 cap and 2 workers ended at $0.046, from 4 calls that cost
  between $0.006 and $0.020 each: per-call cost varies about 3x even at `low`, so the per-call figures above
  are averages, not promises). Calls run one round at a time (every pair's first run, then every pair's
  second), so a budget stop leaves all pairs evenly covered. Everything finished is saved, and re-running
  the same command resumes. Each
  report prints the total cost. A run of about $2 at `low`: `eval --split dev --include annotated --runs 5
  --thinking-level low --budget-usd 2` (24 pairs x 5 runs = 120 calls, about $1.70 at $0.014).
  The thinking level must match what your platform uses, or the numbers describe a different setup:
  find out, or measure a few pairs at both levels before trusting a cheap run.

### Matching your platform

Accuracy numbers only transfer if the model and settings match. Ask whoever runs the platform:

1. The exact model ID (Pro or Flash-Lite, and whether it is a pinned version or a preview alias).
2. Thinking level, temperature and image resolution, if they set any.
3. Whether it sends the "Image A / Image B" text labels (`--labels on|off` here) and in which order.
4. How many calls it makes per drawing, and what it does with a `2`.

`python -I run.py models` lists what your key can use. To see how much the model matters, run the same
prompt under a second model with `--model <id>` and a new `--name`, then `compare` both runs: the table
shows each run's model and non-default settings. Each report also lists the model versions the API
returned, and warns if a run mixed more than one (preview models can change under the same name).
#   b e n z  
 