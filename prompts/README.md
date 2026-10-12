# Prompts: which file is authoritative

**The authoritative policy is [`src/chemverify/policy.py`](../src/chemverify/policy.py).** It holds the evaluation rules,
the chemical and notation category tables, the quorum constants and both prompt generators.

| File | Status | Used by |
|---|---|---|
| `chem_equiv_policy_v1.txt` | **Generated** from `policy.system_prompt()`. Policy version `chem-equiv-1`. | Google pipeline: the policy-prompt calls |
| `chem_equiv_confirm_v1.txt` | **Generated** from `policy.confirmation_prompt()`. | Google pipeline: the confirming call for an unconfirmed reject |
| `archive/qc_equivalence_v1.txt` | **Archived draft, unchanged.** Nothing reads it. It was the starting point of the policy. | nothing |
| `baseline_v0.txt`, `baseline_v0_conf.txt`, `baseline_auditor_verbatim.txt` | Harness prompts for `run.py eval` on the 17-molecule set. **Not part of the Google pipeline.** | harness |
| `perception_probe.txt` | Harness prompt for `run.py probe`. **Not part of the Google pipeline.** | harness |

Do not edit the two generated files by hand. Change `policy.py`, then regenerate and check them:

```
python -I pipeline_google10.py policy --write
python -I pipeline_google10.py policy
```

`tests/test_policy_decision.py` fails if either file differs from what `policy.py` renders, and it pins the hash of
`chem_equiv_policy_v1.txt`: changing that prompt changes what Gemini sees and breaks the link to every recorded run, so it
must come with a new `POLICY_VERSION`.

Three versions, kept apart on purpose:

- `POLICY_VERSION` (`chem-equiv-1`): the text of the policy prompt.
- `DECISION_VERSION` (`chem-decision-2`): the rules in `src/chemverify/decision.py`. It is not part of any prompt.
- `SCHEMA_VERSION` (`2`): the layout of the results CSV.

Every results CSV row and every run folder records the versions that produced it.
