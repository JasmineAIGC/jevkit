# Examples

A progressive tutorial — each script runs offline on the deterministic mock
backend (no service, no API key, no tokens). Point them at a real backend
with environment variables:

```bash
python examples/01_ask_and_answer.py                       # mock (default)
JEVKIT_BACKEND=http://127.0.0.1:8009 JEVKIT_MODEL=kev-latest \
    python examples/01_ask_and_answer.py                  # your local kev.serve
JEVKIT_BACKEND=typesafe://jev-1.13.0 \
    python examples/01_ask_and_answer.py                  # hosted TypeSafe
```

| script | layer | teaches |
|---|---|---|
| [`01_ask_and_answer.py`](./01_ask_and_answer.py) | `jevkit.core` | the three primitives (Choice / Score / Noul), rich-JSON states and criteria, backend switching, reading typed answers |
| [`02_policy_and_ledger.py`](./02_policy_and_ledger.py) | `jevkit.policy` | three-tier gates, `decide()` as a pure function, the noul/confidence guard rail, writing and replaying decision logs |
| [`03_evaluation.py`](./03_evaluation.py) | `jevkit.eval` | per-type calibration + temperature, coverage–accuracy under an error budget, option-order permutation with policy action-flip rate |
| [`04_closed_loop.py`](./04_closed_loop.py) | both | compile thresholds from labeled data into an evidence-bearing lock, run under it, then detect drift on relabeled data |
| [`triage_router.py`](./triage_router.py) | all | a realistic service-shaped router (the "putting it together" version) |
| [`make_example_data.py`](./make_example_data.py) | — | generates `data/triage_labeled.jsonl` (kev.train format: request + inline labels + recorded answers) |

Start with:

```bash
python examples/make_example_data.py     # needed by 03 and 04
python examples/01_ask_and_answer.py
python examples/02_policy_and_ledger.py
python examples/03_evaluation.py
python examples/04_closed_loop.py
```

中文说明见根目录 [README.zh-CN.md](../README.zh-CN.md)。
