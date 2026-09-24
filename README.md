# jevkit

**A unified policy + evaluation framework for [Jev](https://typesafe.ai) / [Kev](https://github.com/jaredpalmer/kev) (System One) decision models.**
Unstructured state in, typed probabilistic decisions out — **the model handles uncertainty, code handles policy**. jevkit is that code.

[![CI](https://github.com/zbloom/jevkit/actions/workflows/ci.yml/badge.svg)](./.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](./pyproject.toml)

[中文文档](./README.zh-CN.md)

Jev (by TypeSafe) turns "ask an LLM" into a single endpoint that returns typed probabilities (`POST /v1/systemone`, three primitives: Choice / Score / Noul; free output tokens, 70–500 ms). [Kev](https://github.com/jaredpalmer/kev) is the Apache-2.0 open-source counterpart with a fully TypeSafe-compatible API. But both only give you **probabilities — never actions**. Translating probabilities into actions (the policy layer), validating that the probabilities themselves can be trusted (the evaluation layer), and closing the loop between the two: that is jevkit.

```
Client      POST /v1/systemone, get probabilities        commodity: typesafe-sdk, kev.serve
Policy      probability → action (data + pure function)  jevkit.policy / jevkit.record
Evaluation  can you trust the probability? where's the   jevkit.metrics / calibration /
            threshold?                                    permute / compile
```

The closed loop:

```
labeled data (kev train format) ──compile──▶ policy.lock.json (thresholds + evidence)
                                                 │ loaded at runtime
                                                 ▼
                                           decide() → decisions.jsonl
                                                 │ labels flow back
                                                 ▼
                                          check drift ──out of bounds──▶ recompile
```

## Why jevkit

Ecosystem check (2026-09): the client layer is saturated (official `typesafe-sdk` plus community wrappers in Java / Elixir / Rust). The policy/evaluation layer has only fragments — `daf-jev` is a research toolbox with its own httpx client, `jevcal` is a 10-star experiment, `jev-starter` is TypeScript. **A pip-installable Python policy/eval framework that sits on the official SDK and speaks kev natively did not exist.** Every design decision maps to a documented failure mode:

- **Policy = data + a pure function, not a DSL.** Thresholds come from *your* false-positive/false-negative costs; the same 0.9 probability means different things in ticket routing and account banning. The framework provides structure, evidence, and audit — not opinions.
- **Everything is per question type.** Calibration error differs by an order of magnitude across types (Noul 0.012 / Choice 0.086 / Score 0.254); one global threshold cannot cover them.
- **Logs keep the full distribution.** A production log must answer: "was it 0.91 vs 0.05, or 0.36 vs 0.34?" Model version, question-set version, policy snapshot, and the action taken are all recorded.
- **Zero required dependencies.** Pure stdlib core (Python ≥ 3.10); the official SDK is an optional extra; a local kev server is reached with the built-in HTTP backend.
- **kev-native.** The labeled-data format *is* the `kev.train` format (one corpus feeds both fine-tuning and evaluation), and the kev-only `/v1/systemone/permute` endpoint is used for server-side permutation tests when available.

## Install

```bash
pip install jevkit                    # core (zero dependencies)
pip install 'jevkit[typesafe]'        # + official typesafe-sdk adapter
pip install 'jevkit[dev]'             # + pytest
```

## Quickstart

The fastest way in is the tutorial in [`examples/`](./examples/README.md) — four progressive scripts that run offline on the deterministic mock backend. The CLI tour:

```bash
jevkit demo                                    # ① three-tier routing on a mock backend
jevkit calibrate --data labeled.jsonl          # ② per-type calibration + temperature fit
jevkit coverage  --data labeled.jsonl          # ③ coverage–accuracy curves
jevkit compile  --data labeled.jsonl \
      --template policy-template.json \
      --budget 0.05                            # ④ compile thresholds under an error budget → lock
jevkit check    --data new-week.jsonl \
      --lock policy.lock.json                  # ⑤ drift check (exit code 1 on drift)
```

Library API (full example in [`examples/triage_router.py`](./examples/triage_router.py)):

```python
from jevkit import Choice, Gate, Noul, Policy, Score, Signal, Tier, decide, make_backend

questions = {
    "department": Choice("Which team should handle this ticket?", {
        "returns": "Exchanges, refunds, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems",
    }),
    "escalate": Noul("Does this need urgent human attention?"),
}

policy = Policy(version="triage-v1", gates=(
    Gate("department", Signal.CONFIDENCE, (
        Tier("AUTO", 0.70), Tier("DEFER", 0.50), Tier("HUMAN", 0.0))),
    Gate("escalate", Signal.PROBABILITY, (
        Tier("ALERT", 0.90), Tier("DEFER", 0.60), Tier("NORMAL", 0.0))),
))

backend = make_backend("http://127.0.0.1:8009")          # local kev.serve
answers = backend.ask(state, questions, model="kev-latest")
record = decide(answers, policy, state_ref=ticket_id, state=state,
                question_set_version="triage-questions-v1", question_set=questions)
if record.actions["department"]["action"] == "AUTO":
    dispatch(ticket_id)
```

## Backends

| spec | implementation | use |
|---|---|---|
| `http://127.0.0.1:8009` | `HttpBackend` (stdlib urllib) | **kev.serve** or any TypeSafe-compatible endpoint; reads the `x-typesafe-request-id` header; uses kev's native `/v1/systemone/permute` |
| `typesafe://jev-1.13.0` | `TypesafeSdkBackend` (official SDK, lazy import) | hosted TypeSafe; inherits SDK auth (`TYPESAFE_API_KEY`) and 429/529 backoff |
| `mock` | `MockBackend` | tests and demos; deterministic (same state → same probabilities), `bias=` simulates position bias |

Running the open-source model (kev):

```bash
git clone https://github.com/jaredpalmer/kev && cd kev
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

**Note:** kev's calibration trails hosted Jev (automatable fraction under a 5% error budget: 0.45–0.57 vs 0.70), so **thresholds do not transfer across models** — recompile on new data when you switch.

## Data format (shared with kev.train)

One full request per line plus an inline `label` on every question — exactly what `kev.data.load_records` expects, so **the same corpus feeds `kev.train` fine-tuning and jevkit evaluation**. An optional `answers` field records predictions for offline (free) evaluation:

```jsonc
{"state": "Order #4411 was charged twice...",
 "questions": {
   "department": {"type": "choice", "instructions": "Which team?",
                  "criteria": {"returns": "…", "shipping": "…", "billing": "…"},
                  "label": "billing"},
   "escalate": {"type": "noul", "instructions": "Urgent?", "label": false}},
 "answers": {"department": {"type": "choice", "choice": "billing",
                            "confidence": 0.88, "probabilities": {"…": "…"}}}}
```

Label semantics match kev: choice → option key; noul → true/false; score → zero-based level index. Rows without embedded answers can be evaluated live with `--backend`.

## CLI reference

| command | what it does |
|---|---|
| `jevkit demo` | three-tier routing on a mock backend; writes `demo-decisions.jsonl` |
| `jevkit calibrate --data F` | per-type ECE / Brier / LogLoss + temperature fit (split-half, no in-sample self-deception) |
| `jevkit coverage --data F --budget 0.05` | coverage–accuracy table; automatable fraction and optimal threshold under an error budget |
| `jevkit permute --data F --policy P --n-perm 6` | option-order stability: drift / KL / **policy action-flip rate** (against your thresholds, not just argmax) |
| `jevkit compile --data F --template T --budget b` | per-gate auto-tier threshold ← `argmax coverage s.t. err ≤ b`, emits a lock file with evidence |
| `jevkit check --data F --lock L` | new data vs lock evidence: accuracy / coverage / ECE drift; exit code 1 on drift |

## Package layout

The package structure mirrors the architecture — the dependency direction is strictly `core ← policy ← eval`:

```
jevkit/
├── errors.py        exception hierarchy (shared across layers)
├── core/            the client abstraction over POST /v1/systemone
│   ├── types.py       Choice / Score / Noul + typed answers (kev-exact contract)
│   └── backend.py     Backend protocol + HttpBackend (kev.serve, native permute)
│                      + TypesafeSdkBackend (official SDK, optional) + MockBackend
├── policy/          probability → action
│   ├── policy.py      Tier / Gate / Policy as data + decide() pure function
│   └── record.py      DecisionRecord + JsonlLedger (full-distribution audit log)
├── eval/            can you trust the probability? where's the threshold?
│   ├── data.py        kev.train-format labeled data (shared corpus for both)
│   ├── metrics.py     ECE / Brier / logloss / coverage / budget thresholds
│   ├── calibration.py per-type temperature scaling (split-half discipline)
│   ├── permute.py     option-order stability incl. policy action-flip rate
│   ├── compile.py     thresholds → evidence-bearing policy.lock + drift check
│   └── synthetic.py   overconfident synthetic data for tests and demos
└── cli.py           jevkit demo / calibrate / coverage / permute / compile / check
```

The public API is stable at the package root: `from jevkit import Choice, Policy, decide, make_backend, …` — subpackages are importable directly (`jevkit.core`, `jevkit.policy`, `jevkit.eval`) for users who prefer explicit layering.

## Key numbers worth memorizing

| quantity | value |
|---|---|
| price | $0.042/MTok input, output free |
| latency | 70–500 ms (frontier LLMs on the same task: 3–329 s) |
| calibration | conf ≥ 0.9 → ~92% accuracy at ~73% coverage (PrimeLine preregistered test) |
| per-type calibration error | Noul 0.012 / Choice 0.086 / Score 0.254 |
| automatable @ 5% error budget | Jev 0.70; Kev-9B 0.45–0.57 |
| agreement | Kev-9B base 66.4% → Nimble 90.1% → Jev 93.2% (324 held-out) |
| context | 64k total; state + longest question ≤ 32k (kev serving: 8k) |
| versions | pin `jev-1.13.0` in production; `-latest` drift silently crosses thresholds |

## Compatibility & verification

Contract compatibility is verified against **kev's actual source code** (`tests/test_kev_compat.py`, gated on `KEV_SRC`):

- jevkit question serialization is accepted by kev's `SystemOneRequest` (pydantic) and `to_record` encoder, including arbitrary-JSON `instructions`/criteria descriptions and boundary option counts (255 options, single-level scores)
- kev's real answer serializer (`to_answers`) output parses cleanly in jevkit
- jevkit-generated labeled data loads via `kev.data.load_records` and is directly feedable to `kev.train`

Live-server checks run when `JEVKIT_KEV_BASE_URL` points at a running `kev.serve`. See [DEVELOPMENT](#development) below.

## Development

```bash
git clone https://github.com/zbloom/jevkit && cd jevkit
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest                                        # full suite, mock backends only

# kev compatibility (contract tests against real kev code):
git clone --depth 1 https://github.com/jaredpalmer/kev /tmp/kev
pip install pydantic datasets
KEV_SRC=/tmp/kev pytest tests/test_kev_compat.py

# live server E2E (optional):
KEV_SRC=/tmp/kev JEVKIT_KEV_BASE_URL=http://127.0.0.1:8009 \
    pytest tests/test_kev_compat.py -k live

python examples/make_example_data.py          # regenerate examples/data/
```

Contributions welcome — see [CONTRIBUTING.md](./CONTRIBUTING.md).

## Scope (not in v0.1)

Async clients, web dashboards, fine-tuning wrappers (kev already ships `kev.train`), LangChain/PydanticAI integrations, advanced ordinal metrics for score questions, multi-policy shadow mode.

## License

[MIT](./LICENSE)
