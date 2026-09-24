# Contributing to jevkit

Thanks for considering a contribution!

## Setup

```bash
git clone https://github.com/JasmineAIGC/jevkit && cd jevkit
python3 -m venv .venv && . .venv/bin/activate   # Python >= 3.10
pip install -e '.[dev]'
pytest
```

The core package has **zero dependencies** — please keep it that way. Anything
that needs a third-party library belongs in an optional extra (`[typesafe]`)
or behind a lazy import with a helpful error message.

## Before you open a PR

1. **Tests must pass**: `pytest` (mock backends only, no network needed).
2. **New behavior needs a test.** Policy and metric code is deliberately pure
   and deterministic — numeric assertions are expected, not bonus.
3. **Keep the contract honest.** Anything touching the request/response shape
   should be checked against kev's actual source (`kev/api.py`) via
   `tests/test_kev_compat.py` (`KEV_SRC=/path/to/kev`).
4. **Docs are bilingual.** README.md is English (primary); README.zh-CN.md is
   the Chinese counterpart. Update both when you change user-facing behavior.
5. Code comments and docstrings may be Chinese or English; match the file you
   are editing.

## Design boundaries (read before proposing big changes)

- Policy is **data + a pure function** — resist DSL-izing it.
- All evaluation is **per question type** — no global thresholds.
- Decision logs always keep the **full probability distribution**.
- Thresholds bind to the data and model they were compiled on; anything that
  silently encourages cross-model threshold reuse is a bug.

## Reporting issues

Include: jevkit version, backend spec (`mock` / `http://…` / `typesafe://…`),
model name, a minimal labeled-data sample, and the CLI command or code snippet.
