# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-28

Performance, statistical rigor, and robustness pass from a full audit.

### Changed

- `fit_temperature`: coarse grid + golden-section search replaces the
  551-point scan — 5k pairs go from ~11s to ~0.6s; numerically stable
  softplus inner loop. An exhaustive-grid equivalence test guards the result.
- `calibrate_report`: canonical sort + seeded shuffle before the split-half —
  results are now independent of input row order (labeled data is usually
  time/source-sorted; raw splitting confounded fit/test with drift).
- `best_threshold`: candidates are the observed signal values themselves
  (the coverage curve is a step function) instead of an integer-cent grid —
  exact at non-grid boundaries (e.g. 0.975) and O(n log n).
- `decide()` now requires `question_set_version` — no more silent
  "questions-untitled" audit records.
- Policy JSON carries a `schema` version (v1); newer schemas are rejected
  with an upgrade hint instead of failing cryptically.
- `make_backend("mock://…")` and non-JSON HTTP responses raise `BackendError`
  with actionable messages instead of raw `FileNotFoundError`/`ValueError`.

### Added

- Multinomial temperature scaling for choice/score questions:
  `fit_temperature_mc`, `apply_temperature_mc`, `mc_nll`,
  `mc_rows_from_examples`; `jevkit calibrate` reports the distribution-level
  temperature (softmax(log p / T)) alongside the binary view.
- Native kev `/v1/systemone/permute` covered by a live fake-server test.
- All examples honor `JEVKIT_BACKEND` / `JEVKIT_MODEL` via a shared factory.
- CI gained a lint job (ruff check + format check + mypy); `mypy` joined the
  `dev` extra. 97 tests total.

## [0.1.0] - 2026-09-24

Initial release.

### Added

- **core** — the three System One primitives (`Choice` / `Score` / `Noul`) with the
  kev-exact request contract (arbitrary-JSON states, instructions, and criteria
  descriptions; 1–255 candidates/levels); typed answer parsing for both native and
  SDK-style response shapes; `Backend` protocol with three implementations:
  `HttpBackend` (kev.serve over stdlib urllib, `x-typesafe-request-id` capture,
  native `/v1/systemone/permute`, 429/529 exponential backoff),
  `TypesafeSdkBackend` (official SDK, optional extra, injectable client for tests),
  `MockBackend` (deterministic, optional position-bias simulation).
- **policy** — `Tier` / `Gate` / `Policy` as plain data plus the pure `decide()`
  function; guard rails encoding documented failure modes (noul questions reject
  `CONFIDENCE` signals); decision records with the full probability distribution,
  model / question-set / policy versions, request id, and usage.
- **eval** — per-question-type calibration (ECE / Brier / log loss, split-half
  temperature fitting), coverage–accuracy curves and error-budget threshold
  selection, option-order permutation with the policy action-flip rate,
  threshold compilation into evidence-bearing policy locks, and drift checking.
- **data** — labeled JSONL in the `kev.train` format (one corpus feeds kev
  fine-tuning and jevkit evaluation).
- **CLI** — `jevkit demo / calibrate / coverage / permute / compile / check`.
- **compat** — contract tests against real kev source code (`KEV_SRC`), live-server
  tests (`JEVKIT_KEV_BASE_URL`), bilingual READMEs, MIT license, GitHub Actions CI.

[0.2.0]: https://github.com/JasmineAIGC/jevkit/releases/tag/v0.2.0
[0.1.0]: https://github.com/JasmineAIGC/jevkit/releases/tag/v0.1.0
