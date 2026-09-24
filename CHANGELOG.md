# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/JasmineAIGC/jevkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JasmineAIGC/jevkit/releases/tag/v0.1.0
