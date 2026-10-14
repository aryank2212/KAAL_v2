# KAAL V3 — Engineering Roadmap

> **North star:** make every output trustworthy before making the system more complex.
>
> V3 is the next software release of KAAL V2, not a claim that a new model checkpoint already exists. The existing RF-DETR-Small v4 detector remains the baseline until a candidate passes the evaluation gates below.

## Product contract

KAAL V3 is a perception-monitoring, robustness-evaluation, and swarm-simulation research system. It does not autonomously engage targets or control a weapon. Consequential decisions remain under human control.

The dashboard and API must never imply that simulated data is live, that a loaded model is necessarily accurate, or that an anomaly score proves an adversarial attack. An anomaly stream flags inputs for review; it does not repair predictions or guarantee robustness.

## Release gates

A milestone is done only when its tests pass and its acceptance evidence is recorded. A UI-only change is not enough for a backend contract; a benchmark number without its dataset and model provenance is not enough for a model promotion.

| Gate | Requirement |
|---|---|
| Truthfulness | Source mode, feed freshness, detector availability, and simulated values are explicitly identified. |
| Reliability | Reconnect, missing-model, invalid-config, stale-feed, and shutdown paths are covered by automated tests. |
| Evaluation | Every reported metric links to model hash, dataset version/split, evaluation command, and environment. |
| Robustness | Clean false-flag rate and attacked-input detection are reported together; aggregate mAP alone cannot promote a model. |
| Deployment | Safe network defaults, authenticated mutations, validated settings, and reproducible installation are documented. |
| Human oversight | No autonomous engagement/weapon control; no individual identification or tracking. |

## Phase 0 — Define scope and preserve a baseline

**Status: baseline established**

- [x] Keep RF-DETR-Small v4 as the production baseline; do not promote the rejected v5 experiment just because its in-distribution score improved.
- [x] Retain the existing dataset/model cards, decision records, and weight hashes.
- [x] State intended use and limitations in the README.
- [ ] Capture a clean baseline run for latency, memory, feed stability, and evaluation metrics on a named environment.
- [ ] Record known limitations in one release checklist: small-object performance, domain shift, camera-motion sensitivity, rough distance estimates, and fallback behavior.

**Exit:** a repeatable baseline command and environment record exist before any future model replacement.

## Phase 1 — Trustworthy telemetry and health contract

**Priority: P0 · Status: implementation in this branch**

- [x] Classify the configured source as `LIVE`, `REPLAY`, or `SIMULATION`.
- [x] Expose feed state (`active`, `waiting`, `disconnected`, `stale`, or `simulation`) and frame age in the API.
- [x] Distinguish a ready detector from a fallback/no-model state.
- [x] Stop sending a cached MJPEG frame once the source is no longer active.
- [x] Broadcast a periodic health heartbeat so clients can observe a source that stopped producing frames.
- [x] Remove randomly drifting coordinates and invented altitude, speed, and orbit values from the mission feed.
- [x] Show explicit warnings when frames are stale or inference is unavailable.
- [x] Add tests for mode resolution, feed-state classification, and the API contract.
- [ ] Run the full backend test suite and frontend lint/build in CI; fix all regressions before merge.
- [ ] Confirm behavior manually for live camera, video-file replay, missing weights, camera disconnect, and `KAAL_ENABLE_CAMERA=0`.

**Exit:** no missing camera/model path can be mistaken for a healthy live inference stream.

## Phase 2 — Honest measurements and calibration

**Priority: P0/P1**

- [ ] Replace uncalibrated distance estimates with explicit `unavailable` by default, or label them as rough estimates with their assumptions. The current fixed focal length and assumed object widths are not deployment-grade measurements.
- [ ] Introduce a versioned calibration profile per camera/lens/resolution; reject profiles that do not match the active source.
- [ ] Include measurement provenance: `value`, `unit`, `calibrated`, `calibration_id`, and a documented uncertainty range when supportable.
- [ ] Never synthesize GPS, altitude, speed, range, or object identity. Missing telemetry must stay missing.
- [ ] Add tests for absent, valid, mismatched, and invalid calibration files.

**Exit:** every displayed physical measurement is either tied to a valid calibration or clearly marked unavailable/approximate.

## Phase 3 — Reproducible robustness evaluation

**Priority: P1**

- [ ] Define one versioned evaluation manifest: model SHA-256, inference backend, dataset registry revision, split/manifest hash, class mapping, image preprocessing, configuration, and environment.
- [ ] Report clean mAP/recall, attacked-input performance, false-flag rate on clean data, flag rate on perturbed data, per-class results, and latency. Report confidence intervals where sample size supports them.
- [ ] Include baseline comparisons and failure examples, not just the best aggregate score.
- [ ] Add a regression suite covering clean images, patched inputs, low-frequency perturbations, image shape changes, tiny boxes, and unknown/empty labels.
- [ ] Prevent train/validation/test leakage and document near-duplicate removal. Keep independent test sets test-only.
- [ ] Add a promotion report that compares a candidate against v4 on both in-distribution and unseen-domain data.

**Exit:** a model cannot be promoted without a reproducible report, provenance, and a predeclared false-alarm budget.

## Phase 4 — Temporal defense under realistic camera changes

**Priority: P1**

- [ ] Benchmark the temporal cue on static cameras, slow panning, fast camera motion, lighting transitions, shadows, scene cuts, moving objects, and repeated frames.
- [ ] Measure false alarms and detection delay per scenario; do not generalize static-camera results to moving platforms.
- [ ] Add explicit cue states such as `warming_up`, `ready`, `rebaselining`, and `unsupported_motion`.
- [ ] Add safe re-baselining rules for scene changes and sensor interruptions, with tests for parked/slowly moving objects.
- [ ] Research ego-motion compensation only as an offline evaluation path until evidence demonstrates improved robustness without unacceptable false positives.

**Exit:** supported camera conditions are documented, and unsupported motion cannot silently produce overconfident status.

## Phase 5 — Backend security and operational reliability

**Priority: P1**

- [ ] Require authentication for all mutating controls when deployed beyond localhost; keep host binding local by default.
- [ ] Validate settings and source configuration at startup and update time; reject unknown enum values and non-finite numeric values.
- [ ] Ensure no secrets, full local paths, or sensitive exception traces are placed in API payloads.
- [ ] Test origin checks, reconnect/backoff, slow WebSocket clients, full queues, worker exceptions, shutdown, and repeated camera reconnects.
- [ ] Expose structured logs and a small metrics surface for frame age, dropped snapshots, detector failures, and processing latency.
- [ ] Add container/deployment health checks and document reverse-proxy configuration without weakening auth.

**Exit:** the service fails visibly and safely, with bounded memory and clear diagnostics, rather than silently presenting stale or fallback output.

## Phase 6 — Dashboard usability and accessibility

**Priority: P2**

- [ ] Keep source mode, source freshness, detector state, and anomaly state visually distinct.
- [ ] Provide a source/health summary that remains visible at narrow widths and in a reduced-motion mode.
- [ ] Add clear empty, waiting, reconnecting, stale, replay, simulation, and no-model states.
- [ ] Add a feed-age indicator and last-successful-frame time.
- [ ] Ensure colour is not the only signal; use text labels and accessible contrast.
- [ ] Remove cosmetic status indicators that claim network, satellite, or swarm capabilities that are not backed by connected systems.

**Exit:** a first-time operator can tell what is real, what is simulated, and whether inference is actually running without reading code.

## Phase 7 — Model/data governance and release packaging

**Priority: P2**

- [ ] Keep dataset source, license, label policy, deduplication and split manifest traceable.
- [ ] Make each release reproducible from pinned dependencies, model registry entry, and verified weight hashes.
- [ ] Keep the Hub model card aligned with the intended-use policy, evaluation evidence, caveats, and access conditions.
- [ ] Document clean-room/non-commercial data constraints before considering any broader licence or deployment.
- [ ] Produce a release note with breaking API changes and a migration guide for the dashboard snapshot schema.
- [ ] Tag only after tests pass and the model card / results / README agree.

**Exit:** a fresh clone can reproduce the supported software setup and verify the exact released model files.

## Phase 8 — Performance and release candidate

**Priority: P2 · only after correctness gates**

- [ ] Profile pipeline time by capture, preprocessing, inference, FFT, temporal scoring, encoding, and transport.
- [ ] Set measured latency/memory budgets for supported hardware; report percentiles, not only averages.
- [ ] Evaluate ONNX/TensorRT or other backend optimizations with parity checks against the baseline.
- [ ] Test long-running stability and reconnect recovery with a fixed-duration stress run.
- [ ] Create a release checklist, changelog, rollback instructions, and a tagged release candidate.

**Exit:** no optimization is accepted unless correctness/parity and robustness gates still pass.

## Execution order — next tasks

1. Complete Phase 1 by passing backend tests and frontend lint/build on this branch.
2. Verify every source state end-to-end: live camera, replay file, camera disconnected, missing weights, and simulation.
3. Add the regression tests or test fixtures needed for any failures found.
4. Merge Phase 1 only when checks pass; then begin Phase 2 calibration semantics.
5. Keep model retraining out of the critical path until the evaluation protocol is reproducible.

## Definition of done for KAAL V3

KAAL V3 is ready when the source and model status are truthful, stale frames cannot masquerade as current, measurements disclose their calibration, robustness claims are reproducible, and the service remains secure and observable under failure. A new model checkpoint is optional; evidence-backed trustworthiness is not.
