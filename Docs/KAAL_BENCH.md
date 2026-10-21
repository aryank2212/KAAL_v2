# KAAL Bench — manifest and report contract

KAAL Bench v1 records experiment provenance and formats existing results. It
does not run detectors, train models, alter production weights, or replace the
existing evaluators.

## Commands

Capture metadata:

~~~bash
python -m MODEL.bench capture --run-id kaal-v4-baseline-001 --model rfdetr-small-v4 --dataset kaal_v4 --split test --seed 42 --weights path/to/checkpoint.pth --dataset-manifest path/to/dataset-manifest.json --output runs/bench/kaal-v4-baseline-001.json
~~~

Missing output directories are created, so the `runs/bench/` path above works in
a fresh checkout.

Record measured metrics and result files in the same manifest:

~~~bash
python -m MODEL.bench capture --run-id kaal-v4-baseline-001 --model rfdetr-small-v4 --dataset kaal_v4 --split test --seed 42 \
    --metric map50=0.405 --metric map50_95=0.31 --metric latency_p95_ms=null \
    --artifact runs/bench/predictions.json \
    --output runs/bench/kaal-v4-baseline-001.json
~~~

`--metric` takes `NAME=VALUE` and may be repeated; the value is a finite number
or `null` for a quantity that was not measured. A name may not be given twice.
`--artifact` takes a file path and may be repeated; each one is recorded with
its SHA-256 and kind.

Validate and render a deterministic Markdown report:

~~~bash
python -m MODEL.bench validate runs/bench/kaal-v4-baseline-001.json
python -m MODEL.bench report runs/bench/kaal-v4-baseline-001.json --output runs/bench/kaal-v4-baseline-001.md
~~~

Weights and dataset-manifest arguments are optional. Missing hashes are null,
meaning unavailable or not measured, never zero. The command captures Git SHA
when available, plus Python and platform information; it does not claim an
evaluation ran. Unknown schema fields are rejected. Metrics are finite numbers
or null. Artifact entries include path, SHA-256, and kind. JSON keys and report
metrics/artifacts use stable sorting.

Invalid usage — an unreadable input, a malformed manifest, an out-of-range
metric — exits with status 2 and a one-line `bench: error:` message rather than
a traceback.

## Boundaries

- Existing evaluators are not replaced; adapters to their outputs are a later step.
- Keep the v4 production baseline and independent test-only datasets unchanged.
- Do not put credentials or sensitive machine paths in manifests.
- A manifest records provenance, not proof of model quality.
