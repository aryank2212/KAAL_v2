# KAAL V3 — OpenCode Task Prompts

This is the execution queue for OpenCode in the local aryank2212/KAAL_v2 checkout. Run one numbered task at a time. The goal is fast, evidence-backed engineering progress, not a speculative rewrite.

## Rules for every task

- Start by reading this file, README.md, pyproject.toml, the relevant implementation, and its tests. Follow repository conventions.
- Before editing, run git status --short --branch. Preserve user changes; never reset, clean, stash, overwrite, or switch away from a dirty worktree without permission.
- Work on a dedicated feature branch. Never commit or push directly to main, and never merge a pull request.
- Keep downloaded data, model checkpoints, generated reports, caches, and credentials out of Git. Never print tokens or private environment values.
- Never invent metrics. Fixture values are not fresh evaluations. Unavailable or unmeasured metrics must be null, not zero.
- Do not change production weights, the selected production detector, or the existing nine-class taxonomy.
- Verify each dataset's upstream provenance, license and access terms before downloading it. Do not bypass access controls. Use the repository's ignored data directory; never commit raw data.
- The new dataset experiment is limited to general-purpose civilian aerial traffic and scene-understanding research. Do not use it to expand military-target detection, target prioritization, tracking of people, or autonomous engagement. Keep consequential decisions human-reviewed.
- Do not launch paid cloud compute, subscribe to resources, or create scheduled jobs without explicit user approval of estimated cost.
- After each task, report changed files, exact commands run, real test/CI results, blockers, and the next recommended task. Stop if required access or inputs are missing.

## Known context

- Repository: aryank2212/KAAL_v2.
- The production detector and existing nine-class dataset pipeline must remain unchanged.
- PR #6 adds versioned experiment manifests, hashes, deterministic JSON/Markdown reports, and the kaal-bench CLI. At last check it was open and CI had passed Ruff, mypy, dashboard lint/build, and 223 CPU tests with 3 skipped on Python 3.11 and 3.12. Re-check the current state before building on it; do not duplicate its work.
- The preferred first external dataset is AU-AIR, a civilian low-altitude traffic dataset. Its upstream repository describes 32,283 annotated frames and eight traffic categories. Images and annotations are separate downloads, so verify the actual license terms, archive contents, and schema before acquisition.
- UAVDT and UAVid are possible later candidates but need their own license/access review. Do not download everything at once.

---

## Task 0 — Inspect before editing

**Prompt**

Read tasks_prompts.md and inspect the repository state. Run git status --short --branch. Inspect README, pyproject.toml, current dataset builders and validators, tests, CI, and PR #6 if GitHub CLI/authentication is available. Do not modify files, download data, train models, or commit. Report the branch/worktree state, existing benchmark tools, relevant commands, blockers, and safest next action.

**Acceptance criteria:** no changes or downloads; user changes preserved; facts clearly separated from assumptions.

## Task 1 — Reconcile KAAL Bench

**Prompt**

Inspect the KAAL Bench implementation and PR #6 before editing. If the branch is not local, fetch it without overwriting local changes. Check whether the PR is open, merged, or superseded. Run focused tests and inspect schema validation, CLI behavior, hashing, and deterministic output. Do not merge or rewrite existing detector evaluators. Make a small follow-up PR only if a concrete defect is found; otherwise report no code change is needed.

**Acceptance criteria:** no duplicate schema/CLI; focused tests, Ruff and mypy run where available; queued CI is never described as passed.

## Task 2 — AU-AIR acquisition preflight (no download)

**Prompt**

Prepare an acquisition preflight for AU-AIR as a separate civilian-only experiment. Read the upstream README and official dataset page. Record authoritative download locations, archive names/sizes if published, license/usage terms, expected image and annotation layout, schema, category mapping, and redistribution limits. Check that the destination is Git-ignored. Add a concise source/provenance note to Docs/CIVILIAN_AERIAL_DATA.md if it fits repository conventions. Do not download files yet. Do not modify MODEL/datasets_registry.yaml or the production taxonomy.

**Acceptance criteria:** statements about license and source cite official documentation; unknowns stay explicitly unresolved; no archive downloaded and no raw data committed.

## Task 3 — Isolated AU-AIR importer

**Prompt**

Implement a small, tested importer for AU-AIR civilian traffic data, separate from MODEL/build_dataset.py and the existing nine-class taxonomy. Inspect the real annotation schema from upstream documentation or a user-supplied sample before coding; do not guess field meanings. Convert labels into a separate YOLO-format dataset with a documented eight-class civilian mapping. Create deterministic train/validation/test splits grouped by original video or sequence, so adjacent frames cannot leak across splits. If reliable sequence identity cannot be recovered, fail closed and document the blocker rather than randomly splitting frames. Produce a JSON manifest with source/license, seed, per-split and per-class counts, invalid/dropped annotations, and hashes of the inputs. Keep raw data in the ignored data directory. Do not touch production data or weights. Add small CPU-only fixture tests for category mapping, box validation, deterministic grouping, missing files and manifest generation.

**Acceptance criteria:** works with a small fixture and later with the verified real schema; unknown labels and malformed boxes are reported, not silently remapped; split groups are disjoint; CI requires no full dataset; no data/checkpoints are committed.

## Task 4 — Dataset quality and leakage audit

**Prompt**

Add a read-only audit command for the isolated AU-AIR dataset. Report readable/corrupt images, missing annotations, invalid/out-of-bounds boxes, class balance, object-size distribution, exact duplicates, likely near-duplicates and split/group leakage. Produce deterministic JSON and a concise Markdown summary with source and input hashes. Do not alter labels automatically. Test with small CPU fixtures, including intentionally broken examples. If near-duplicate analysis has runtime/threshold limits, make them explicit.

**Acceptance criteria:** read-only and repeatable; label issues are flagged for human review; tests include clean and broken cases.

## Task 5 — Civilian-only baseline training (only after Tasks 2–4)

**Prompt**

Only proceed when real AU-AIR data is present, its terms are verified, the importer and audit pass, and the user has requested the run. Train a separate civilian-only baseline using the repository's supported training interface. Use a new run name/output directory; do not initialize from, overwrite, publish, or change KAAL's production checkpoint. Begin with a short smoke test for data loading, class names, GPU memory, checkpoint creation and inference. Before a full run, estimate duration and storage from measured throughput and ask the user if cost/time is material or uncertain. No paid cloud job without explicit cost approval. Record dataset-manifest hash, model/base-weight hash, Git SHA, library versions, seed, resolution, batch/accumulation, epoch and early-stopping settings. Do not claim accuracy until evaluation actually runs.

**Acceptance criteria:** smoke test passes; the model class mapping exactly matches the civilian dataset; outputs are isolated and ignored by Git; significant compute is explicitly approved.

## Task 6 — Evaluate baseline and benign robustness

**Prompt**

Evaluate the civilian-only model on the held-out sequence-level test split. Report mAP@50, mAP@50:95, precision/recall, per-class results, object-size buckets when supportable, latency p50/p95, throughput and peak VRAM. Separately evaluate documented non-adversarial image-quality changes such as compression, blur, brightness and low-light conditions, including clean false-positive rates. Clearly label these as controlled transformations, not real-world guarantees. Do not use the test set for model selection. Save results using KAAL Bench where supported. Keep claims limited to actual measurements.

**Acceptance criteria:** test split is not used for tuning; clean and transformed conditions are separated; reports identify exact code/data/model/configuration; no automatic production promotion.

## Task 7 — Final handoff

**Prompt**

Write a concise research handoff with exact commands for acquisition, validation, conversion, auditing, training and evaluation; changed files; test/CI outputs; dataset license/source status; measured results; compute/storage actually used; limitations; and the next evidence-backed experiment. Separate completed work from proposed work. Do not merge a PR, publish weights, or claim success without evidence.

---

## Local OpenCode operating loop

From the repository root, start OpenCode and give it the exact prompt for one task. Begin with Task 0 and review its report before proceeding.

1. Execute one numbered task.
2. Inspect the diff and tests/output.
3. Accept the result only when its acceptance criteria pass.
4. Commit on the feature branch; never merge to main automatically.
5. Proceed only after resolving prior-task blockers.

**Definition of progress:** reproducible provenance, validated annotations, leakage-free splits, passing tests, and measured outcomes tied to exact artifacts—not downloaded gigabytes or training epochs.
