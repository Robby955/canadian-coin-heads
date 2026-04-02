# Benchmark Status

> [!WARNING]
> The shipping app currently has a known Core ML regression affecting on-device predictions.

This public repository does not currently include a reproducible benchmark bundle for the on-device model.

## Current Public Position

- This repo documents the architecture, training approach, and known limitations of the on-device pipeline.
- It does not publish the full machine-readable holdout set, scoring outputs, artifact checksums, and reproduction steps needed to support a serious public benchmark claim.
- For that reason, exact historical internal percentages are intentionally not presented here as a front-facing public result.

## What Would Be Required For A Professional Public Benchmark

1. A published holdout manifest with stable identifiers for every evaluation image.
2. Published checksums for the exact model artifact and embedding bundle used in evaluation.
3. A committed scoring script that reproduces the reported metrics from those exact artifacts.
4. A machine-readable results file with per-image outputs, not just a prose summary.
5. Clear separation between internal offline benchmarks and current shipping app behavior.

## Why This Matters

Without the pieces above, exact percentages read as a claim rather than evidence. That is not the standard this repo should present publicly.

## Current Guidance

- Read this repo as a technical showcase of the intended system design.
- Do not read it as proof of current shipping accuracy.
- Do not read it as an independently reproducible ML benchmark package.
