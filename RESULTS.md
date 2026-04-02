# Benchmark Notes

> [!WARNING]
> The shipping app currently has a known Core ML regression affecting on-device predictions.

This repo does not currently include a reproducible public benchmark bundle for the on-device model.

## What This Means

- The repo is useful for understanding the model and system design.
- It is not yet a publishable benchmark package in the academic or reproducible-ML sense.
- That is why exact historical percentages are not being presented as the headline public story here.

## What A Serious Public Benchmark Release Would Need

1. A published holdout manifest with stable identifiers for every evaluation image.
2. Published checksums for the exact model artifact and embedding bundle used in evaluation.
3. A committed scoring script that reproduces the reported metrics from those exact artifacts.
4. A machine-readable results file with per-image outputs, not just a prose summary.
5. Clear separation between internal offline benchmarks and current shipping app behavior.

## Why This Matters

Without those pieces, exact percentages read as a claim rather than evidence. That is not the standard this repo should present publicly.

## How To Read This Repo

- Read this repo as a technical showcase of the intended system design.
- Do not read it as proof of current shipping accuracy.
- Do not read it as an independently reproducible ML benchmark package.
