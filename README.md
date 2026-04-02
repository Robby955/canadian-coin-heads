# Canadian Coin Heads

Public technical showcase for the on-device identification pipeline behind the Canadian Coin Heads iOS app.

> [!WARNING]
> Current status: the shipping app currently has a known Core ML regression affecting on-device predictions. The benchmark numbers documented in this repository describe the offline evaluated pipeline and artifact set, not the current live app behavior until that bug is fixed.

<p align="center">
  <img src="assets/portfolio-stack.png" width="250" alt="Portfolio Stack View" />
  <img src="assets/portfolio-collectibles.png" width="250" alt="Collectibles View" />
  <img src="assets/portfolio-series.png" width="250" alt="Series Completion Tracking" />
</p>

## At a Glance

| Metric | Value | Notes |
|--------|-------|-------|
| Held-out Top-1 | 99.2% | CoinCLIP v4.2 on 8,112 held-out test images |
| Held-out Top-1 + OCR | 99.9% | Same benchmark with OCR reranking |
| Benchmarked on-device set | 1,103 coins | 25,427 bundled embeddings |
| Model size | 137 MB | MobileCLIP-S2 Core ML export |
| Runtime | iOS 17+ | Core ML + Vision + Accelerate |
| License | Proprietary | Public showcase materials only |

## Scope

This repository documents the on-device system only. It is not the full production app source code, and it does not publish the private production catalog or backend implementation.

The benchmark numbers here are engineering measurements for the documented on-device subset. They should not be read as a blanket guarantee for every possible real-world coin photo.

At the moment, they also should not be read as a statement of the current shipping app's on-device prediction quality because of the known Core ML regression above.

## Overview

Canadian Coin Heads is a production iOS app for Canadian coin collectors and precious-metals stackers. The production app uses a broader progressive pipeline, but this public repo stays focused on the local-first identification stage: CoinCLIP v4.2, a custom-trained MobileCLIP-S2 model running through Core ML with OCR-assisted reranking.

## Technical Highlights

- **Custom LoRA fine-tuning** (rank-16) on Apple's MobileCLIP-S2 architecture -- 274K trainable parameters on a 137M parameter backbone
- **Contrastive learning with design-family grouping** -- 1,103 coins organized into 59 design families + 154 unique solo classes (213 total)
- **25,591 real training photos** (deduplicated from 44K) with hard-negative mining and confusion pair extraction (1,004 mined pairs)
- **OCR post-processing** with fuzzy year/denomination matching via Apple Vision framework (+0.7% accuracy boost, eliminates confident misidentifications)
- **Float32 Core ML deployment** tuned for iPhone execution, with a bundled benchmark harness for real-device latency validation
- **Three-phase progressive pipeline**: on-device CLIP, cloud CLIP (pgvector), Claude Vision hybrid -- each phase fires only if the previous one is uncertain
- **25,427 pre-computed embeddings across 1,103 unique coins** with vDSP-accelerated cosine similarity search
- **Production test coverage**: 49 iOS test files and 241 backend tests in the broader app codebase

## Results

CoinCLIP v4.2 reaches **99.2% Top-1 accuracy** on a held-out benchmark of 8,112 images, rising to **99.9% with OCR reranking** on the same split.

Those numbers describe the benchmark pipeline, not the currently affected shipping Core ML path.

| Mode | Top-1 | Top-5 | Evaluation |
|------|-------|-------|------------|
| Pure CLIP | 99.2% | ~99.7% | 8,112-image holdout benchmark |
| CLIP + OCR | 99.9% | ~99.9% | Same holdout benchmark |

Small external spot-check sets also improved materially with OCR, but the main benchmark above is the number this repo is centered on. See [RESULTS.md](RESULTS.md) for methodology, sample sizes, limitations, and version history.

## Architecture

The production app uses a progressive three-phase pipeline:

1. **Phase 0 -- On-device:** CoinCLIP v4.2 produces a 512-dimensional embedding and ranks against the bundled embedding set.
2. **Phase 1 -- Cloud CLIP:** a separate embedding space provides a second opinion when confidence is weak.
3. **Phase 2 -- Hybrid analysis:** structured reasoning is used only for the hardest cases.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline diagram and system notes.

## Training Pipeline

The model is trained using contrastive learning on design families -- groups of visually identical coins that differ only in mint year or minor variants. This lets the model learn what makes a Silver Maple Leaf look different from a Caribou quarter, rather than memorizing year text.

Key innovations:
- Design-family grouping (59 families from 1,103 coins, with 154 unique solo classes)
- Hard-negative mining with confusion pair feedback loops (1,004 auto-mined pairs)
- OCR-based post-processing for disambiguation
- Aggressive label cleanup (1,201 fixes between v3.2 and v4.2)

See [APPROACH.md](APPROACH.md) for the full training methodology and architecture decisions.

## Repo Contents

- [APPROACH.md](APPROACH.md): training strategy, data curation, and design-family setup
- [ARCHITECTURE.md](ARCHITECTURE.md): runtime pipeline and fallback system design
- [RESULTS.md](RESULTS.md): benchmark methodology, historical results, and limitations

## Stack

| Layer | Technologies |
|-------|-------------|
| iOS App | Swift, SwiftUI, SwiftData, Core ML, vDSP/Accelerate, Vision (OCR) |
| ML Model | MobileCLIP-S2, LoRA, OpenCLIP, PyTorch |
| Backend | Python, FastAPI, PostgreSQL + pgvector, Google Cloud Run |
| Training | PyTorch, A100 GPU, custom contrastive loss |
| Data Pipeline | Python, Playwright (scraping), deduplication pipeline |

## Links

- **App Store**: [Canadian Coin Heads](https://apps.apple.com/app/canadian-coin-heads/id6740244078)
- **Website**: [canadiancoinheads.com](https://canadiancoinheads.com)
