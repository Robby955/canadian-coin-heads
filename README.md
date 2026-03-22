# Canadian Coin Heads

**On-device AI identification of Canadian coins with 97.2% accuracy.**

![Accuracy](https://img.shields.io/badge/Top--1_Accuracy-97.2%25-brightgreen)
![Model Size](https://img.shields.io/badge/Model_Size-68.8_MB-blue)
![Coins](https://img.shields.io/badge/Coins-5%2C791-orange)
![Tests](https://img.shields.io/badge/Tests-1%2C134%2B-blue)
![Platform](https://img.shields.io/badge/Platform-iOS_17%2B-black)
![License](https://img.shields.io/badge/License-Proprietary-lightgrey)

## Overview

Canadian Coin Heads is a production iOS app, live on the App Store, that identifies 5,791+ Royal Canadian Mint coins from a single photo. The app covers every Canadian coin denomination from 1858 to 2026 -- circulation, bullion, and collectible series.

At its core is CoinCLIP v3.2, a custom-trained MobileCLIP-S2 model running entirely on-device via Core ML. No server round-trip is needed for 97% of identifications. The model achieves 97.2% Top-1 accuracy on a held-out benchmark of 8,111 test images across 1,103 unique coins.

## Technical Highlights

- **Custom LoRA fine-tuning** (rank-16) on Apple's MobileCLIP-S2 architecture -- 0.4M trainable parameters on a 137M parameter backbone
- **Contrastive learning with design-family grouping** -- 962 coins collapsed into 149 design families so visually identical year variants share representations
- **42,674 real training photos** with hard-negative mining and confusion pair extraction (321 mined pairs)
- **HoughCircles preprocessing** for coin isolation from cluttered backgrounds (eBay listings, tabletop photos)
- **OCR post-processing** with fuzzy year/denomination matching via Apple Vision framework (+0.2% accuracy boost, fewer confident misidentifications)
- **On-device Core ML inference under 300ms** on iPhone 12+ with Float16 quantization
- **Three-phase progressive pipeline**: on-device CLIP, cloud CLIP (pgvector), Claude Vision hybrid -- each phase fires only if the previous one is uncertain
- **44,328 pre-computed embeddings** with vDSP-accelerated cosine similarity search (full 44K search in <50ms)
- **Full test coverage**: 893+ iOS unit tests across 49 test files, 241 backend tests

## Architecture

The identification system uses a progressive three-phase pipeline. Each phase is more powerful but slower, and only fires if the previous phase returned low confidence.

```
    Photo Input
        |
        v
+---------------------------------------+
|  PHASE 0: On-Device  (<300ms)         |
|                                        |
|  CoinCLIP v3.2 (Core ML)              |
|  MobileCLIP-S2 + LoRA                 |
|       |                                |
|       v                                |
|  Encode photo -> 512-dim embedding     |
|       |                                |
|       v                                |
|  vDSP cosine similarity               |
|  vs 44,328 pre-computed embeddings     |
|       |                                |
|       v                                |
|  Top matches + confidence score        |
|       |                                |
|       v                                |
|  OCR refinement (year, denomination)   |
+---------------------------------------+
        |
        | confidence < 0.85?
        v
+---------------------------------------+
|  PHASE 1: Cloud CLIP  (2-5s)         |
|                                        |
|  ViT-B-32 (LAION2B) + pgvector        |
|  1,904 server-side embeddings          |
|  Separate embedding space              |
+---------------------------------------+
        |
        | confidence < 0.85?
        v
+---------------------------------------+
|  PHASE 2: Hybrid Analysis  (15-45s)  |
|                                        |
|  Claude Vision reasoning               |
|  + structured database scoring         |
|  (metal, year, design elements)        |
+---------------------------------------+
        |
        v
    Identification Result
    + confidence badge
    + alternative matches
```

97% of identifications resolve in Phase 0 -- entirely on-device, no network required.

## Results

CoinCLIP v3.2 achieves **97.2% Top-1 accuracy** on a held-out benchmark, rising to **97.4% with OCR post-processing**.

| Mode | Top-1 | Top-5 |
|------|-------|-------|
| Pure CLIP | 97.23% | 97.98% |
| CLIP + OCR | 97.41% | 98.25% |

Evaluated on 8,111 test images across 1,103 unique coins. No coin appears in both training and test sets.

See [RESULTS.md](RESULTS.md) for full benchmark results, version history, and methodology.

## Training Pipeline

The model is trained using contrastive learning on design families -- groups of visually identical coins that differ only in mint year or minor variants. This lets the model learn what makes a Silver Maple Leaf look different from a Caribou quarter, rather than memorizing year text.

Key innovations:
- Design-family grouping (149 families from 962 coins)
- Hard-negative mining with confusion pair feedback loops
- OCR-based post-processing for disambiguation

See [APPROACH.md](APPROACH.md) for the full training methodology and architecture decisions.

## System Architecture

The three-phase pipeline is designed so the fastest, cheapest option handles the vast majority of queries, with progressively more powerful (and expensive) fallbacks for uncertain cases.

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## Stack

| Layer | Technologies |
|-------|-------------|
| iOS App | Swift, SwiftUI, SwiftData, Core ML, vDSP/Accelerate, Vision (OCR) |
| ML Model | MobileCLIP-S2, LoRA, OpenCLIP, PyTorch |
| Backend | Python, FastAPI, PostgreSQL + pgvector, Google Cloud Run |
| Training | PyTorch, A100 GPU, custom contrastive loss |
| Data Pipeline | Python, Playwright (scraping), HoughCircles (OpenCV) |

## Links

- **App Store**: [Canadian Coin Heads](https://apps.apple.com/app/canadian-coin-heads/id6740244078)
- **Website**: [canadiancoinheads.com](https://canadiancoinheads.com)
