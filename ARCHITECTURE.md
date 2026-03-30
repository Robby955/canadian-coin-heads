# System Architecture

## Overview

Canadian Coin Heads uses a progressive three-phase identification pipeline. The design principle: resolve as many queries as possible on-device, escalating to the cloud only when the on-device model is uncertain. This minimizes latency, bandwidth, and cost for the 97% of identifications that the on-device model handles confidently.

## Three-Phase Pipeline

```
                            +------------------+
                            |   Coin Photo     |
                            |   (from camera   |
                            |    or gallery)    |
                            +--------+---------+
                                     |
                                     v
                  +------------------------------------------+
                  |         PHASE 0: On-Device               |
                  |         Target: <300ms                   |
                  |                                          |
                  |  +------------------------------------+  |
                  |  | CoinCLIP v4.2 (Core ML)            |  |
                  |  | MobileCLIP-S2 + LoRA (137 MB)      |  |
                  |  | Input: 256x256 photo               |  |
                  |  | Output: 512-dim embedding          |  |
                  |  +------------------+-----------------+  |
                  |                     |                    |
                  |                     v                    |
                  |  +------------------------------------+  |
                  |  | vDSP Similarity Search             |  |
                  |  | 25,427 pre-computed embeddings     |  |
                  |  | Cosine similarity via Accelerate   |  |
                  |  | Full search: <50ms                 |  |
                  |  +------------------+-----------------+  |
                  |                     |                    |
                  |                     v                    |
                  |  +------------------------------------+  |
                  |  | OCR Refinement                     |  |
                  |  | Apple Vision text extraction       |  |
                  |  | Year match: +0.08 boost            |  |
                  |  | Denom mismatch: -0.20 penalty      |  |
                  |  | Keyword match: +0.04 (cap +0.12)   |  |
                  |  +------------------+-----------------+  |
                  |                     |                    |
                  +------------------------------------------+
                                        |
                          confidence >= 0.85
                          AND gap >= 0.10?
                                        |
                           +------------+------------+
                           |                         |
                          YES                        NO
                           |                         |
                           v                         v
                    +-----------+     +------------------------------------------+
                    |  RESULT   |     |         PHASE 1: Cloud CLIP             |
                    |  (97% of  |     |         Target: 2-5s                    |
                    |  queries) |     |                                          |
                    +-----------+     |  ViT-B-32 (LAION2B pretrained)          |
                                      |  PostgreSQL + pgvector                  |
                                      |  1,904 server-side embeddings           |
                                      |  Google Cloud Run                       |
                                      +------------------+---------------------+
                                                         |
                                           confidence >= 0.85?
                                                         |
                                            +------------+------------+
                                            |                         |
                                           YES                        NO
                                            |                         |
                                            v                         v
                                     +-----------+    +------------------------------------------+
                                     |  RESULT   |    |         PHASE 2: Hybrid Analysis        |
                                     +-----------+    |         Target: 15-45s                  |
                                                      |                                          |
                                                      |  Claude Vision (visual reasoning)        |
                                                      |  + Structured database scoring:          |
                                                      |    - Metal type match: +20/-25           |
                                                      |    - Year range match: +25/+10/-10       |
                                                      |    - Design element match: +25           |
                                                      |    - Special features: +10 to +20        |
                                                      |                                          |
                                                      +------------------+---------------------+
                                                                         |
                                                                         v
                                                                  +-----------+
                                                                  |  RESULT   |
                                                                  +-----------+
```

## Why Three Phases

The progressive architecture is motivated by a simple observation: most coins are common, well-photographed types that the on-device model handles confidently. Sending every identification request to the cloud would add unnecessary latency and cost.

**Phase 0 handles ~97% of queries.** The on-device model resolves the vast majority of identifications in under 300ms with no network dependency. This means the app works offline, responds instantly, and costs nothing per query.

**Phase 1 catches the next ~2%.** When the on-device model is uncertain (similarity score below 0.85 or small gap to the second-best match), the cloud CLIP model provides a second opinion. It uses a different model architecture (ViT-B-32) and different pretrained weights (LAION2B), so its failure modes are largely independent of the on-device model.

**Phase 2 handles the hardest ~1%.** For the most ambiguous cases, Claude Vision provides visual reasoning capabilities that pure embedding similarity cannot: reading partially obscured text, recognizing design motifs, and reasoning about coin characteristics. Combined with structured database scoring (metal type, year range, design elements), this phase achieves reliable identification even for the hardest cases.

## On-Device Model

### CoinCLIP v4.2

The on-device model is a MobileCLIP-S2 visual encoder fine-tuned with LoRA on 44,332 real coin photographs (25,591 unique after deduplication).

**Architecture:**
- Base: MobileCLIP-S2 visual encoder (Apple's distilled CLIP variant)
- Fine-tuning: LoRA rank-16 on Q/K/V attention projections
- Input: 256 x 256 RGB image, ImageNet normalization
- Output: 512-dimensional L2-normalized embedding
- Trainable parameters: 274K (of 137M total)

**Core ML conversion:**
- Weights: LoRA merged into base model (zero additional inference cost)
- Precision: Float32 (MobileCLIP-S2's reparameterized convolutions have weights that exceed Float16 range, causing NaN on device)
- File size: 137 MB on disk
- Compatibility: iOS 17+ (iPhone 12 and later)
- Inference: CPU compute (required for Float32 precision; ANE/GPU use Float16 arithmetic internally)

**Latency breakdown (iPhone 12):**

| Stage | Time |
|-------|------|
| Image preprocessing (resize, normalize) | ~10ms |
| Core ML model inference | ~200-250ms |
| vDSP similarity search (25,427 embeddings) | ~30-50ms |
| OCR text extraction + scoring | ~20-50ms |
| **Total** | **<300ms** |

### Embedding Database

The 25,427 pre-computed embeddings are stored in two files bundled with the app:

- **Binary file** (49.7 MB): Flat array of Float32 values, 25,427 embeddings x 512 dimensions. Memory-mapped at runtime for zero-copy access.
- **Index file** (~3.4 MB JSON): Maps each embedding position to coin ID, design family, denomination, year, and metadata needed for result display.

Similarity search is implemented with vDSP matrix multiplication (Apple Accelerate framework), computing cosine similarity between the query embedding and all 25,427 stored embeddings in a single vectorized operation.

## Cloud Fallback

### Phase 1: Cloud CLIP

When the on-device model returns low confidence, the original photo is sent to a Cloud Run service for a second-opinion identification.

**Key design decision:** The cloud model uses a *different* embedding space (LAION2B-pretrained ViT-B-32) rather than the same CoinCLIP model. This is intentional -- if CoinCLIP is confused by a particular photo, a model with different pretrained features may not be. The two models' error modes are largely independent, so disagreement between them is informative.

- Model: ViT-B-32 (OpenAI CLIP, LAION2B pretrained)
- Search: PostgreSQL + pgvector (approximate nearest neighbor)
- Embeddings: 1,904 server-side coin embeddings
- Hosting: Google Cloud Run (auto-scaling, pay-per-request)
- Latency: 2-5 seconds including network round-trip

### Phase 2: Hybrid Analysis

For the most uncertain identifications, the system combines LLM-based visual reasoning with structured database scoring.

**Visual reasoning:** The coin photo is analyzed by Claude Vision, which can reason about design elements, read partially obscured text, identify metal types from color, and describe stylistic features that pure embedding similarity cannot capture.

**Structured scoring:** The LLM's observations are scored against the coin database using weighted criteria:

| Criterion | Match Score | Mismatch Score |
|-----------|-------------|----------------|
| Metal type | +20 | -25 |
| Year (exact) | +25 | -- |
| Year (within range) | +10 | -10 |
| Reverse design/animal | +25 | -- |
| Gold plating | +20 | -- |
| Proof/specimen finish | +10 | -- |

The combination of visual reasoning and structured scoring provides reliable identification even for coins that are ambiguous in embedding space.

## Multi-Angle Support

The pipeline supports a second photo (typically obverse + reverse of the same coin). When two photos are provided:

1. Both photos are independently encoded and matched
2. The system selects the best match across both photos
3. If one photo yields a confident match and the other is uncertain, the confident match is used
4. If both photos yield different confident matches, additional disambiguation logic applies

This significantly improves accuracy for cases where one side of the coin is more distinctive than the other (e.g., a commemorative reverse with a standard obverse).
