# Training Approach

## Problem Statement

Identifying Canadian coins from a single photograph is a deceptively hard visual recognition problem.

**Why it is hard:**

1. **Shared obverse designs across decades.** Every Canadian coin from 1953-present features one of a handful of Queen Elizabeth II or King Charles III portraits on the obverse. A 1985 quarter and a 2005 quarter have the same front face -- the model must rely entirely on the reverse design and subtle date text.

2. **Tiny, worn year text.** The mint year is typically 2-3mm tall on the physical coin. On a phone photo, that is a handful of pixels. Circulated coins may have the year partially or fully worn away.

3. **Extreme lighting and angle variation.** Metallic surfaces produce specular highlights, color shifts, and reflections that vary dramatically with angle. A silver coin can appear white, grey, black, or rainbow-tinted depending on the photo.

4. **Scale of the catalog.** 5,791 coins across 168 years (1858-2026), spanning circulation coins, bullion, commemoratives, provincial pre-Confederation issues, and collector sets. Many share nearly identical designs with only year or mint mark differences.

5. **Visual near-duplicates.** The Silver Maple Leaf bullion coin has been issued annually since 1988 with the same reverse design. All 38 years must be distinguishable. The same applies to dozens of other long-running series.

## Architecture Decisions

### Why MobileCLIP-S2

The on-device model must run in under 300ms on an iPhone 12, fit within a reasonable app binary size, and produce embeddings suitable for similarity search against a large reference database.

Models evaluated:

| Model | Embedding Dim | Core ML Size | iPhone 12 Latency | Why Not |
|-------|--------------|-------------|-------------------|---------|
| ViT-B-32 (OpenAI CLIP) | 512 | ~150 MB | ~800ms | Too slow for on-device, too large |
| DINO v2 (ViT-S) | 384 | ~90 MB | ~400ms | No text-image alignment; cannot leverage OCR or text prompts |
| EfficientNet-B3 | 1536 | ~50 MB | ~200ms | Classification-only; no zero-shot or embedding similarity capability |
| **MobileCLIP-S2** | **512** | **68.8 MB** | **<300ms** | **Selected.** Apple's distilled CLIP variant, optimized for mobile inference. Retains CLIP's embedding space for similarity search. |

MobileCLIP-S2 is Apple's knowledge-distilled variant of CLIP, trained with their DataCompDR methodology. It achieves competitive zero-shot accuracy with significantly lower latency than standard ViT models. The 512-dimensional embedding space is compatible with the broader CLIP ecosystem while being efficient enough for real-time on-device inference.

### Why LoRA

The MobileCLIP-S2 visual encoder has ~137M parameters. Fine-tuning all of them on 42,674 images risks severe overfitting, especially given that many coin classes have fewer than 50 training images.

LoRA (Low-Rank Adaptation) freezes the pretrained weights and trains small rank-decomposition matrices on the attention layers. This provides:

- **0.4M trainable parameters** (0.3% of the model) -- dramatically reduced overfitting risk
- **2-hour training time** on a single A100 -- fast iteration cycles
- **Mergeable weights** -- at inference time, LoRA weights are merged into the base model with zero additional latency
- **Preserved pretrained features** -- the model retains MobileCLIP-S2's general visual understanding while learning coin-specific discriminations

LoRA configuration:
- Rank: 16
- Alpha: 32
- Target modules: visual encoder Q, K, V projections
- Dropout: 0.05

### Why Contrastive Loss with Design Families

The naive approach -- treating each coin as an independent class and training with cross-entropy -- fails for this domain.

**The problem:** A 2020 Silver Maple Leaf and a 2021 Silver Maple Leaf are visually identical except for a single digit in tiny text. Training with 38 separate classes for 38 years of Maple Leaves forces the model to learn OCR rather than visual features. But the model is not a text recognition system -- it is an image encoder.

**The solution:** Group visually identical coins into "design families." All Silver Maple Leaves (1988-2026) form one family. All Bluenose dimes (1937-1989) form another. This reduces 962 unique coins to 149 design families.

The contrastive loss (in-batch negatives, InfoNCE-style) then teaches the model: "a Maple Leaf should be similar to other Maple Leaves and dissimilar from Caribou quarters." This is what the visual encoder can actually learn -- the structural and design differences between coin types.

Post-identification disambiguation within a design family is handled by OCR, which matches extracted year text against candidate coins. This division of labor -- vision model for design, OCR for year -- plays to each system's strengths.

## Training Pipeline

### Data Collection

42,674 real coin photographs sourced from:
- eBay auction listings (automated scraping with Playwright)
- Collector community submissions
- Royal Canadian Mint product photography

No synthetic data augmentation was used. The real-world variety in eBay listings -- different lighting, backgrounds, angles, image quality -- provides sufficient augmentation naturally. Training images range from professional studio shots to blurry phone photos on kitchen tables.

### Preprocessing

1. **Coin detection**: OpenCV HoughCircles to locate the circular coin region in cluttered photos (eBay listings often show coins on fabric, in cases, or alongside other items)
2. **Center crop**: Extract the detected coin region with a small margin
3. **Resize**: 256x256 pixels (MobileCLIP-S2 input resolution)
4. **Normalization**: ImageNet channel means and standard deviations (required by the MobileCLIP-S2 pretrained weights)

### Design Family Construction

Design families are derived from `reverseDesignId` metadata in the coin database. Each unique reverse design defines a family. The process:

1. Group all coins by `reverseDesignId`
2. Coins with the same reverse design across different years join the same family
3. Result: 149 design families from 962 unique coins

Examples:
- `maple-leaf-bullion` family: 38 coins (1988-2026), all with Walter Ott's Sugar Maple Leaf reverse
- `caribou-10c` family: ~80 coins (1937-present), all with Emanuel Hahn's Bluenose schooner reverse
- `1d-voyageur` family: ~50 coins (1935-1986), all with the canoe reverse

Single-member families exist for unique commemorative designs (e.g., the 1999 Nunavut toonie).

### Hard Negative Mining

After each training epoch, the model's top confusion pairs are extracted -- cases where the model assigns high similarity to coins from different design families.

**Process:**
1. Encode all training images with current model weights
2. For each image, find the highest-similarity image from a *different* design family
3. Rank these cross-family similarities
4. Top pairs are flagged as "hard negatives" and weighted more heavily in subsequent epochs

CoinCLIP v3.2 identified 321 confusion pairs during training (up from 95 in v3). Common confusion categories:
- Coins with similar wildlife reverses (e.g., caribou vs. elk)
- Same denomination with similar composition (nickel 5-cent pieces across design eras)
- Commemorative coins that share design elements with circulation types

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | MobileCLIP-S2 (pretrained) |
| Fine-tuning method | LoRA (rank 16, alpha 32) |
| Trainable parameters | ~0.4M (of 137M total) |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate schedule | Cosine decay |
| Peak learning rate | 2e-4 |
| Training duration | ~2 hours on A100 |
| Precision | bfloat16 (mixed precision) |
| Loss | InfoNCE contrastive (in-batch negatives) |

## OCR Post-Processing

After the CLIP model produces its top-k candidate matches, OCR post-processing refines the ranking using text extracted from the coin photo.

### Pipeline

1. **Text extraction**: Apple's Vision framework runs on-device OCR on the coin image
2. **Year matching**: Regex extracts 4-digit numbers (1800-2030 range). Fuzzy match against candidate coins' mint years. A year match boosts the candidate by +0.08 in similarity score.
3. **Denomination matching**: Regex patterns for denomination text ("25 CENTS", "DOLLAR", "$2", "10 CENTS", etc.). A denomination *mismatch* penalizes the candidate by -0.20. This is the strongest OCR signal -- if the coin clearly says "25 CENTS", a dollar coin candidate should be suppressed.
4. **Keyword matching**: Metal-specific terms ("FINE SILVER", "9999", "PURE GOLD"), design keywords, and series identifiers. Each keyword match boosts by +0.04, capped at +0.12 total.

### Impact

| Metric | Without OCR | With OCR | Delta |
|--------|-------------|----------|-------|
| Top-1 | 97.23% | 97.41% | +0.18% |
| Top-5 | 97.98% | 98.25% | +0.27% |

The accuracy improvement is modest (+0.2%), but the qualitative impact is larger: OCR primarily eliminates *confident wrong answers* -- cases where the CLIP model is highly certain about a visually similar but incorrect coin. The denomination penalty is particularly effective at catching cross-denomination confusion.

### Parameter Optimization

OCR boost/penalty values were optimized via grid search on a validation split and evaluated on a held-out test split. The parameters are stored in a configuration file read by both the iOS app and the Python benchmark pipeline, ensuring consistency between production inference and evaluation.

## Embedding Pipeline

### Pre-Computation

All 42,674 training photos are encoded with the merged LoRA model (base MobileCLIP-S2 + trained LoRA weights merged into a single checkpoint). This produces 44,328 512-dimensional embeddings. The count exceeds the photo count because some photos generate multiple crops during preprocessing.

### Storage Format

Embeddings are stored as a flat binary file of Float32 values:
- `coin_embeddings.bin`: 44,328 embeddings x 512 dimensions x 4 bytes = 86.6 MB
- `coin_embeddings_index.json`: Maps each embedding position to its coin ID, design family, and metadata

Both files ship in the iOS app bundle. No network request needed for the full embedding database.

### Similarity Search

At inference time:
1. The input photo is encoded to a 512-dim vector by the Core ML model (~250ms)
2. The vector is compared against all 44,328 stored embeddings via cosine similarity
3. Similarity computation uses Apple's vDSP framework (part of Accelerate) for SIMD-optimized matrix operations
4. Full search across 44,328 embeddings completes in <50ms
5. Top-k results are returned with similarity scores

Total on-device latency: model inference (~250ms) + similarity search (~50ms) + OCR (~variable) = under 300ms for most identifications.
