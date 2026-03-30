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
| **MobileCLIP-S2** | **512** | **137 MB** | **<300ms** | **Selected.** Apple's distilled CLIP variant, optimized for mobile inference. Retains CLIP's embedding space for similarity search. |

MobileCLIP-S2 is Apple's knowledge-distilled variant of CLIP, trained with their DataCompDR methodology. It achieves competitive zero-shot accuracy with significantly lower latency than standard ViT models. The 512-dimensional embedding space is compatible with the broader CLIP ecosystem while being efficient enough for real-time on-device inference.

### Why LoRA

The MobileCLIP-S2 visual encoder has ~137M parameters. Fine-tuning all of them on 44,332 images risks severe overfitting, especially given that many coin classes have fewer than 50 training images.

LoRA (Low-Rank Adaptation) freezes the pretrained weights and trains small rank-decomposition matrices on the attention layers. This provides:

- **274K trainable parameters** (0.2% of the model) -- dramatically reduced overfitting risk
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

**The solution:** Group visually identical coins into "design families." All Silver Maple Leaves (1988-2026) form one family. All Bluenose dimes (1937-1989) form another. 1,103 unique coins are organized into 59 design families (919 coins) plus 154 solo classes for coins with unique designs.

The contrastive loss (in-batch negatives, InfoNCE-style) then teaches the model: "a Maple Leaf should be similar to other Maple Leaves and dissimilar from Caribou quarters." This is what the visual encoder can actually learn -- the structural and design differences between coin types.

Post-identification disambiguation within a design family is handled by OCR, which matches extracted year text against candidate coins. This division of labor -- vision model for design, OCR for year -- plays to each system's strengths.

## Training Pipeline

### Data Collection

25,591 unique real coin photographs (deduplicated from 44,332) sourced from:
- eBay auction listings (automated scraping with Playwright)
- Collector community submissions
- Royal Canadian Mint product photography

No synthetic data augmentation was used. The real-world variety in eBay listings -- different lighting, backgrounds, angles, image quality -- provides sufficient augmentation naturally. Training images range from professional studio shots to blurry phone photos on kitchen tables.

Post-training deduplication removed 18,887 byte-identical images (confirmed via CoinCLIP embedding identity + MD5 verification). The model was trained on the full 44K set; the deduplicated 25K set is used for the shipped embedding database.

### Preprocessing

1. **Resize**: 256x256 pixels (MobileCLIP-S2 input resolution)
2. **Normalization**: ImageNet channel means and standard deviations (required by the MobileCLIP-S2 pretrained weights)

The model is trained directly on raw photos without explicit coin isolation. Earlier versions used OpenCV HoughCircles for coin detection, but this was removed -- the model generalizes better when trained on the full range of backgrounds and framing found in real-world photos.

### Design Family Construction

Design families are derived from `reverseDesignId` metadata in the coin database. Each unique reverse design defines a family. The process:

1. Group all coins by `reverseDesignId`
2. Coins with the same reverse design across different years join the same family
3. Strip collectibles from standard circulation families (they share denominations but not designs)
4. Give circulation commemoratives their own unique IDs (e.g., poppy quarter, hockey toonie)
5. Result: 59 design families (919 coins) + 154 solo classes = 213 total classes from 1,103 unique coins

Examples:
- `maple-leaf-bullion` family: 38 coins (1988-2026), all with Walter Ott's Sugar Maple Leaf reverse
- `caribou-10c` family: ~80 coins (1937-present), all with Emanuel Hahn's Bluenose schooner reverse
- `1d-voyageur` family: ~50 coins (1935-1986), all with the canoe reverse

Solo classes exist for unique commemorative designs (e.g., the 1999 Nunavut toonie, 2004 Poppy quarter). The distinction matters: earlier versions lumped collectibles into standard families (e.g., putting a commemorative gold quarter into the caribou family), which produced contradictory training gradients. Cleaning this up was the single biggest accuracy improvement from v3.2 to v4.2.

### Hard Negative Mining

After each training epoch, the model's top confusion pairs are extracted -- cases where the model assigns high similarity to coins from different design families.

**Process:**
1. Encode all training images with current model weights
2. For each image, find the highest-similarity image from a *different* design family
3. Rank these cross-family similarities
4. Top pairs are flagged as "hard negatives" and weighted more heavily in subsequent epochs

CoinCLIP v4.2 identified 1,004 confusion pairs during training (up from 321 in v3.2). Common confusion categories:
- Coins with similar wildlife reverses (e.g., caribou vs. elk)
- Same denomination with similar composition (nickel 5-cent pieces across design eras)
- Commemorative coins that share design elements with circulation types

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | MobileCLIP-S2 (pretrained) |
| Fine-tuning method | LoRA (rank 16, alpha 32) |
| Trainable parameters | ~274K (of 137M total) |
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
| Top-1 | 99.2% | 99.9% | +0.7% |

The accuracy improvement is +0.7% on the holdout set, but the real impact shows on wild images: pure CLIP gets 25-33% on eBay photos and random internet images, while CLIP+OCR hits 100%. OCR eliminates *confident wrong answers* -- cases where the CLIP model is highly certain about a visually similar but incorrect coin. The denomination penalty is particularly effective at catching cross-denomination confusion.

### Parameter Optimization

OCR boost/penalty values were optimized via grid search on a validation split and evaluated on a held-out test split. The parameters are stored in a configuration file read by both the iOS app and the Python benchmark pipeline, ensuring consistency between production inference and evaluation.

## Embedding Pipeline

### Pre-Computation

All 25,591 unique photos are encoded with the merged LoRA model (base MobileCLIP-S2 + trained LoRA weights merged into a single checkpoint). This produces 25,427 512-dimensional L2-normalized embeddings covering 1,103 unique coins.

### Storage Format

Embeddings are stored as a flat binary file of Float32 values:
- `coin_embeddings.bin`: 25,427 embeddings x 512 dimensions x 4 bytes = 49.7 MB
- `coin_embeddings_index.json`: Maps each embedding position to its coin ID, design family, and metadata (~3.4 MB)

Both files ship in the iOS app bundle. No network request needed for the full embedding database.

### Similarity Search

At inference time:
1. The input photo is encoded to a 512-dim vector by the Core ML model (~250ms)
2. The vector is compared against all 25,427 stored embeddings via cosine similarity
3. Similarity computation uses Apple's vDSP framework (part of Accelerate) for SIMD-optimized matrix operations
4. Full search across 25,427 embeddings completes in <50ms
5. Top-k results are returned with similarity scores

Total on-device latency: model inference (~250ms) + similarity search (~50ms) + OCR (~variable) = under 300ms for most identifications.
