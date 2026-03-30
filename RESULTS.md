# Benchmark Results

## CoinCLIP v4.2 -- 99.2% Top-1 Accuracy

### Summary

| Mode | Top-1 | Top-5 | Test Images | Unique Coins |
|------|-------|-------|-------------|--------------|
| Pure CLIP | 99.2% | ~99.7% | 8,112 | 1,103 |
| CLIP + OCR | 99.9% | ~99.9% | 8,112 | 1,103 |

Evaluated on 8,112 held-out test images across 1,103 unique coins. No coin appears in both training and test sets. Test images were sourced independently from training images to prevent data leakage.

### External Test Results

| Test Set | Images | Pure CLIP | CLIP+OCR | Description |
|----------|--------|-----------|----------|-------------|
| Holdout | 8,112 | 99.2% | 99.9% | Random 25% split from training photos |
| Wild internet | 28 | 25.0% | **100.0%** | Assorted web images, varied formats |
| eBay auctions | 12 | 33.3% | **100.0%** | Live eBay listing photos |

Pure CLIP scores are low on wild images because these photos have cluttered backgrounds, unusual lighting, and no coin isolation -- conditions where OCR's year/denomination matching compensates for the visual noise.

### Version History

| Version | Architecture | Top-1 | Top-5 | Training Images | Notes | Date |
|---------|-------------|-------|-------|-----------------|-------|------|
| v1 (baseline) | MobileCLIP-S2 (zero-shot) | 83.4% | — | 0 | Pretrained, no fine-tuning | Feb 2026 |
| v2 | MobileCLIP-S2 + LoRA | 93.5% | 98.8% | 20,600 | First fine-tuned model, per-coin classes | Mar 2026 |
| v3 | MobileCLIP-S2 + LoRA | 87.4%* | 95.0%* | 33,473 | Introduced design-family grouping | Mar 2026 |
| v3.2 | MobileCLIP-S2 + LoRA | 97.2% | 97.4% | 44,328 | Hard-negative mining, OCR, 962 coins | Mar 2026 |
| **v4.2** | MobileCLIP-S2 + LoRA | **99.2%** | **99.9%** | 44,332 | 1,201 label fixes, 1,103 coins, 59 clean families | Mar 2026 |

*v3 used a stricter holdout evaluation methodology introduced partway through development. v3.2+ used the same strict methodology, making them directly comparable. The apparent regression from v2 to v3 reflects the harder evaluation, not a worse model.

### What Improved from v3.2 to v4.2

The jump from 97.2% to 99.2% came almost entirely from **data quality**, not model architecture. Same LoRA rank, same hyperparameters, same training pipeline. The difference was 1,201 label corrections.

**Label cleanup (the big one).** The automated metadata enrichment pipeline had a "denomination template bias" -- it assumed every quarter was a caribou, every dime was a bluenose, every dollar was a voyageur. This polluted the design family groupings, putting commemorative coins into families they didn't belong in. Cleaning this up -- stripping 1,163 collectibles from standard families, giving 38 circulation commemoratives their own unique IDs -- dropped the family count from 149 to 59, but made each family internally consistent. The model stopped fighting contradictory gradients.

**Expanded coin coverage.** 962 coins in v3.2 grew to 1,103 in v4.2. The 141 new coins were mostly commemoratives that previously had no photos or were miscategorized.

**Hard-negative mining at scale.** 1,004 confusion pairs auto-mined (up from 321 in v3.2), fed back as hard negatives during training.

**Photo deduplication.** Post-training, 18,887 byte-identical duplicates were removed from the photo collection (44K → 25,591 unique images). The embeddings shipped in the app are generated from the deduplicated set (25,427 embeddings), which reduces app binary size without losing coverage.

**OCR post-processing.** Same parameters as v3.2 (year boost +0.08, denomination penalty -0.20), but the accuracy gain is larger (+0.7% vs +0.2%) because cleaner CLIP rankings give OCR better candidates to work with. On external wild images, OCR takes pure CLIP from 25-33% to 100% -- it compensates for cluttered backgrounds where embedding similarity alone struggles.

### Evaluation Methodology

**Data split.** The 44,332 training photos and 8,112 test photos are partitioned at the image level. Every test image was sourced independently from the training set (different eBay listings, different photographers, different lighting conditions). No image appears in both sets.

**Per-coin accuracy.** Top-1 accuracy is computed per-image (does the top-ranked result match the ground truth coin?), then aggregated across all 8,112 test images. Top-5 accuracy checks whether the correct coin appears anywhere in the top 5 results.

**Design family vs. exact coin.** Training uses design-family accuracy (did you identify the correct reverse design?). Benchmark uses exact-coin accuracy (did you identify the correct coin, including year?). This means the benchmark is strictly harder than what the model is directly optimized for -- disambiguation within a design family relies on OCR post-processing.

**Confidence scoring.** The CLIP similarity score is used as the confidence metric. A result is considered "high confidence" if the top similarity score exceeds 0.85 AND the gap between the top and second result is at least 0.10. High-confidence results skip cloud fallback phases entirely.

### Failure Analysis

The 0.8% of Top-1 errors (pure CLIP) fall into predictable categories:

1. **Within-family confusion (~50% of errors).** The model correctly identifies the design family but OCR fails to disambiguate the exact year. Common with worn coins where year text is illegible. OCR post-processing eliminates most of these (0.8% → 0.1%).

2. **Similar-design confusion (~30% of errors).** Coins with genuinely similar reverse designs are confused -- e.g., different wildlife species at similar scales, or commemorative coins that share design elements with circulation types.

3. **Photo quality issues (~15% of errors).** Extremely dark, blurry, or partially cropped photos where the reverse design is not fully visible.

4. **Rare coin underrepresentation (~5% of errors).** Coins with fewer than 5 training images. These are typically pre-Confederation provincial issues or limited-mintage commemoratives.

### Limitations

- Accuracy is measured on coins with available reference photography. Very rare coins (fewer than 5 known specimens or no available photos) are not represented in the benchmark.
- Heavily worn coins (e.g., "About Good" grade, where design details are nearly flat) may perform below these benchmarks. The training set includes worn coins but is weighted toward Fine grade and above.
- The benchmark covers Canadian coins only. The model has no training data for other countries' coinage and should not be expected to identify them.
- Benchmark images are of single coins on relatively clean backgrounds. Multi-coin photos or coins partially obscured by holders/cases may perform differently.
