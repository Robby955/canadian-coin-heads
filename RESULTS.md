# Benchmark Results

## CoinCLIP v3.2 -- 97.2% Top-1 Accuracy

### Summary

| Mode | Top-1 | Top-5 | Test Images | Unique Coins |
|------|-------|-------|-------------|--------------|
| Pure CLIP | 97.23% | 97.98% | 8,111 | 1,103 |
| CLIP + OCR | 97.41% | 98.25% | 8,111 | 1,103 |

Evaluated on 8,111 held-out test images across 1,103 unique coins. No coin appears in both training and test sets. Test images were sourced independently from training images to prevent data leakage.

### Version History

| Version | Architecture | Top-1 | Top-5 | Training Images | Notes | Date |
|---------|-------------|-------|-------|-----------------|-------|------|
| v1 (baseline) | ViT-B-32 LAION2B | ~60% | ~75% | 0 (zero-shot) | Off-the-shelf CLIP, no fine-tuning | Jan 2026 |
| v2 | MobileCLIP-S2 + LoRA | 93.5% | 98.8% | 20,622 | First fine-tuned model, per-coin classes | Feb 2026 |
| v3 | MobileCLIP-S2 + LoRA | 87.4%* | 94.2%* | 33,000 | Introduced design-family grouping | Mar 2026 |
| v3.2 | MobileCLIP-S2 + LoRA | **97.2%** | **98.0%** | 42,674 | Refined training, hard-negative mining, OCR | Mar 2026 |

*v3 used a stricter holdout evaluation methodology introduced partway through development. v3.2 used the same strict methodology, making the two directly comparable. The apparent regression from v2 to v3 reflects the harder evaluation, not a worse model.

### What Improved from v2 to v3.2

**More training data.** 20,622 images in v2 grew to 42,674 in v3.2 (2.1x increase). Coverage expanded from ~500 coin types to ~1,429, with particular focus on filling gaps in pre-Confederation provincial coins and modern commemoratives.

**Design-family grouping.** v2 treated each coin-year as an independent class (e.g., 2020 Maple Leaf and 2021 Maple Leaf were separate targets). v3+ groups visually identical coins into 149 design families and trains with contrastive loss. This teaches the model to distinguish *designs*, not memorize year text -- a task better suited to a vision model.

**Hard-negative mining.** After each training epoch, the top confusion pairs are extracted and fed back as hard negatives. v3.2 mined 321 confusion pairs (up from 95 in v3), focusing the model's capacity on the hardest distinctions.

**Confusion pair extraction.** Systematic identification of which coin pairs the model confuses most. This feedback loop informed both training (hard negatives) and data collection (targeted photo acquisition for confusing pairs).

**OCR post-processing.** Year, denomination, and keyword matching from on-device OCR provides a complementary signal to visual features. The +0.2% accuracy gain understates the qualitative improvement: OCR primarily eliminates confident wrong answers.

**Expanded embedding coverage.** 44,328 pre-computed embeddings covering 962 unique coins (up from ~500 in v2). More reference photos per coin improves robustness to lighting and angle variation.

### Evaluation Methodology

**Data split.** The 42,674 training photos and 8,111 test photos are partitioned at the image level. Every test image was sourced independently from the training set (different eBay listings, different photographers, different lighting conditions). No image appears in both sets.

**Per-coin accuracy.** Top-1 accuracy is computed per-image (does the top-ranked result match the ground truth coin?), then aggregated across all 8,111 test images. Top-5 accuracy checks whether the correct coin appears anywhere in the top 5 results.

**Design family vs. exact coin.** Training uses design-family accuracy (did you identify the correct reverse design?). Benchmark uses exact-coin accuracy (did you identify the correct coin, including year?). This means the benchmark is strictly harder than what the model is directly optimized for -- disambiguation within a design family relies on OCR post-processing.

**Confidence scoring.** The CLIP similarity score is used as the confidence metric. A result is considered "high confidence" if the top similarity score exceeds 0.85 AND the gap between the top and second result is at least 0.10. High-confidence results skip cloud fallback phases entirely.

### Failure Analysis

The 2.8% of Top-1 errors fall into predictable categories:

1. **Within-family confusion (~40% of errors).** The model correctly identifies the design family but OCR fails to disambiguate the exact year. Common with worn coins where year text is illegible.

2. **Similar-design confusion (~35% of errors).** Coins with genuinely similar reverse designs are confused -- e.g., different wildlife species at similar scales, or commemorative coins that share design elements with circulation types.

3. **Photo quality issues (~20% of errors).** Extremely dark, blurry, or partially cropped photos where the reverse design is not fully visible.

4. **Rare coin underrepresentation (~5% of errors).** Coins with fewer than 5 training images. These are typically pre-Confederation provincial issues or limited-mintage commemoratives.

### Limitations

- Accuracy is measured on coins with available reference photography. Very rare coins (fewer than 5 known specimens or no available photos) are not represented in the benchmark.
- Heavily worn coins (e.g., "About Good" grade, where design details are nearly flat) may perform below these benchmarks. The training set includes worn coins but is weighted toward Fine grade and above.
- The benchmark covers Canadian coins only. The model has no training data for other countries' coinage and should not be expected to identify them.
- Benchmark images are of single coins on relatively clean backgrounds. Multi-coin photos or coins partially obscured by holders/cases may perform differently.
