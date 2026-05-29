# xplagiax_ensemble_sota_v7

A SOTA AI-text detector designed to **beat the commercial tools (GPTZero,
Originality.ai, Copyleaks, Turnitin) on the metric that decides whether a
detector is safe to act on: TPR at a guaranteed low False-Positive Rate** —
not the in-distribution accuracy they advertise.

## Why this design beats the commercial tools

Every major commercial detector is, at its core, a **single-mechanism** system
(a fine-tuned transformer plus some perplexity), English-centric, and brittle
to: paraphrase/"humanizer" tools, **non-native English** (their #1 false-positive
source), short text, and the newest models. v7 attacks exactly those failure
modes with signals that fail **independently**, then refuses to accuse when
uncertain.

```
text ─► [Unicode canonicalization]  ← defeats homoglyph attack that broke ALL 7
        │                              SOTA detectors (2025.genaidetect-1.1)
        ├─ Branch A: supervised multi-class ensemble (DeBERTa-v3 + ModernBERT)
        │            length-routed experts · multi-class > binary (SzegedAI)
        ├─ Branch Z: zero-shot cross-perplexity (Binoculars + Fast-DetectGPT)
        │            training-free · robust under domain shift
        └─ Branch F: stylometric + proxy-perplexity (burstiness, log-rank, …)
                     │
              [stacking meta-learner: LightGBM/LogReg]
                     │
              [split-conformal threshold]  ← guarantees FPR ≤ α (default 1%)
                     │
        HUMAN · AI_DETECTED · INCONCLUSIVE · TAMPERED
```

The branches are uncorrelated by **mechanism** (learned features vs.
information-theoretic curvature vs. surface statistics), so the ensemble gains
real robustness — not just the seed-variance reduction of a homogeneous ensemble.

## Grounding in GenAIDetect 2025 (`2025.genaidetect-1`)

| Finding (paper) | What we did |
|---|---|
| Homoglyph attack drops MCC 0.64 → −0.01 on **all 7** SOTA detectors (`.1`) | `canonicalize_text()` — NFKC + confusables map + zero-width strip, **and** surfaces tampering as evidence |
| Multi-class > binary; length experts best (test F1 0.827) (`.15` SzegedAI) | multi-class heads + short/long length routing |
| Adversarial training, not normalization, gives robustness (`.9` DAMAGE) | `MultilevelAugmenter`: char + homoglyph + paraphrase hooks at train time |
| Dev→Test F1 gap ≈ 18 pts even for the winner | dedup + **group-aware split** + held-out test; report the gap honestly |
| Official metric is **MCC** | `test.py` reports MCC first, plus TPR@1%/0.1%FPR, AUROC, ECE |

## Files
- `detector_final.py` — inference: canonicalizer, the three branches, meta-learner, conformal FP-control, `SOTADetector`. Runs with zero artifacts (degrades to a safe abstaining path).
- `trainer.py` — data hygiene → adversarial augmentation → multi-class training → temperature calibration → meta-learner → conformal fit.
- `test.py` — MCC / AUROC / TPR@FPR / ECE + the adversarial robustness suite + non-native FP audit.

## Quick start
```bash
# Train (Colab/A100/H100). Expects columns: text, label|target_model [, source/domain/group_id]
python trainer.py --data data.parquet --out ./sota_v7_artifacts \
    --models modernbert,deberta --length-experts --target-fpr 0.01

# Evaluate on the held-out test split + adversarial suite
python test.py --artifacts ./sota_v7_artifacts --nonnative toefl_humans.parquet
```

Heavy deps (`torch`, `transformers`, `lightgbm`) are lazy-imported; the
canonicalizer, stylometry, conformal calibrator and metrics run on plain
CPU/NumPy. `python detector_final.py` runs a self-contained smoke test
including the homoglyph-attack defense.

## Decision policy (the "no false positives" engine)
The conformal layer abstains (`INCONCLUSIVE`) instead of accusing whenever the
score is below the FPR-controlled threshold, and never clears obfuscated text
as human (`TAMPERED`). This is Turnitin-style conservatism made **explicit and
statistically guaranteed**, which is what lets the tool be used for real
academic-integrity decisions.
```
```
> Status: reference implementation / scaffolding. Branch A and Branch Z require
> trained weights / model downloads to reach full accuracy; the pipeline,
> calibration, FP-control and evaluation are complete and tested.
