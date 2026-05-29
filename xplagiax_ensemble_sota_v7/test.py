"""
xplagiax_ensemble_sota_v7 — test.py
===================================
Rigorous evaluation harness. The whole point of v7 is to beat the commercial
detectors on the metrics that *matter for accusations*, not the vanity numbers
they advertise. So we report:

  - MCC          Matthews Correlation Coefficient — the official metric of the
                 GenAIDetect'25 shared task; robust to class imbalance.
  - AUROC        threshold-free ranking quality.
  - TPR@1%FPR    and TPR@0.1%FPR — recall at a *fixed, low* false-positive rate.
                 This is the number that decides whether a tool can be used to
                 accuse a student. Commercial tools rarely publish it.
  - FPR_human    realized false-positive rate on humans at the deployed threshold.
  - ECE          Expected Calibration Error.
  - Acc / F1     reported, but explicitly secondary.

And — critically — an ADVERSARIAL ROBUSTNESS SUITE that re-runs every metric
under attacks, reporting the degradation (the RAID / DAMAGE protocol):

  - homoglyph    Cyrillic/Greek lookalikes (2025.genaidetect-1.1 — broke ALL 7
                 SOTA detectors, MCC 0.64 -> -0.01). v7 should barely move.
  - zero_width   invisible-character injection.
  - char_noise   swap/delete perturbations.
  - whitespace   spacing / newline perturbations.

If a single in-distribution accuracy number is all a detector reports, treat
it as marketing. This harness reports the gap.

Usage:
  python test.py --artifacts ./sota_v7_artifacts            # uses held-out test
  python test.py --artifacts ./sota_v7_artifacts --data extra_test.parquet
  python test.py --artifacts ./sota_v7_artifacts --nonnative toefl.parquet
"""

from __future__ import annotations

import os
import json
import random
import logging
import argparse
from typing import Dict, List, Tuple, Callable

import numpy as np

from detector_final import SOTADetector, canonicalize_text, _CONFUSABLES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sota_v7.test")
random.seed(0)


# =====================================================================
# METRICS
# =====================================================================

def mcc(y: np.ndarray, pred: np.ndarray) -> float:
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    denom = math_sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom > 0 else 0.0


def math_sqrt(x: float) -> float:
    import math
    return math.sqrt(x) if x > 0 else 0.0


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney U)."""
    pos = score[y == 1]; neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    avg = sums / counts
    ranks = avg[inv]
    r_pos = ranks[y == 1].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def tpr_at_fpr(y: np.ndarray, score: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    """Return (TPR at the threshold whose human-FPR <= target, that threshold)."""
    neg = np.sort(score[y == 0])
    if len(neg) == 0:
        return float("nan"), float("nan")
    tau = np.quantile(neg, 1.0 - target_fpr)
    tpr = float(np.mean(score[y == 1] >= tau)) if np.any(y == 1) else float("nan")
    return tpr, float(tau)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """Standard binary ECE: confidence = max(p, 1-p) of the predicted class."""
    pred = (p >= 0.5).astype(int)
    conf = np.maximum(p, 1.0 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if i == bins - 1:
            m = (conf >= edges[i]) & (conf <= edges[i + 1])
        if not np.any(m):
            continue
        e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


# =====================================================================
# ADVERSARIAL ATTACKS
# =====================================================================

_INV = {}
for _k, _v in _CONFUSABLES.items():
    _INV.setdefault(_v, _k)
_ZW = "​"


def atk_homoglyph(t: str, rate: float = 0.5) -> str:
    return "".join(_INV.get(c, c) if (c in _INV and random.random() < rate) else c
                   for c in t)


def atk_zero_width(t: str, rate: float = 0.1) -> str:
    return "".join(c + (_ZW if random.random() < rate else "") for c in t)


def atk_char_noise(t: str, rate: float = 0.1) -> str:
    ws = t.split()
    for i, w in enumerate(ws):
        if len(w) > 4 and random.random() < rate:
            cs = list(w); j = random.randint(0, len(cs) - 2)
            cs[j], cs[j + 1] = cs[j + 1], cs[j]; ws[i] = "".join(cs)
    return " ".join(ws)


def atk_whitespace(t: str, rate: float = 0.1) -> str:
    return "".join(c + ("  " if (c == " " and random.random() < rate) else "")
                   for c in t)


ATTACKS: Dict[str, Callable[[str], str]] = {
    "clean": lambda t: t,
    "homoglyph": atk_homoglyph,
    "zero_width": atk_zero_width,
    "char_noise": atk_char_noise,
    "whitespace": atk_whitespace,
}


# =====================================================================
# EVALUATION
# =====================================================================

def score_rows(detector: SOTADetector, rows: List[dict],
               transform: Callable[[str], str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, p_ai, predicted_label) under a text transform."""
    y, p, pred = [], [], []
    for r in rows:
        text = transform(str(r["text"]))
        v = detector.classify(text)
        y.append(0 if r["label"] == "Human" else 1)
        p.append(v.p_ai)
        # INCONCLUSIVE/TAMPERED count as "not a confident human-clear":
        # for accusation safety, only AI_DETECTED is a positive.
        pred.append(1 if v.verdict == "AI_DETECTED" else 0)
    return np.array(y), np.array(p, dtype=np.float64), np.array(pred)


def evaluate_block(y, p, pred, deployed_target_fpr: float) -> dict:
    tpr1, _ = tpr_at_fpr(y, p, 0.01)
    tpr01, _ = tpr_at_fpr(y, p, 0.001)
    realized_fpr = float(np.mean(pred[y == 0] == 1)) if np.any(y == 0) else float("nan")
    acc = float(np.mean(pred == y))
    # F1 on the AI class
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "MCC": round(mcc(y, pred), 4),
        "AUROC": round(auroc(y, p), 4),
        "TPR@1%FPR": round(tpr1, 4),
        "TPR@0.1%FPR": round(tpr01, 4),
        "FPR_human_deployed": round(realized_fpr, 4),
        "ECE": round(ece(y, p), 4),
        "accuracy": round(acc, 4),
        "F1_ai": round(f1, 4),
        "n": int(len(y)),
    }


def load_eval_rows(path: str) -> List[dict]:
    from trainer import load_rows
    return load_rows(path)


def main():
    ap = argparse.ArgumentParser(description="Evaluate xplagiax_ensemble_sota_v7")
    ap.add_argument("--artifacts", default="./sota_v7_artifacts")
    ap.add_argument("--data", default=None,
                    help="optional external test set; default = held-out test_rows.pkl")
    ap.add_argument("--nonnative", default=None,
                    help="optional non-native human set to audit FP bias")
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    detector = SOTADetector(args.artifacts, target_fpr=args.target_fpr,
                            enable_zeroshot=not args.no_zeroshot)

    if args.data:
        rows = load_eval_rows(args.data)
    else:
        import pickle
        pth = os.path.join(args.artifacts, "test_rows.pkl")
        if not os.path.exists(pth):
            raise SystemExit("No test set: pass --data or train first (creates test_rows.pkl).")
        rows = pickle.load(open(pth, "rb"))
    if args.limit:
        rows = rows[:args.limit]
    logger.info("Evaluating on %d rows", len(rows))

    report = {"in_distribution": {}, "adversarial": {}, "summary": {}}

    # In-distribution + adversarial suite
    base = None
    for name, fn in ATTACKS.items():
        y, p, pred = score_rows(detector, rows, fn)
        block = evaluate_block(y, p, pred, args.target_fpr)
        if name == "clean":
            report["in_distribution"] = block
            base = block
        else:
            report["adversarial"][name] = block
        logger.info("[%-10s] MCC=%.3f  TPR@1%%FPR=%.3f  FPR=%.3f",
                    name, block["MCC"], block["TPR@1%FPR"],
                    block["FPR_human_deployed"])

    # Robustness summary: worst-case MCC drop vs clean (RAID-style headline).
    if base:
        drops = {k: round(base["MCC"] - v["MCC"], 4)
                 for k, v in report["adversarial"].items()}
        report["summary"]["mcc_drop_under_attack"] = drops
        report["summary"]["worst_attack"] = max(drops, key=drops.get) if drops else None

    # Non-native FP audit (the #1 real-world false-positive source).
    if args.nonnative:
        nn = load_eval_rows(args.nonnative)
        nn = [r for r in nn if r["label"] == "Human"]
        if nn:
            _, _, pred = score_rows(detector, nn, ATTACKS["clean"])
            report["summary"]["nonnative_FPR"] = round(float(np.mean(pred == 1)), 4)
            logger.info("Non-native human FPR = %.4f (lower is better)",
                        report["summary"]["nonnative_FPR"])

    out = os.path.join(args.artifacts, "evaluation_report.json")
    json.dump(report, open(out, "w"), indent=2)
    print("\n" + "=" * 64)
    print("EVALUATION REPORT  (in-distribution)")
    print("=" * 64)
    for k, v in report["in_distribution"].items():
        print(f"  {k:<22} {v}")
    print("\nAdversarial MCC drop (lower = more robust):")
    for k, v in report["summary"].get("mcc_drop_under_attack", {}).items():
        print(f"  {k:<22} {v}")
    print(f"\nFull report -> {out}")


if __name__ == "__main__":
    main()
