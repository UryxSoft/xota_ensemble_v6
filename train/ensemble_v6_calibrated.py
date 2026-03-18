"""
Ensemble v6.1 — False Positive Reduction Layer
================================================
Wraps the raw ensemble detector with calibrated thresholds,
agreement gating, and two-stage classification to minimize
the rate of humans incorrectly flagged as AI.

Key principle: In academic integrity, a false positive
(human accused of using AI) is 10x worse than a false negative
(AI text passing as human). The system must be tuned accordingly.

5 layers of FP protection:
  1. Confidence thresholding (configurable profiles)
  2. Model agreement gating (require consensus)
  3. Temperature scaling (Platt calibration)
  4. Two-stage classification (binary then attribution)
  5. Sentence-level aggregation (for documents)

Usage:
    from ensemble_v6_calibrated import CalibratedDetector

    detector = CalibratedDetector(
        ensemble_dir="/path/to/ensemble_v6/",
        fp_tolerance="strict",
    )

    # Single text
    result = detector.classify("Text to analyze...")

    # Full document
    doc_result = detector.classify_document("Full essay text here...")

    # Batch
    results = detector.classify_batch(["text1", "text2", ...])

    # Calibration (run ONCE after training):
    from ensemble_v6_calibrated import calibrate_from_saved_logits
    calibrate_from_saved_logits("/path/to/ensemble_v6/")
"""

import os
import re
import math
import json
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =====================================================================
# THRESHOLD PROFILES
# =====================================================================

@dataclass
class ThresholdConfig:
    """Threshold configuration for false positive control.

    ai_confidence_min: minimum ensemble P(AI) to flag as AI
    agreement_min:     minimum model agreement ratio (0.33 | 0.67 | 1.0)
    margin_min:        minimum gap between top-1 and top-2 class probabilities
    inconclusive_band: if P(Human) falls in this range, verdict = INCONCLUSIVE
    """
    ai_confidence_min: float
    agreement_min: float
    margin_min: float
    inconclusive_band: Tuple[float, float]


THRESHOLD_PROFILES: Dict[str, ThresholdConfig] = {
    "strict": ThresholdConfig(
        ai_confidence_min=0.75,
        agreement_min=1.0,
        margin_min=0.25,
        inconclusive_band=(0.20, 0.55),
    ),
    "moderate": ThresholdConfig(
        ai_confidence_min=0.60,
        agreement_min=0.67,
        margin_min=0.15,
        inconclusive_band=(0.25, 0.50),
    ),
    "permissive": ThresholdConfig(
        ai_confidence_min=0.45,
        agreement_min=0.33,
        margin_min=0.10,
        inconclusive_band=(0.30, 0.45),
    ),
}

# Must match training config exactly
LABEL_ORDER = ["Human", "Claude", "DeepSeek", "Gemini", "GPT", "Grok", "Mistral"]
HUMAN_IDX = LABEL_ORDER.index("Human")
TRAINING_ORDER = ["deberta", "electra", "modernbert"]

# FK constants (must match preprocess)
FK_MAX_CHARS = 3000
FK_CLAMP_MIN = -5.0
FK_CLAMP_MAX = 30.0
FK_DEFAULT = 0.0


# =====================================================================
# TEMPERATURE SCALING
# =====================================================================

class TemperatureScaler:
    """Platt-style temperature scaling for calibrated probabilities.

    Neural networks are typically overconfident. Temperature scaling
    divides logits by T before softmax:
      T > 1.0 = less confident (more calibrated)
      T < 1.0 = more confident
      T = 1.0 = unchanged
    """

    def __init__(self, init_temperature: float = 1.5):
        self.temperature: float = init_temperature
        self._fitted: bool = False

    def fit(self, logits: np.ndarray, labels: np.ndarray,
            lr: float = 0.01, max_iter: int = 100) -> float:
        """Optimize temperature via NLL minimization on validation logits."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.long)
        temperature = torch.tensor([self.temperature], dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = logits_t / temperature.clamp(min=0.01)
            loss = F.cross_entropy(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = max(0.01, temperature.item())
        self._fitted = True
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling: logits -> calibrated probabilities."""
        t = self.temperature if self._fitted else 1.5
        scaled = torch.tensor(logits, dtype=torch.float32) / t
        return F.softmax(scaled, dim=-1).numpy()


# =====================================================================
# SAFE FK SCORE (must match preprocess exactly)
# =====================================================================

def safe_fk_score(text: str) -> float:
    """Compute Flesch-Kincaid grade with NaN/Inf protection."""
    import textstat
    if not isinstance(text, str) or len(text.strip()) == 0:
        return FK_DEFAULT
    try:
        fk = textstat.flesch_kincaid_grade(text[:FK_MAX_CHARS])
    except Exception:
        return FK_DEFAULT
    if math.isnan(fk) or math.isinf(fk):
        return FK_DEFAULT
    return round(max(FK_CLAMP_MIN, min(FK_CLAMP_MAX, fk)), 1)


# =====================================================================
# TWO-STAGE CLASSIFICATION
# =====================================================================

def two_stage_classify(probs: np.ndarray) -> dict:
    """Stage 1: Human vs AI (binary). Stage 2: Which AI model."""
    p_human = float(probs[HUMAN_IDX])
    p_ai_total = 1.0 - p_human

    ai_probs = {}
    for i, label in enumerate(LABEL_ORDER):
        if i != HUMAN_IDX:
            ai_probs[label] = float(probs[i])

    top_ai_model = max(ai_probs, key=ai_probs.get) if ai_probs else None
    top_ai_prob = ai_probs[top_ai_model] if top_ai_model else 0.0

    return {
        "p_human": p_human,
        "p_ai_total": p_ai_total,
        "top_ai_model": top_ai_model,
        "top_ai_confidence": top_ai_prob,
        "ai_breakdown": ai_probs,
    }


# =====================================================================
# SENTENCE-LEVEL AGGREGATION
# =====================================================================

def aggregate_sentences(sentence_results: List[dict],
                        min_sentences: int = 3) -> dict:
    """Aggregate sentence-level predictions to document-level verdict."""
    if len(sentence_results) < min_sentences:
        return {
            "ai_sentence_ratio": 0.0,
            "human_sentence_ratio": 1.0,
            "inconclusive_ratio": 0.0,
            "dominant_ai_model": None,
            "model_distribution": {},
            "confidence": "insufficient_data",
            "n_sentences": len(sentence_results),
            "n_ai": 0, "n_human": 0, "n_inconclusive": 0,
        }

    n = len(sentence_results)
    n_ai = 0
    n_human = 0
    n_inconclusive = 0
    model_votes: Dict[str, int] = {}

    for r in sentence_results:
        v = r["verdict"]
        if v == "AI_DETECTED":
            n_ai += 1
            model = r.get("ai_model", "unknown")
            model_votes[model] = model_votes.get(model, 0) + 1
        elif v == "HUMAN":
            n_human += 1
        else:
            n_inconclusive += 1

    ai_ratio = n_ai / n
    dominant_model = max(model_votes, key=model_votes.get) if model_votes else None
    model_dist = {}
    if model_votes:
        total_ai = sum(model_votes.values())
        model_dist = {k: round(v / total_ai, 3)
                      for k, v in sorted(model_votes.items(), key=lambda x: -x[1])}

    if ai_ratio > 0.85 or ai_ratio < 0.10:
        confidence = "high"
    elif ai_ratio > 0.70 or ai_ratio < 0.20:
        confidence = "moderate"
    else:
        confidence = "low"

    return {
        "ai_sentence_ratio": round(ai_ratio, 3),
        "human_sentence_ratio": round(n_human / n, 3),
        "inconclusive_ratio": round(n_inconclusive / n, 3),
        "dominant_ai_model": dominant_model,
        "model_distribution": model_dist,
        "confidence": confidence,
        "n_sentences": n,
        "n_ai": n_ai, "n_human": n_human, "n_inconclusive": n_inconclusive,
    }


# =====================================================================
# CALIBRATED DETECTOR (main class)
# =====================================================================

class CalibratedDetector:
    """Production detector with 5-layer false positive control.

    Models are lazy-loaded on first classify() call.
    """

    def __init__(self, ensemble_dir: str, fp_tolerance: str = "moderate",
                 device: str = "auto", max_length: int = 512):
        if fp_tolerance not in THRESHOLD_PROFILES:
            raise ValueError(f"fp_tolerance must be one of {list(THRESHOLD_PROFILES.keys())}")

        self._ensemble_dir = ensemble_dir
        self._device_str = device
        self._max_length = max_length
        self.fp_tolerance = fp_tolerance
        self.thresholds = THRESHOLD_PROFILES[fp_tolerance]

        self._models: Optional[Dict] = None
        self._tokenizers: Optional[Dict] = None
        self._torch_device: Optional[torch.device] = None
        self._loaded = False

        self._scalers: Dict[str, TemperatureScaler] = {}
        self._load_calibration()

    def _load_calibration(self) -> None:
        """Load temperature calibration if calibration.json exists."""
        cal_path = os.path.join(self._ensemble_dir, "calibration.json")
        if os.path.exists(cal_path):
            try:
                with open(cal_path) as f:
                    temps = json.load(f)
                for key, t_val in temps.items():
                    scaler = TemperatureScaler(init_temperature=t_val)
                    scaler._fitted = True
                    self._scalers[key] = scaler
                logger.info(f"Calibration loaded: {temps}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load calibration: {e}")

    def _ensure_loaded(self) -> None:
        """Lazy-load models on first inference call."""
        if self._loaded:
            return

        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if self._device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self._device_str)

        self._torch_device = device
        self._models = {}
        self._tokenizers = {}
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

        for key in TRAINING_ORDER:
            model_dir = os.path.join(self._ensemble_dir, f"{key}_final")
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model not found: {model_dir}")
            print(f"Loading {key} from {model_dir}...")
            self._tokenizers[key] = AutoTokenizer.from_pretrained(
                model_dir, model_max_length=self._max_length)
            self._models[key] = AutoModelForSequenceClassification.from_pretrained(
                model_dir, torch_dtype=dtype).to(device).eval()

        self._loaded = True
        print(f"Ensemble loaded: {list(self._models.keys())} on {device}")

    # ----- RAW PREDICTION -----

    @torch.no_grad()
    def _predict_raw(self, text: str) -> dict:
        """Get raw predictions from all 3 models."""
        self._ensure_loaded()
        fk = safe_fk_score(text)
        input_text = f"[FK_SCORE: {fk:.1f}] {text}"

        all_probs = []
        individual = {}

        for key in TRAINING_ORDER:
            tok = self._tokenizers[key](
                input_text, truncation=True, max_length=self._max_length,
                padding=True, return_tensors="pt",
            ).to(self._torch_device)

            logits = self._models[key](**tok).logits.cpu().numpy()

            if key in self._scalers:
                probs = self._scalers[key].calibrate(logits)[0]
            else:
                probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()[0]

            all_probs.append(probs)
            pred_idx = int(np.argmax(probs))
            individual[key] = {
                "label": LABEL_ORDER[pred_idx],
                "confidence": round(float(probs[pred_idx]), 4),
            }

        ensemble_probs = np.mean(all_probs, axis=0)
        return {"ensemble_probs": ensemble_probs, "individual": individual}

    # ----- SINGLE TEXT -----

    def classify(self, text: str) -> dict:
        """Classify a single text with full false positive protection.

        Decision flow:
          1. Ensemble probabilities (temp-scaled if calibrated)
          2. Two-stage: P(Human) vs P(AI)
          3. Inconclusive band check
          4. AI confidence threshold
          5. Agreement gating
          6. Margin check
          7. Verdict: HUMAN | AI_DETECTED | INCONCLUSIVE
        """
        raw = self._predict_raw(text)
        probs = raw["ensemble_probs"]
        individual = raw["individual"]
        th = self.thresholds

        stage = two_stage_classify(probs)
        p_human = stage["p_human"]
        p_ai = stage["p_ai_total"]
        top_ai = stage["top_ai_model"]

        # Agreement: how many models agree on top AI model?
        model_agreement = sum(
            1 for m in individual.values() if m["label"] == top_ai
        ) / len(individual)

        # How many models say "not Human"?
        ai_vote_ratio = sum(
            1 for m in individual.values() if m["label"] != "Human"
        ) / len(individual)

        flags: List[str] = []
        verdict = "HUMAN"
        safe_for_action = False

        # Gate 1: Inconclusive band
        low, high = th.inconclusive_band
        if low <= p_human <= high:
            verdict = "INCONCLUSIVE"
            flags.append(f"P(Human)={p_human:.3f} in band [{low:.2f},{high:.2f}]")

        # Gate 2: AI confidence
        elif p_ai >= th.ai_confidence_min:
            # Gate 3: Agreement
            if ai_vote_ratio >= th.agreement_min:
                # Gate 4: Margin
                sorted_probs = sorted(probs, reverse=True)
                margin = sorted_probs[0] - sorted_probs[1]

                if margin >= th.margin_min:
                    verdict = "AI_DETECTED"
                    if (self.fp_tolerance == "strict"
                            and model_agreement == 1.0
                            and p_ai > 0.85):
                        safe_for_action = True
                else:
                    verdict = "INCONCLUSIVE"
                    flags.append(f"Margin {margin:.3f} < {th.margin_min}")
            else:
                verdict = "INCONCLUSIVE"
                flags.append(f"AI votes {ai_vote_ratio:.0%} < {th.agreement_min:.0%}")

        result = {
            "verdict": verdict,
            "confidence": round(float(np.max(probs)), 4),
            "p_human": round(p_human, 4),
            "p_ai_total": round(p_ai, 4),
            "agreement": round(model_agreement, 2),
            "ai_vote_ratio": round(ai_vote_ratio, 2),
            "flags": flags,
            "safe_for_action": safe_for_action,
            "fp_tolerance": self.fp_tolerance,
            "probabilities": {
                LABEL_ORDER[i]: round(float(probs[i]), 4) for i in range(len(LABEL_ORDER))
            },
            "individual": individual,
        }

        if verdict == "AI_DETECTED":
            result["ai_model"] = top_ai
            result["ai_model_confidence"] = round(stage["top_ai_confidence"], 4)
            result["ai_breakdown"] = {k: round(v, 4) for k, v in stage["ai_breakdown"].items()}

        return result

    # ----- DOCUMENT LEVEL -----

    def classify_document(self, text: str,
                          min_sentence_len: int = 20,
                          min_sentences: int = 3) -> dict:
        """Classify a full document via sentence-level aggregation.

        Splits into sentences, classifies each, aggregates.
        Detects partially AI-written documents.

        Verdict thresholds:
          >70% AI sentences = AI_DETECTED
          30-70%            = MIXED_CONTENT
          <30%              = HUMAN
        """
        try:
            import nltk
            nltk.download('punkt_tab', quiet=True)
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        valid = [s for s in sentences if len(s.strip()) >= min_sentence_len]

        if not valid:
            return {
                "verdict": "INSUFFICIENT_TEXT",
                "n_sentences": 0,
                "detail": "No sentences with sufficient length.",
            }

        sentence_results = [self.classify(s) for s in valid]
        agg = aggregate_sentences(sentence_results, min_sentences=min_sentences)

        if agg["confidence"] == "insufficient_data":
            doc_verdict = "INSUFFICIENT_TEXT"
        elif agg["ai_sentence_ratio"] > 0.70:
            doc_verdict = "AI_DETECTED"
        elif agg["ai_sentence_ratio"] > 0.30:
            doc_verdict = "MIXED_CONTENT"
        else:
            doc_verdict = "HUMAN"

        doc_safe = (
            doc_verdict == "AI_DETECTED"
            and self.fp_tolerance == "strict"
            and agg["confidence"] == "high"
            and agg["ai_sentence_ratio"] > 0.85
        )

        return {
            "verdict": doc_verdict,
            "safe_for_action": doc_safe,
            "fp_tolerance": self.fp_tolerance,
            "aggregation": agg,
            "sentence_verdicts": [
                {
                    "text": s[:120] + "..." if len(s) > 120 else s,
                    "verdict": r["verdict"],
                    "p_human": r["p_human"],
                    "p_ai": r["p_ai_total"],
                    "ai_model": r.get("ai_model"),
                    "confidence": r["confidence"],
                }
                for s, r in zip(valid, sentence_results)
            ],
        }

    # ----- BATCH -----

    @torch.no_grad()
    def classify_batch(self, texts: List[str], batch_size: int = 32) -> List[dict]:
        """Classify multiple texts efficiently with batched inference."""
        self._ensure_loaded()
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            processed = [f"[FK_SCORE: {safe_fk_score(t):.1f}] {t}" for t in batch]

            all_probs = []
            all_individual = [{} for _ in batch]

            for key in TRAINING_ORDER:
                tok = self._tokenizers[key](
                    processed, truncation=True, max_length=self._max_length,
                    padding=True, return_tensors="pt",
                ).to(self._torch_device)

                logits = self._models[key](**tok).logits.cpu().numpy()
                if key in self._scalers:
                    probs = self._scalers[key].calibrate(logits)
                else:
                    probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

                all_probs.append(probs)
                for j in range(len(batch)):
                    pred_idx = int(np.argmax(probs[j]))
                    all_individual[j][key] = {
                        "label": LABEL_ORDER[pred_idx],
                        "confidence": round(float(probs[j][pred_idx]), 4),
                    }

            ensemble_probs = np.mean(all_probs, axis=0)
            th = self.thresholds

            for j in range(len(batch)):
                pj = ensemble_probs[j]
                ind = all_individual[j]
                stage = two_stage_classify(pj)
                p_human = stage["p_human"]
                p_ai = stage["p_ai_total"]
                top_ai = stage["top_ai_model"]

                model_agr = sum(1 for m in ind.values() if m["label"] == top_ai) / len(ind)
                ai_votes = sum(1 for m in ind.values() if m["label"] != "Human") / len(ind)

                flags = []
                verdict = "HUMAN"
                safe = False

                low, high = th.inconclusive_band
                if low <= p_human <= high:
                    verdict = "INCONCLUSIVE"
                elif p_ai >= th.ai_confidence_min:
                    if ai_votes >= th.agreement_min:
                        sp = sorted(pj, reverse=True)
                        margin = sp[0] - sp[1]
                        if margin >= th.margin_min:
                            verdict = "AI_DETECTED"
                            if self.fp_tolerance == "strict" and model_agr == 1.0 and p_ai > 0.85:
                                safe = True
                        else:
                            verdict = "INCONCLUSIVE"
                    else:
                        verdict = "INCONCLUSIVE"

                r = {
                    "verdict": verdict,
                    "p_human": round(p_human, 4),
                    "p_ai_total": round(p_ai, 4),
                    "agreement": round(model_agr, 2),
                    "safe_for_action": safe,
                }
                if verdict == "AI_DETECTED":
                    r["ai_model"] = top_ai
                results.append(r)

        return results


# =====================================================================
# CALIBRATION UTILITY — run once after training
# =====================================================================

def calibrate_from_saved_logits(ensemble_dir: str) -> dict:
    """Load saved logits from Phase 4/5 and fit temperature scaling.

    Run ONCE after training. Saves calibration.json to ensemble_dir.

    Usage:
        from ensemble_v6_calibrated import calibrate_from_saved_logits
        calibrate_from_saved_logits("/path/to/ensemble_v6/")
    """
    print("=" * 60)
    print("TEMPERATURE CALIBRATION")
    print("=" * 60)

    logits_per_model = {}
    labels = None

    for key in TRAINING_ORDER:
        lp = os.path.join(ensemble_dir, f"{key}_logits.npy")
        labp = os.path.join(ensemble_dir, f"{key}_labels.npy")
        if not os.path.exists(lp):
            print(f"  {lp} not found. Run training Phase 4 first.")
            return {}
        logits_per_model[key] = np.load(lp)
        if labels is None:
            labels = np.load(labp)
        print(f"  [{key}] Loaded {logits_per_model[key].shape}")

    temperatures = {}
    calibrated_probs = []

    print("\nFitting per-model temperatures...")
    for key in TRAINING_ORDER:
        scaler = TemperatureScaler()
        t = scaler.fit(logits_per_model[key], labels)
        temperatures[key] = round(t, 4)
        calibrated_probs.append(scaler.calibrate(logits_per_model[key]))
        print(f"  [{key}] T = {t:.4f}")

    print("\nFitting ensemble temperature...")
    ens_probs = np.mean(calibrated_probs, axis=0)
    ens_logits = np.log(ens_probs + 1e-10)
    ens_scaler = TemperatureScaler()
    t_ens = ens_scaler.fit(ens_logits, labels)
    temperatures["ensemble"] = round(t_ens, 4)
    print(f"  [ensemble] T = {t_ens:.4f}")

    cal_path = os.path.join(ensemble_dir, "calibration.json")
    with open(cal_path, "w") as f:
        json.dump(temperatures, f, indent=2)

    print(f"\nCalibration saved to {cal_path}")
    print(f"CalibratedDetector will load this automatically on next init.")
    return temperatures
