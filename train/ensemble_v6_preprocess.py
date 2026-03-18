"""
ENSEMBLE AI Text Detector v6.1 — PREPROCESSING ONLY (CPU Runtime)
==================================================================
Ejecutar este script en un runtime CPU (sin GPU) en Colab Pro+.
Costo: 0 CU — toda la carga es CPU/IO.

Hace exactamente:
  Phase 0  — Montar Drive + copiar parquets a SSD local (con SHA256)
  Phase 1  — Limpiar dataset + train/test split
  Phase 2a — Pre-computar FK scores (shared, one-time, NaN-safe)
  Phase 2b — Tokenizar los 3 modelos (TRAIN con augmentation, TEST sin augmentation)

v6.1 FIXES over v6.0:
  [CRITICAL] Augmentation NO longer applied to test set (was contaminating eval)
  [CRITICAL] FK NaN/Inf validation added (clamp + isnan guard)
  [CRITICAL] Bare except replaced with except OSError + max 5 retries
  [MAJOR]    SHA256 checksum on parquet copies (detects silent corruption)
  [MAJOR]    Config imported from shared ensemble_v6_config.py
  [MAJOR]    Global random seed for reproducibility
  [MODERATE] Bare except in keep-alive → except Exception
  [MODERATE] Magic numbers extracted to named constants

INSTRUCCIONES:
  1. Abre un notebook Colab → Runtime → Change runtime type → CPU
  2. Sube ensemble_v6_config.py + este script
  3. !pip install datasets transformers textstat nltk -q
  4. !python ensemble_v6_preprocess.py
  5. Espera ~5–6h (paralelizado con todos los cores disponibles)
  6. Verifica: "✅ PREPROCESSING COMPLETE — Ready for GPU training!"
  7. Cambia runtime a H100 y ejecuta ensemble_v6_train.py
"""

import os
import gc
import glob
import time
import math
import random
import hashlib
import logging
import multiprocessing

import torch
import numpy as np
import textstat
import nltk
from tqdm.auto import tqdm
from google.colab import drive
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

# ── Shared config (single source of truth) ──
#from ensemble_v6_config import (
#    LABEL_ORDER, LABEL2ID, ID2LABEL, LABEL_MAP,
#    MODEL_CONFIGS, TRAINING_ORDER,
#    DRIVE_BASE, DRIVE_DATASET_DIR, LOCAL_DATASET_DIR,
#    CLEAN_DATASET_DIR, SPLIT_CACHE_DIR, ENSEMBLE_DIR,
#    FK_CACHE_DIR, TOKENIZED_BASE,
#    model_paths,
#    AUGMENT_SKIP_PROB, AUGMENT_CHAR_PROB, AUGMENT_MERGE_PROB,
#    MIN_WORD_LEN_AUGMENT,
#    FK_MAX_CHARS, FK_CLAMP_MIN, FK_CLAMP_MAX, FK_DEFAULT,
#    MAX_COPY_RETRIES, COPY_RETRY_BASE_SEC,
#)

nltk.download('cmudict', quiet=True)
nltk.download('punkt', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# REPRODUCIBILITY — global seed
# ═══════════════════════════════════════════════════════════════

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

# ═══════════════════════════════════════════════════════════════
# SAFETY CHECK — este script NO debe correr con GPU asignada
# ═══════════════════════════════════════════════════════════════

if torch.cuda.is_available():
    print("⚠️  ADVERTENCIA: Se detectó una GPU en este runtime.")
    print("    Este script está diseñado para correr en runtime CPU (0 CU).")
    print("    Continúa si realmente quieres, pero estarás gastando CU.")
    print()

NUM_CPU = max(1, multiprocessing.cpu_count())
print(f"🖥️  CPU cores disponibles: {NUM_CPU}")

# ═══════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════

def flush_ram(label: str = "") -> None:
    """Force garbage collection and report RAM usage."""
    gc.collect()
    gc.collect()
    time.sleep(0.3)
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"🧹 RAM [{label}]: {ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB ({ram.percent}%)")
    except ImportError:
        print(f"🧹 RAM [{label}]: gc.collect() done")


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def sha256_file(filepath: str) -> str:
    """Compute SHA256 hash of a file in chunks."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_fk_score(text: str) -> float:
    """Compute Flesch-Kincaid grade with NaN/Inf protection."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return FK_DEFAULT
    try:
        fk = textstat.flesch_kincaid_grade(text[:FK_MAX_CHARS])
    except Exception:
        return FK_DEFAULT
    # Guard against NaN, Inf, and extreme outliers
    if math.isnan(fk) or math.isinf(fk):
        return FK_DEFAULT
    return round(max(FK_CLAMP_MIN, min(FK_CLAMP_MAX, fk)), 1)


# Colab keep-alive
try:
    from google.colab import output
    output.eval_js('''
        function ClickConnect(){
            var buttons = document.querySelectorAll("colab-connect-button");
            if(buttons.length > 0){ buttons[0].click(); }
        }
        setInterval(ClickConnect, 60000);
    ''')
    print("✅ Colab keep-alive active")
except Exception:
    # Not in Colab or JS unavailable — benign
    pass


# ═══════════════════════════════════════════════════════════════
# AUGMENTATION
# ═══════════════════════════════════════════════════════════════

class AdvancedAdversarialAugmenter:
    """Character-level adversarial noise for training robustness.

    IMPORTANT: Apply ONLY to training data. Test data must remain clean
    to produce valid evaluation metrics.
    """

    @staticmethod
    def apply_noise(text: str, prob: float = AUGMENT_CHAR_PROB) -> str:
        """Apply random character-level perturbations to text."""
        if not isinstance(text, str):
            return ""
        if random.random() > (1.0 - AUGMENT_SKIP_PROB):
            return text

        words = text.split()
        for i in range(len(words)):
            if random.random() < prob:
                word = words[i]
                if len(word) > MIN_WORD_LEN_AUGMENT - 1:
                    attack = random.choice(["swap", "delete", "insert"])
                    chars = list(word)
                    idx = random.randint(0, len(chars) - 2)
                    if attack == "swap":
                        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
                    elif attack == "delete" and len(chars) > 3:
                        chars.pop(idx)
                    elif attack == "insert":
                        chars.insert(idx, random.choice("abcdefghijklmnopqrstuvwxyz"))
                    words[i] = "".join(chars)
            if random.random() < AUGMENT_MERGE_PROB and i < len(words) - 1:
                words[i] = words[i] + words[i + 1]
                words[i + 1] = ""

        return " ".join([w for w in words if w])


# ═══════════════════════════════════════════════════════════════
# PHASE 0: MONTAR DRIVE + COPIAR PARQUETS (con SHA256)
# ═══════════════════════════════════════════════════════════════

def phase_0_setup() -> None:
    """Mount Drive and copy parquets to local SSD with integrity checks."""
    print("\n" + "=" * 70)
    print("PHASE 0: MOUNT DRIVE + DOWNLOAD DATA (SHA256 verified)")
    print("=" * 70)

    drive.mount('/content/drive')
    ensure_dirs(
        LOCAL_DATASET_DIR, ENSEMBLE_DIR,
        CLEAN_DATASET_DIR, SPLIT_CACHE_DIR,
        FK_CACHE_DIR, TOKENIZED_BASE,
    )

    archivos_drive = sorted(glob.glob(os.path.join(DRIVE_DATASET_DIR, "*.parquet")))
    print(f"📦 Files in Drive: {len(archivos_drive)}")

    if len(archivos_drive) == 0:
        raise FileNotFoundError(
            f"❌ No parquet files found in {DRIVE_DATASET_DIR}\n"
            "   Verifica que el dataset esté en la ruta correcta en Drive."
        )

    copied = 0
    skipped = 0
    for archivo_drive in tqdm(archivos_drive, desc="Downloading to SSD"):
        nombre = os.path.basename(archivo_drive)
        local = os.path.join(LOCAL_DATASET_DIR, nombre)

        # Skip if already copied and size matches
        if os.path.exists(local) and os.path.getsize(local) == os.path.getsize(archivo_drive):
            skipped += 1
            continue

        # Copy with retry + exponential backoff + SHA256 verification
        success = False
        for attempt in range(1, MAX_COPY_RETRIES + 1):
            try:
                with open(archivo_drive, 'rb') as fsrc, open(local, 'wb') as fdst:
                    while True:
                        chunk = fsrc.read(10 * 1024 * 1024)
                        if not chunk:
                            break
                        fdst.write(chunk)

                # Verify integrity
                src_hash = sha256_file(archivo_drive)
                dst_hash = sha256_file(local)
                if src_hash != dst_hash:
                    raise IOError(
                        f"SHA256 mismatch for {nombre}: "
                        f"src={src_hash[:16]}... dst={dst_hash[:16]}..."
                    )

                copied += 1
                success = True
                break

            except (OSError, IOError) as e:
                logger.warning(f"Copy attempt {attempt}/{MAX_COPY_RETRIES} failed for {nombre}: {e}")
                if os.path.exists(local):
                    os.remove(local)
                if attempt < MAX_COPY_RETRIES:
                    wait = COPY_RETRY_BASE_SEC * (2 ** (attempt - 1))
                    logger.info(f"Retrying in {wait}s...")
                    time.sleep(wait)
                    try:
                        drive.mount('/content/drive', force_remount=True)
                    except Exception as mount_err:
                        logger.warning(f"Drive remount failed: {mount_err}")
                else:
                    raise RuntimeError(
                        f"❌ Failed to copy {nombre} after {MAX_COPY_RETRIES} attempts. "
                        "Check Drive connectivity and disk space."
                    ) from e

    print(f"📊 Copied: {copied} | Skipped (cached): {skipped} | Total: {len(archivos_drive)}")
    flush_ram("post-download")


# ═══════════════════════════════════════════════════════════════
# PHASE 1: LIMPIAR DATASET + SPLIT
# ═══════════════════════════════════════════════════════════════

def phase_1_clean_and_split() -> tuple:
    """Clean raw dataset, unify labels, and create train/test split."""
    print("\n" + "=" * 70)
    print("PHASE 1: CLEAN DATASET + TRAIN/TEST SPLIT")
    print("=" * 70)

    train_path = os.path.join(SPLIT_CACHE_DIR, "train")
    test_path = os.path.join(SPLIT_CACHE_DIR, "test")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("✅ Splits already exist, skipping Phase 1.")
        return train_path, test_path

    clean_flag = os.path.join(CLEAN_DATASET_DIR, "dataset_info.json")
    if os.path.exists(clean_flag):
        print("✅ Clean cache exists.")
    else:
        print("📥 Loading raw dataset...")
        dataset = load_dataset("parquet", data_dir=LOCAL_DATASET_DIR, split="train")

        def unify_labels(example):
            modelo = str(example.get('target_model', "Unknown"))
            for k, v in LABEL_MAP.items():
                if k.lower() in modelo.lower():
                    example['final_model'] = v
                    return example
            example['final_model'] = "Unknown AI"
            return example

        dataset = dataset.map(
            unify_labels, keep_in_memory=False,
            num_proc=NUM_CPU, desc="Unify labels"
        )
        initial = len(dataset)
        dataset = dataset.filter(
            lambda x: isinstance(x.get('text'), str) and len(str(x.get('text', '')).strip()) > 0,
            num_proc=NUM_CPU, desc="Filter empty"
        )
        removed = initial - len(dataset)
        if removed > 0:
            print(f"⚠️  Removed {removed} empty rows from {initial}")
        dataset.save_to_disk(CLEAN_DATASET_DIR)
        del dataset
        flush_ram("post-clean")

    print("📂 Loading clean dataset for splitting...")
    dataset = load_from_disk(CLEAN_DATASET_DIR, keep_in_memory=False)
    print(f"📊 {len(dataset)} rows total")

    splits = dataset.train_test_split(test_size=0.10, seed=GLOBAL_SEED)
    splits["train"].save_to_disk(train_path)
    splits["test"].save_to_disk(test_path)
    print(f"✅ Train: {len(splits['train'])} | Test: {len(splits['test'])}")
    del dataset, splits
    flush_ram("post-split")

    return train_path, test_path


# ═══════════════════════════════════════════════════════════════
# PHASE 2a: PRE-COMPUTAR FK SCORES (NaN-safe, shared, one-time)
#
# Mejora clave:
#   - FK calculado 1 sola vez para los 3 modelos (~4M llamadas vs ~12M)
#   - NaN/Inf protection via safe_fk_score()
#   - Clamped to [-5.0, 30.0] range
# ═══════════════════════════════════════════════════════════════

def phase_2a_precompute_fk(train_path: str, test_path: str) -> tuple:
    """Pre-compute Flesch-Kincaid scores with NaN protection."""
    print("\n" + "=" * 70)
    print("PHASE 2a: PRE-COMPUTE FK SCORES (NaN-safe, shared, one-time)")
    print("=" * 70)

    fk_train_path = os.path.join(FK_CACHE_DIR, "fk_train")
    fk_test_path = os.path.join(FK_CACHE_DIR, "fk_test")

    if (os.path.exists(os.path.join(fk_train_path, "dataset_info.json")) and
            os.path.exists(os.path.join(fk_test_path, "dataset_info.json"))):
        print("✅ FK cache exists, skipping.")
        return fk_train_path, fk_test_path

    def add_fk(examples):
        scores = []
        for t in examples['text']:
            scores.append(safe_fk_score(str(t) if t else ""))
        examples['fk_score'] = scores
        return examples

    for split_name, split_path, out_path in [
        ("train", train_path, fk_train_path),
        ("test",  test_path,  fk_test_path),
    ]:
        if os.path.exists(os.path.join(out_path, "dataset_info.json")):
            print(f"✅ FK {split_name} cache exists, skipping.")
            continue
        print(f"⏳ Computing FK for {split_name}...")
        ds = load_from_disk(split_path, keep_in_memory=False)
        ds = ds.map(
            add_fk,
            batched=True, batch_size=2000,
            keep_in_memory=False,
            num_proc=NUM_CPU,
            desc=f"FK {split_name}",
        )
        ds.save_to_disk(out_path)
        del ds
        flush_ram(f"post-fk-{split_name}")

    return fk_train_path, fk_test_path


# ═══════════════════════════════════════════════════════════════
# PHASE 2b: TOKENIZAR POR MODELO
#   CRITICAL FIX: Train gets augmentation, Test does NOT.
# ═══════════════════════════════════════════════════════════════

def phase_2b_tokenize(model_key: str, fk_train_path: str, fk_test_path: str) -> tuple:
    """Tokenize train (with augmentation) and test (clean) for a model."""
    cfg = MODEL_CONFIGS[model_key]
    paths = model_paths(model_key)
    ensure_dirs(paths["tokenized_train"], paths["tokenized_test"])

    print(f"\n{'='*70}")
    print(f"PHASE 2b: TOKENIZE for {cfg['name']}")
    print(f"{'='*70}")

    train_tok_flag = os.path.join(paths["tokenized_train"], "dataset_info.json")
    test_tok_flag = os.path.join(paths["tokenized_test"], "dataset_info.json")

    if os.path.exists(train_tok_flag) and os.path.exists(test_tok_flag):
        print(f"✅ [{model_key}] Tokenized cache exists, skipping.")
        return paths["tokenized_train"], paths["tokenized_test"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["name"], model_max_length=cfg["max_length"])

    # ── TRAIN preprocessing: FK prefix + adversarial augmentation ──
    def preprocess_fn_train(examples):
        textos = []
        for t, fk in zip(examples['text'], examples['fk_score']):
            txt = str(t) if t else ""
            noisy = AdvancedAdversarialAugmenter.apply_noise(txt)
            textos.append(f"[FK_SCORE: {fk:.1f}] {noisy}")
        tok = tokenizer(textos, truncation=True, max_length=cfg["max_length"], padding=False)
        tok['labels'] = [LABEL2ID[m] for m in examples['final_model']]
        return tok

    # ── TEST preprocessing: FK prefix ONLY, NO augmentation ──
    def preprocess_fn_test(examples):
        textos = []
        for t, fk in zip(examples['text'], examples['fk_score']):
            txt = str(t) if t else ""
            textos.append(f"[FK_SCORE: {fk:.1f}] {txt}")
        tok = tokenizer(textos, truncation=True, max_length=cfg["max_length"], padding=False)
        tok['labels'] = [LABEL2ID[m] for m in examples['final_model']]
        return tok

    # Process each split with the correct function
    splits_config = [
        ("train", fk_train_path, paths["tokenized_train"], preprocess_fn_train),
        ("test",  fk_test_path,  paths["tokenized_test"],  preprocess_fn_test),
    ]

    for split_name, fk_path, out_path, preprocess_fn in splits_config:
        flag = os.path.join(out_path, "dataset_info.json")
        if os.path.exists(flag):
            print(f"✅ [{model_key}] {split_name} already tokenized, skipping.")
            continue

        augment_status = "WITH augmentation" if split_name == "train" else "NO augmentation (clean)"
        print(f"⏳ [{model_key}] Tokenizing {split_name} ({augment_status})...")
        ds = load_from_disk(fk_path, keep_in_memory=False)

        # Attempt parallel tokenization; fallback to serial on pickling issues
        try:
            tok_ds = ds.map(
                preprocess_fn,
                batched=True, batch_size=1000,
                remove_columns=ds.column_names,
                keep_in_memory=False,
                num_proc=NUM_CPU,
                desc=f"Tokenize {split_name} [{model_key}]",
            )
        except Exception as e:
            # DeBERTa's SentencePiece tokenizer can't be pickled for multiprocessing.
            # RuntimeError and AttributeError are the usual symptoms.
            if any(kw in str(e).lower() for kw in ["pickle", "can't pickle", "fork", "serializ"]):
                print(f"⚠️  Parallel tokenization failed ({type(e).__name__}: {e})")
                print(f"    Falling back to num_proc=1 (serial)...")
                tok_ds = ds.map(
                    preprocess_fn,
                    batched=True, batch_size=1000,
                    remove_columns=ds.column_names,
                    keep_in_memory=False,
                    num_proc=1,
                    desc=f"Tokenize {split_name} [{model_key}] (serial)",
                )
            else:
                raise

        tok_ds.save_to_disk(out_path)
        del ds, tok_ds
        flush_ram(f"{model_key}-{split_name}-tokenized")

    del tokenizer
    flush_ram(f"{model_key}-tokenizer-freed")
    return paths["tokenized_train"], paths["tokenized_test"]


# ═══════════════════════════════════════════════════════════════
# VERIFICACIÓN FINAL
# ═══════════════════════════════════════════════════════════════

def verify_readiness() -> bool:
    """Verify all tokenized datasets exist before GPU training."""
    print("\n" + "=" * 70)
    print("READINESS CHECK — verifying all artifacts for GPU training")
    print("=" * 70)

    all_ok = True
    for model_key in TRAINING_ORDER:
        paths = model_paths(model_key)
        for split, path in [("train", paths["tokenized_train"]), ("test", paths["tokenized_test"])]:
            flag = os.path.join(path, "dataset_info.json")
            exists = os.path.exists(flag)
            status = "✅" if exists else "❌"
            if not exists:
                all_ok = False
            print(f"  {status}  {model_key} {split}: {path}")

    if all_ok:
        print("\n🎉 ✅ PREPROCESSING COMPLETE — Ready for GPU training!")
        print("    Siguiente paso:")
        print("    1. Cambia el runtime a H100 (Runtime → Change runtime type)")
        print("    2. Ejecuta: !python ensemble_v6_train.py")
    else:
        print("\n❌ Some tokenized datasets are missing. Re-run this script.")

    return all_ok


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 70)
    print("ENSEMBLE v6.1 — PREPROCESSING (CPU Runtime, 0 CU)")
    print("Fixes: test augmentation, FK NaN, SHA256, retry limits")
    print("=" * 70)

    # Phase 0
    phase_0_setup()

    # Phase 1
    train_path, test_path = phase_1_clean_and_split()

    # Phase 2a — FK pre-computado (NaN-safe)
    fk_train_path, fk_test_path = phase_2a_precompute_fk(train_path, test_path)

    # Phase 2b — Tokenizar: TRAIN con augmentation, TEST sin augmentation
    for model_key in TRAINING_ORDER:
        print(f"\n{'#'*70}")
        print(f"# TOKENIZING: {MODEL_CONFIGS[model_key]['name']}")
        print(f"{'#'*70}")
        phase_2b_tokenize(model_key, fk_train_path, fk_test_path)
        flush_ram(f"between-tokenize-{model_key}")

    # Verificación
    verify_readiness()
