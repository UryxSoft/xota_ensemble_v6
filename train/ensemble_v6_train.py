"""
ENSEMBLE AI Text Detector v6.1 — TRAINING ONLY (H100 / A100 Runtime)
=====================================================================
Ejecutar DESPUÉS de ensemble_v6_preprocess.py.
Solo consume CU durante fases que realmente necesitan GPU.

Requisito previo:
  ensemble_v6_preprocess.py debe haber completado exitosamente.

Hace exactamente:
  Phase 0  — Montar Drive + verificar datos tokenizados
  Phase 3  — Entrenar DeBERTa / ELECTRA / ModernBERT secuencialmente
  Phase 4  — Evaluar cada modelo individualmente
  Phase 5  — Ensemble soft voting + confusion matrix

v6.1 FIXES over v6.0:
  [CRITICAL] SIGTERM handler: bare except → except Exception with logging
  [MAJOR]    Config imported from shared ensemble_v6_config.py
  [MAJOR]    class_weights device moved to compute_loss (not __init__)
  [MAJOR]    evaluate.load() cached at module level (no redundant downloads)
  [MODERATE] wandb optional via USE_WANDB flag
  [MODERATE] dataloader_num_workers=2 for GPU overlap
  [MODERATE] Bare except in callbacks → except Exception
  [MODERATE] Fixed final print to reference correct import path
  [MINOR]    Type hints on all public functions

INSTRUCCIONES:
  1. Confirma "✅ PREPROCESSING COMPLETE" del preprocess script
  2. Runtime → Change runtime type → GPU → H100
  3. !pip install datasets transformers evaluate wandb scikit-learn matplotlib seaborn textstat nltk -q
  4. !python ensemble_v6_train.py
  5. Si la sesión cae, re-ejecutar. Retoma automáticamente.
"""

import os
import gc
import sys
import time
import json
import signal
import shutil
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import evaluate
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Colab stability
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import confusion_matrix, classification_report

import nltk
nltk.download('punkt', quiet=True)

# ── Shared config (single source of truth) ──
#from ensemble_v6_config import (
#    LABEL_ORDER, LABEL2ID, ID2LABEL, NUM_LABELS,
#    CLASS_WEIGHTS, CLASS_WEIGHTS_DICT,
#    MODEL_CONFIGS, TRAINING_ORDER,
#    DRIVE_BASE, ENSEMBLE_DIR, LOCAL_BASE, TOKENIZED_BASE,
#    model_paths,
#)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# WANDB — optional, controlled by env var or flag
# Set USE_WANDB=0 to disable: export USE_WANDB=0
# ═══════════════════════════════════════════════════════════════

USE_WANDB = os.environ.get("USE_WANDB", "1") == "1"
if USE_WANDB:
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed. Disabling W&B logging.")
        USE_WANDB = False
else:
    wandb = None
    logger.info("W&B disabled via USE_WANDB=0")

# ═══════════════════════════════════════════════════════════════
# SAFETY CHECK — este script REQUIERE GPU
# ═══════════════════════════════════════════════════════════════

if not torch.cuda.is_available():
    raise RuntimeError(
        "❌ No GPU detected.\n"
        "   Este script requiere un runtime GPU (H100 o A100).\n"
        "   Ve a Runtime → Change runtime type → GPU."
    )

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"✅ GPU detectada: {gpu_name} ({gpu_mem:.0f} GB)")

# ═══════════════════════════════════════════════════════════════
# PRE-LOAD METRICS (avoid redundant downloads during eval)
# ═══════════════════════════════════════════════════════════════

_accuracy_metric = evaluate.load("accuracy")
_f1_metric = evaluate.load("f1")

# ═══════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════

def flush_ram(label: str = "") -> None:
    """Force garbage collection and report RAM usage."""
    gc.collect()
    gc.collect()
    time.sleep(0.5)
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"🧹 RAM [{label}]: {ram.used/1e9:.1f}/{ram.total/1e9:.1f} GB ({ram.percent}%)")
    except ImportError:
        print(f"🧹 RAM [{label}]: gc.collect() done")


def flush_vram(label: str = "") -> None:
    """Release cached VRAM and report usage."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        alloc = torch.cuda.memory_allocated() / 1e9
        res = torch.cuda.memory_reserved() / 1e9
        print(f"🎮 VRAM [{label}]: {alloc:.1f}GB alloc / {res:.1f}GB reserved")


def ensure_dirs(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


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
    pass


# ═══════════════════════════════════════════════════════════════
# MULTI-LEVEL CONTRASTIVE LOSS (tau=0.3)
# ═══════════════════════════════════════════════════════════════

class MultiLevelContrastiveLoss(nn.Module):
    """Custom multi-level contrastive loss for AI text attribution.

    Three contrastive levels:
      L1: Human vs AI (weight δ=3.0 — strongest boundary)
      L2: Same AI model clusters (weight α=1.0)
      L4: Inter-AI separation (weight γ=1.0)

    Temperature τ=0.3 (smoother gradients than v5.3's τ=0.1).
    """

    def __init__(self, temperature: float = 0.3, alpha: float = 1.0,
                 beta: float = 1.0, gamma: float = 1.0, delta: float = 3.0):
        super().__init__()
        self.tau = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def compute_lq(self, sim_matrix: torch.Tensor,
                   pos_mask: torch.Tensor, neg_mask: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for one level given positive/negative masks."""
        pos_sum = (sim_matrix * pos_mask).sum(dim=1)
        pos_count = pos_mask.sum(dim=1).clamp(min=1e-8)
        pos_term = torch.exp(pos_sum / pos_count)
        neg_term = (torch.exp(sim_matrix) * neg_mask).sum(dim=1)
        l_q = -torch.log(pos_term / (pos_term + neg_term + 1e-8))
        return l_q * (pos_mask.sum(dim=1) > 0)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute multi-level contrastive loss on CLS embeddings."""
        B = features.size(0)
        fn = F.normalize(features, p=2, dim=1)
        sim = torch.matmul(fn, fn.T) / self.tau
        diag = torch.eye(B, dtype=torch.bool, device=features.device)
        is_h = (labels == 0)   # Human = label 0
        is_a = (labels != 0)
        same = (labels.unsqueeze(1) == labels.unsqueeze(0))

        L1 = self.compute_lq(
            sim,
            is_h.unsqueeze(1) & is_h.unsqueeze(0) & ~diag,
            is_h.unsqueeze(1) & is_a.unsqueeze(0),
        )
        L2 = self.compute_lq(
            sim,
            is_a.unsqueeze(1) & is_a.unsqueeze(0) & same & ~diag,
            is_a.unsqueeze(1) & is_a.unsqueeze(0) & ~same,
        )
        L4 = self.compute_lq(
            sim,
            is_a.unsqueeze(1) & is_a.unsqueeze(0) & ~same,
            is_a.unsqueeze(1) & is_h.unsqueeze(0),
        )
        return torch.where(is_h, self.delta * L1, self.alpha * L2 + self.gamma * L4).mean()


class ContrastiveEnsembleTrainer(Trainer):
    """Custom Trainer with MCL + weighted CrossEntropy.

    FIX v6.1: class_weights moved to device inside compute_loss()
    instead of __init__ to avoid device mismatch on multi-GPU.
    """

    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcl_loss_fn = MultiLevelContrastiveLoss(temperature=0.3)
        # Store on CPU; move to correct device in compute_loss
        self._class_weights_cpu = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs, output_hidden_states=True)

        # CLS embedding from last hidden state
        emb = outputs.hidden_states[-1][:, 0, :]

        # MCL loss on embeddings
        mcl_loss = self.mcl_loss_fn(emb, labels)

        # Weighted CE loss on logits — device-safe
        if self._class_weights_cpu is not None:
            weights = self._class_weights_cpu.to(outputs.logits.device)
            ce_loss = F.cross_entropy(outputs.logits, labels, weight=weights)
        else:
            ce_loss = F.cross_entropy(outputs.logits, labels)

        loss = mcl_loss + ce_loss
        if return_outputs:
            outputs.hidden_states = None  # Free memory
            return (loss, outputs)
        return loss


# ═══════════════════════════════════════════════════════════════
# CALLBACK: VRAM flush + checkpoint copy to Drive
# ═══════════════════════════════════════════════════════════════

class StabilityCallback(TrainerCallback):
    """Periodic VRAM flush and checkpoint backup to Drive."""

    def __init__(self, local_ckpt_dir: str, drive_ckpt_dir: str,
                 model_name: str, flush_every: int = 500):
        self.local_ckpt_dir = local_ckpt_dir
        self.drive_ckpt_dir = drive_ckpt_dir
        self.model_name = model_name
        self.flush_every = flush_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.flush_every == 0 and state.global_step > 0:
            torch.cuda.empty_cache()
            gc.collect()

    def on_save(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        gc.collect()
        ckpt_name = f"checkpoint-{state.global_step}"
        local_path = os.path.join(self.local_ckpt_dir, ckpt_name)
        drive_path = os.path.join(self.drive_ckpt_dir, ckpt_name)
        if os.path.exists(local_path):
            try:
                print(f"📤 [{self.model_name}] Copying {ckpt_name} to Drive...")
                if os.path.exists(drive_path):
                    shutil.rmtree(drive_path)
                shutil.copytree(local_path, drive_path)
                print(f"✅ [{self.model_name}] {ckpt_name} saved to Drive")
            except OSError as e:
                # Non-fatal: training continues, Drive backup just failed
                print(f"⚠️  [{self.model_name}] Drive copy failed (non-fatal): {e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % 2000 == 0 and state.global_step > 0:
            try:
                import psutil
                ram = psutil.virtual_memory()
                vram_alloc = torch.cuda.memory_allocated() / 1e9
                loss_val = logs.get('loss', 'N/A') if logs else 'N/A'
                print(f"📊 [{self.model_name}] Step {state.global_step} | "
                      f"RAM: {ram.used/1e9:.1f}/{ram.total/1e9:.1f}GB | "
                      f"VRAM: {vram_alloc:.1f}GB | Loss: {loss_val}")
            except Exception as e:
                logger.debug(f"Monitoring log failed: {e}")


# ═══════════════════════════════════════════════════════════════
# METRICS — uses pre-loaded metric objects
# ═══════════════════════════════════════════════════════════════

def compute_metrics(eval_pred) -> dict:
    """Compute accuracy and weighted F1 from predictions."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": _accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
        "f1_weighted": _f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }


# ═══════════════════════════════════════════════════════════════
# PHASE 0 (GPU) — Mount Drive + verify tokenized data exists
# ═══════════════════════════════════════════════════════════════

def phase_0_mount_and_verify() -> None:
    """Mount Drive and verify all tokenized datasets exist."""
    print("\n" + "=" * 70)
    print("PHASE 0: MOUNT DRIVE + VERIFY TOKENIZED DATA")
    print("=" * 70)

    drive.mount('/content/drive')
    ensure_dirs(ENSEMBLE_DIR, LOCAL_BASE)

    missing = []
    for model_key in TRAINING_ORDER:
        paths = model_paths(model_key)
        for split in ["tokenized_train", "tokenized_test"]:
            flag = os.path.join(paths[split], "dataset_info.json")
            if not os.path.exists(flag):
                missing.append(f"{model_key} / {split}")

    if missing:
        raise FileNotFoundError(
            "❌ Los siguientes datasets tokenizados NO existen:\n"
            + "\n".join(f"   • {m}" for m in missing)
            + "\n\n   Ejecuta ensemble_v6_preprocess.py en un runtime CPU primero."
        )

    print("✅ Todos los datasets tokenizados están presentes.")
    print("🚀 Iniciando entrenamiento GPU...")


# ═══════════════════════════════════════════════════════════════
# PHASE 3: ENTRENAR UN MODELO
# ═══════════════════════════════════════════════════════════════

def phase_3_train_model(model_key: str) -> None:
    """Train a single model. Fully resumable from checkpoints."""
    cfg = MODEL_CONFIGS[model_key]
    paths = model_paths(model_key)
    ensure_dirs(paths["local_ckpt"], paths["drive_ckpt"], paths["final_model"])

    print(f"\n{'='*70}")
    print(f"PHASE 3: TRAIN {cfg['name']} [{model_key}]")
    print(f"{'='*70}")

    # Skip if already trained
    final_flag = os.path.join(paths["final_model"], "config.json")
    if os.path.exists(final_flag):
        print(f"✅ [{model_key}] Final model exists, skipping training.")
        return

    # Load tokenized train
    print(f"📂 [{model_key}] Loading tokenized train...")
    train_dataset = load_from_disk(paths["tokenized_train"], keep_in_memory=False)
    print(f"📊 [{model_key}] Train: {len(train_dataset)} rows")

    # W&B (optional)
    if USE_WANDB:
        wandb.init(
            project="ai-text-detector-ensemble",
            name=f"v6.1-{model_key}",
            config={
                "model": cfg["name"],
                "dataset_size": len(train_dataset),
                "batch_effective": cfg["batch_size"] * cfg["grad_accum"],
                "learning_rate": cfg["lr"],
                "epochs": cfg["epochs"],
                "version": "v6.1-ensemble",
                "mcl_temperature": 0.3,
            },
            resume="allow",
        )

    # Crash handler — FIXED: specific exception + logging
    def handler_crash(signum, frame):
        try:
            if USE_WANDB:
                wandb.alert(
                    title=f"💥 [{model_key}] Crash",
                    text=f"Checkpoint: {paths['drive_ckpt']}",
                    level=wandb.AlertLevel.ERROR,
                )
                wandb.finish(exit_code=1)
        except Exception as e:
            # Log instead of silencing — operator needs to know alerting failed
            print(f"⚠️  [{model_key}] W&B crash alert failed: {e}", file=sys.stderr)
    signal.signal(signal.SIGTERM, handler_crash)

    flush_ram(f"{model_key}-pre-model")
    flush_vram(f"{model_key}-pre-model")

    # Load model with dtype handling per architecture
    model_kwargs = {
        "num_labels": NUM_LABELS,
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()},
    }

    if model_key == "modernbert":
        model_kwargs["dtype"] = cfg["dtype"]
        model_kwargs["attn_implementation"] = cfg["attn_impl"]
    elif model_key == "deberta":
        model_kwargs["dtype"] = cfg["dtype"]

    print(f"📥 [{model_key}] Loading model...")
    if model_key == "electra":
        # ELECTRA: some configs don't support bf16 init → load fp32 then cast
        model = AutoModelForSequenceClassification.from_pretrained(cfg["name"], **model_kwargs)
        model = model.to(dtype=cfg["dtype"])
    else:
        model = AutoModelForSequenceClassification.from_pretrained(cfg["name"], **model_kwargs)

    flush_ram(f"{model_key}-post-model")
    flush_vram(f"{model_key}-post-model")

    tokenizer = AutoTokenizer.from_pretrained(cfg["name"], model_max_length=cfg["max_length"])

    # Compute training steps and checkpoint interval
    total_steps_est = (len(train_dataset) // (cfg["batch_size"] * cfg["grad_accum"])) * cfg["epochs"]
    save_steps = max(10000, total_steps_est // 12)

    training_args = TrainingArguments(
        output_dir=paths["local_ckpt"],
        eval_strategy="no",              # NUCLEAR: no eval during training
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=500,
        learning_rate=cfg["lr"],   #warmup_ratio=cfg["warmup_ratio"],
        warmup_steps=int(total_steps_est * 0.1),  # reemplaza warmup_ratio (deprecado)
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],  #gradient_checkpointing=True,
        gradient_checkpointing=cfg.get("grad_ckpt", False),  # usa el valor del config
        optim="adamw_torch_fused",
        num_train_epochs=cfg["epochs"],
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        save_total_limit=3,
        dataloader_num_workers=2,        # FIX: overlap data loading with GPU
        dataloader_pin_memory=True,      # FIX: faster host→device transfer
        dataloader_prefetch_factor=2,    # FIX: prefetch 2 batches
        report_to="wandb" if USE_WANDB else "none",
        run_name=f"v6.1-{model_key}",
        torch_compile=cfg.get("torch_compile", True),   # ~25-35% speedup gratis en H100
    )

    trainer = ContrastiveEnsembleTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        class_weights=CLASS_WEIGHTS,
        callbacks=[
            StabilityCallback(
                paths["local_ckpt"], paths["drive_ckpt"],
                model_key, flush_every=500
            ),
        ],
    )

    print(f"\n🚀 [{model_key}] STARTING TRAINING (NUCLEAR — NO EVAL)")
    print(f"   Estimated steps: ~{total_steps_est:,}")
    print(f"   Checkpoints every: {save_steps:,} steps")
    print(f"   Effective batch: {cfg['batch_size'] * cfg['grad_accum']}")

    # Resume from checkpoint (Drive first, then local)
    last_checkpoint = None
    if os.path.isdir(paths["drive_ckpt"]):
        last_checkpoint = get_last_checkpoint(paths["drive_ckpt"])
    if last_checkpoint is None and os.path.isdir(paths["local_ckpt"]):
        last_checkpoint = get_last_checkpoint(paths["local_ckpt"])

    gc.collect()
    torch.cuda.empty_cache()

    if last_checkpoint:
        print(f"🔄 [{model_key}] RESUMING from: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print(f"▶️  [{model_key}] Training from scratch...")
        trainer.train()

    # Save final model
    print(f"\n💾 [{model_key}] Saving final model...")
    trainer.save_model(paths["final_model"])
    tokenizer.save_pretrained(paths["final_model"])
    with open(os.path.join(paths["final_model"], "label_mappings.json"), "w") as f:
        json.dump({
            "label2id": LABEL2ID,
            "id2label": {str(k): v for k, v in ID2LABEL.items()},
        }, f, indent=2)

    if USE_WANDB:
        wandb.alert(
            title=f"✅ [{model_key}] Training Complete",
            text=f"Steps: {trainer.state.global_step}. Model: {paths['final_model']}",
            level=wandb.AlertLevel.INFO,
        )
        wandb.finish()

    del model, trainer, train_dataset, tokenizer
    flush_ram(f"{model_key}-post-train-cleanup")
    flush_vram(f"{model_key}-post-train-cleanup")


# ═══════════════════════════════════════════════════════════════
# PHASE 4: EVALUAR CADA MODELO
# ═══════════════════════════════════════════════════════════════

def phase_4_eval_single(model_key: str) -> tuple:
    """Evaluate a single model and return logits for ensemble."""
    cfg = MODEL_CONFIGS[model_key]
    paths = model_paths(model_key)

    print(f"\n{'='*70}")
    print(f"PHASE 4: EVALUATE {model_key}")
    print(f"{'='*70}")

    print(f"📥 [{model_key}] Loading final model for eval...")
    tokenizer = AutoTokenizer.from_pretrained(paths["final_model"])

    model = AutoModelForSequenceClassification.from_pretrained(
        paths["final_model"], dtype=torch.bfloat16,
    )

    print(f"📂 [{model_key}] Loading tokenized test...")
    test_dataset = load_from_disk(paths["tokenized_test"], keep_in_memory=False)
    print(f"📊 [{model_key}] Test: {len(test_dataset)} rows")

    eval_args = TrainingArguments(
        output_dir="/content/eval_tmp/",
        per_device_eval_batch_size=16,
        eval_accumulation_steps=4,
        bf16=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
    )

    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    print(f"🔍 [{model_key}] Running predictions...")
    results = eval_trainer.predict(test_dataset)

    logits = results.predictions
    labels = results.label_ids

    preds = np.argmax(logits, axis=-1)
    acc = _accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1 = _f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    print(f"🏆 [{model_key}] Accuracy: {acc:.4f} | F1: {f1:.4f}")

    np.save(os.path.join(ENSEMBLE_DIR, f"{model_key}_logits.npy"), logits)
    np.save(os.path.join(ENSEMBLE_DIR, f"{model_key}_labels.npy"), labels)
    with open(os.path.join(ENSEMBLE_DIR, f"{model_key}_metrics.json"), "w") as f:
        json.dump({"accuracy": acc, "f1_weighted": f1, "test_size": len(test_dataset)}, f, indent=2)

    del model, eval_trainer, test_dataset, tokenizer
    flush_ram(f"{model_key}-eval-cleanup")
    flush_vram(f"{model_key}-eval-cleanup")

    return logits, labels, acc, f1


# ═══════════════════════════════════════════════════════════════
# PHASE 5: ENSEMBLE — SOFT VOTING
# ═══════════════════════════════════════════════════════════════

def phase_5_ensemble(all_logits: dict, labels: np.ndarray) -> tuple:
    """Compute soft voting ensemble from individual model logits."""
    print(f"\n{'='*70}")
    print("PHASE 5: ENSEMBLE — SOFT VOTING")
    print(f"{'='*70}")

    probs_list = []
    for key in TRAINING_ORDER:
        logits = all_logits[key]
        probs = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
        probs_list.append(probs)
        print(f"  [{key}] Probs shape: {probs.shape}")

    ensemble_probs = np.mean(probs_list, axis=0)
    ensemble_preds = np.argmax(ensemble_probs, axis=-1)

    acc = _accuracy_metric.compute(predictions=ensemble_preds, references=labels)["accuracy"]
    f1 = _f1_metric.compute(predictions=ensemble_preds, references=labels, average="weighted")["f1"]

    print(f"\n🏆 ENSEMBLE RESULTS:")
    print(f"   Accuracy:    {acc:.4f}")
    print(f"   F1 Weighted: {f1:.4f}")

    report = classification_report(
        labels, ensemble_preds,
        target_names=LABEL_ORDER, digits=4, output_dict=True,
    )
    print(f"\n📊 Per-Class Performance:")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    for cls in LABEL_ORDER:
        r = report[cls]
        print(f"{cls:<12} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1-score']:>10.4f} {r['support']:>10.0f}")

    # Confusion matrix
    cm = confusion_matrix(labels, ensemble_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=LABEL_ORDER, yticklabels=LABEL_ORDER,
    )
    plt.xlabel('Predicted', fontsize=12, fontweight='bold')
    plt.ylabel('Actual', fontsize=12, fontweight='bold')
    plt.title(
        f'Ensemble Confusion Matrix — Acc: {acc:.4f} | F1: {f1:.4f}\n'
        f'DeBERTa + ELECTRA + ModernBERT (Soft Voting)',
        fontsize=13, pad=15,
    )
    cm_path = os.path.join(ENSEMBLE_DIR, "ensemble_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()  # FIX: close figure to free memory (plt.show() not needed in script)
    print(f"✅ Confusion matrix: {cm_path}")

    # Save metrics
    ensemble_metrics = {
        "ensemble_accuracy": acc,
        "ensemble_f1_weighted": f1,
        "method": "soft_voting",
        "models": list(TRAINING_ORDER),
        "version": "v6.1",
        "per_class": {cls: report[cls] for cls in LABEL_ORDER},
        "individual_metrics": {},
    }
    for key in TRAINING_ORDER:
        ind_path = os.path.join(ENSEMBLE_DIR, f"{key}_metrics.json")
        if os.path.exists(ind_path):
            with open(ind_path) as f:
                ensemble_metrics["individual_metrics"][key] = json.load(f)

    with open(os.path.join(ENSEMBLE_DIR, "ensemble_metrics.json"), "w") as f:
        json.dump(ensemble_metrics, f, indent=2)

    np.save(os.path.join(ENSEMBLE_DIR, "ensemble_probs.npy"), ensemble_probs)

    print(f"\n✅ All ensemble artifacts saved to: {ENSEMBLE_DIR}")
    return acc, f1, ensemble_probs


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 70)
    print("ENSEMBLE AI TEXT DETECTOR v6.1 — GPU TRAINING")
    print("DeBERTa-v3 + ELECTRA + ModernBERT — Soft Voting")
    print(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"W&B: {'enabled' if USE_WANDB else 'disabled'}")
    print("Fixes: device safety, wandb optional, dataloader workers, metrics cache")
    print("=" * 70)

    # Phase 0
    phase_0_mount_and_verify()

    # Phase 3: Train each model sequentially
    for model_key in TRAINING_ORDER:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {MODEL_CONFIGS[model_key]['name']}")
        print(f"{'#'*70}")
        phase_3_train_model(model_key)
        flush_ram(f"between-models-after-{model_key}")
        flush_vram(f"between-models-after-{model_key}")

    # Phase 4: Evaluate each model
    print("\n" + "#" * 70)
    print("# INDIVIDUAL EVALUATIONS")
    print("#" * 70)

    all_logits = {}
    labels_ref = None
    individual_results = {}

    for model_key in TRAINING_ORDER:
        logits, labels, acc, f1 = phase_4_eval_single(model_key)
        all_logits[model_key] = logits
        labels_ref = labels
        individual_results[model_key] = {"accuracy": acc, "f1": f1}

    # Phase 5: Ensemble
    ens_acc, ens_f1, ens_probs = phase_5_ensemble(all_logits, labels_ref)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1':>10}")
    print("-" * 42)
    for key in TRAINING_ORDER:
        r = individual_results[key]
        print(f"{key:<20} {r['accuracy']:>10.4f} {r['f1']:>10.4f}")
    print("-" * 42)
    print(f"{'ENSEMBLE':<20} {ens_acc:>10.4f} {ens_f1:>10.4f}")
    print("=" * 70)

    # FIX: correct import path (was referencing ensemble_v6_nuclear)
    print(f"\n✅ ALL DONE. Artifacts in: {ENSEMBLE_DIR}")
    print("To load for production:")
    print(f'  detector = EnsembleAIDetector("{ENSEMBLE_DIR}")')
    print('  result = detector.predict("Your text here")')
