# Guía de Entrenamiento — xplagiax_ensemble_sota_v7
### Step-by-step para principiantes · Google Colab H100/A100

---

## TL;DR — ¿Qué vas a construir?

```
Tus archivos .parquet  →  limpieza + dedup  →  entrenar 2 transformers
    (text / model / idiom)      →  calibración  →  meta-learner  →  threshold conformal
                                                          ↓
                                              detector listo para producción
```

---

## ÍNDICE

1. [Prepara tu entorno en Colab](#1-prepara-tu-entorno-en-colab)
2. [Sube tus datasets](#2-sube-tus-datasets)
3. [Convierte y valida tus .parquet](#3-convierte-y-valida-tus-parquet)
4. [Sube los scripts del detector](#4-sube-los-scripts-del-detector)
5. [Stage 1 — Limpieza y split](#5-stage-1--limpieza-y-split)
6. [Stage 2-3 — Entrenamiento (ModernBERT + DeBERTa)](#6-stage-2-3--entrenamiento)
7. [Stage 4 — Calibración de temperatura](#7-stage-4--calibración-de-temperatura)
8. [Stage 5-6 — Meta-learner y threshold conformal](#8-stage-5-6--meta-learner-y-threshold-conformal)
9. [Evaluación y métricas](#9-evaluación-y-métricas)
10. [Guardar y reutilizar artefactos](#10-guardar-y-reutilizar-artefactos)
11. [Solución de problemas frecuentes](#11-solución-de-problemas-frecuentes)
12. [Tiempos y costes estimados](#12-tiempos-y-costes-estimados)

---

## 1. Prepara tu entorno en Colab

### 1.1 Abre Colab con GPU H100 o A100

1. Ve a [colab.research.google.com](https://colab.research.google.com)
2. Menú → **Entorno de ejecución → Cambiar tipo de entorno de ejecución**
3. Acelerador de hardware: **A100** (o H100 si tienes Colab Pro+)
4. Confirma y espera que cargue

### 1.2 Verifica la GPU

```python
# CELDA 1 — Verificar GPU
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
```

Deberías ver `A100` o `H100` en la salida. Si no, repite el paso 1.1.

### 1.3 Instala dependencias

```python
# CELDA 2 — Instalar dependencias (tarda ~3 minutos)
!pip install -q \
    transformers==4.47.0 \
    datasets==3.2.0 \
    lightgbm==4.5.0 \
    scikit-learn==1.5.2 \
    accelerate==1.2.1 \
    pyarrow==18.0.0 \
    pandas==2.2.3 \
    torch==2.5.1 \
    sentencepiece \
    protobuf

print("✅ Dependencias instaladas")
```

---

## 2. Sube tus datasets

### 2.1 Opción A — Subir desde tu PC (datasets pequeños < 2GB)

```python
# CELDA 3A — Subir archivos manualmente
from google.colab import files
import os

os.makedirs('/content/data', exist_ok=True)
print("📂 Selecciona tus archivos .parquet cuando aparezca el diálogo:")
uploaded = files.upload()

# Mueve los archivos subidos a /content/data/
for fname in uploaded.keys():
    os.rename(fname, f'/content/data/{fname}')
    print(f"  → {fname} movido a /content/data/")
```

### 2.2 Opción B — Desde Google Drive (recomendado para datasets grandes)

```python
# CELDA 3B — Montar Google Drive
from google.drive import drive
from google.colab import drive as gdrive
gdrive.mount('/content/drive')

# Copia tus archivos desde Drive
import shutil, glob

# ⚠️ AJUSTA esta ruta a donde tienes tu carpeta en Drive
DRIVE_PATH = '/content/drive/MyDrive/TU_CARPETA_DE_DATASETS/'

os.makedirs('/content/data', exist_ok=True)
archivos = glob.glob(DRIVE_PATH + '*.parquet')
print(f"Encontrados {len(archivos)} archivos .parquet:")
for f in archivos:
    dest = f'/content/data/{os.path.basename(f)}'
    shutil.copy(f, dest)
    size_mb = os.path.getsize(dest) / 1024**2
    print(f"  → {os.path.basename(f)} ({size_mb:.1f} MB)")
```

### 2.3 Verifica que los archivos están bien

```python
# CELDA 4 — Verificar archivos
import glob
archivos = glob.glob('/content/data/*.parquet')
print(f"Total archivos: {len(archivos)}")
for f in archivos:
    size_mb = os.path.getsize(f) / 1024**2
    print(f"  {os.path.basename(f):40s} {size_mb:8.1f} MB")
```

---

## 3. Convierte y valida tus .parquet

Tu dataset tiene columnas: **`text`**, **`model`**, **`idiom`**

El trainer espera: **`text`** + **`label`** (+ opcionales: `source`, `domain`, `group_id`)

### 3.1 Explora el formato de tus datos

```python
# CELDA 5 — Explorar datos
import pandas as pd
import glob

# Lee el primer archivo para ver la estructura
archivos = glob.glob('/content/data/*.parquet')
df_muestra = pd.read_parquet(archivos[0])

print("=== COLUMNAS ===")
print(df_muestra.columns.tolist())

print("\n=== PRIMERAS 3 FILAS ===")
print(df_muestra.head(3).to_string())

print("\n=== VALORES ÚNICOS EN 'model' ===")
print(df_muestra['model'].value_counts())

print("\n=== VALORES ÚNICOS EN 'idiom' ===")
print(df_muestra['idiom'].value_counts())
```

### 3.2 Mapeo de etiquetas (MUY IMPORTANTE)

El detector maneja 7 clases. Necesitas mapear tus valores de `model` a estas clases:

| Tu valor en `model`                          | Clase del detector |
|----------------------------------------------|--------------------|
| `human`, `Human`, `HUMAN`                   | `Human`            |
| `gpt-4`, `gpt-3.5`, `gpt4o`, `openai`, ...  | `GPT`              |
| `claude`, `claude-3`, `anthropic`, ...       | `Claude`           |
| `gemini`, `bard`, `palm`, ...                | `Gemini`           |
| `grok`, `xai`, ...                           | `Grok`             |
| `mistral`, `mixtral`, ...                    | `Mistral`          |
| `deepseek`, ...                              | `DeepSeek`         |

```python
# CELDA 6 — Ver exactamente qué valores tienes
archivos = glob.glob('/content/data/*.parquet')
todos_los_modelos = set()
for f in archivos:
    df = pd.read_parquet(f, columns=['model'])
    todos_los_modelos.update(df['model'].unique())

print("TODOS los valores de 'model' en tus datasets:")
for m in sorted(todos_los_modelos):
    print(f"  '{m}'")
```

### 3.3 Crea el script de conversión personalizado

```python
# CELDA 7 — Convertir tus datos al formato del trainer

# ⚠️ EDITA este diccionario con TUS valores reales de 'model'
MI_MAPA_MODELOS = {
    # Human
    'human':   'Human',
    'Human':   'Human',
    'HUMAN':   'Human',

    # GPT / OpenAI
    'gpt4':    'GPT',
    'gpt4o':   'GPT',
    'gpt-4':   'GPT',
    'gpt-3.5': 'GPT',
    'gpt3':    'GPT',
    'chatgpt': 'GPT',
    'openai':  'GPT',
    'davinci': 'GPT',

    # Claude / Anthropic
    'claude':    'Claude',
    'claude-3':  'Claude',
    'claude3':   'Claude',
    'anthropic': 'Claude',

    # Gemini / Google
    'gemini':  'Gemini',
    'bard':    'Gemini',
    'palm':    'Gemini',
    'palm2':   'Gemini',

    # Grok / xAI
    'grok':  'Grok',
    'xai':   'Grok',

    # Mistral
    'mistral':  'Mistral',
    'mixtral':  'Mistral',

    # DeepSeek
    'deepseek': 'DeepSeek',

    # Agrega aquí los tuyos si faltan:
    # 'llama':    'GPT',   # ejemplo: LLaMA lo mapeas a GPT o creas una clase nueva
}

def convertir_dataset(archivos_parquet, salida='/content/data_unificado.parquet'):
    """
    Lee todos los .parquet, mapea columnas y guarda en un único archivo.
    """
    CLASES_VALIDAS = {'Human', 'GPT', 'Claude', 'Gemini', 'Grok', 'Mistral', 'DeepSeek'}
    filas = []
    rechazados = {}

    for f in archivos_parquet:
        df = pd.read_parquet(f)
        for _, row in df.iterrows():
            modelo_raw = str(row.get('model', '')).strip()
            label = MI_MAPA_MODELOS.get(modelo_raw)

            if label is None:
                # Búsqueda parcial (minúsculas)
                for k, v in MI_MAPA_MODELOS.items():
                    if k.lower() in modelo_raw.lower():
                        label = v
                        break

            if label not in CLASES_VALIDAS:
                rechazados[modelo_raw] = rechazados.get(modelo_raw, 0) + 1
                continue

            filas.append({
                'text':     str(row.get('text', '')),
                'label':    label,
                'domain':   str(row.get('idiom', '')),  # idiom → domain para group-aware split
                'source':   str(f).split('/')[-1],      # nombre del archivo como source
                'group_id': None,
            })

    if rechazados:
        print("⚠️  Valores NO mapeados (se descartaron):")
        for k, v in sorted(rechazados.items(), key=lambda x: -x[1]):
            print(f"    '{k}': {v} filas")
        print("   → Agrégalos a MI_MAPA_MODELOS y vuelve a correr esta celda")

    df_out = pd.DataFrame(filas)
    df_out.to_parquet(salida, index=False)

    print(f"\n✅ Dataset unificado guardado: {salida}")
    print(f"   Total filas: {len(df_out):,}")
    print("\n📊 Distribución de clases:")
    dist = df_out['label'].value_counts()
    for label, count in dist.items():
        pct = 100 * count / len(df_out)
        barra = '█' * int(pct / 2)
        print(f"   {label:12s} {count:8,}  ({pct:5.1f}%)  {barra}")

    print("\n📊 Distribución por idioma (top 10):")
    print(df_out['domain'].value_counts().head(10).to_string())
    return df_out

archivos = glob.glob('/content/data/*.parquet')
df = convertir_dataset(archivos)
```

### 3.4 Filtra textos muy cortos o vacíos

```python
# CELDA 8 — Filtrar calidad mínima
antes = len(df)
df = df[df['text'].str.len() > 50]        # mínimo 50 caracteres
df = df[df['text'].notna()]
df = df.dropna(subset=['text', 'label'])
df.to_parquet('/content/data_unificado.parquet', index=False)

print(f"Filas antes: {antes:,}")
print(f"Filas después del filtro: {len(df):,}")
print(f"Descartadas: {antes - len(df):,}")
```

---

## 4. Sube los scripts del detector

### 4.1 Opción A — Desde archivo .b64 descargado

Si descargaste el `cambios.tar.gz.b64`:

```python
# CELDA 9A — Descomprimir desde base64
from google.colab import files
print("Sube el archivo cambios.tar.gz.b64:")
uploaded = files.upload()

import base64, tarfile, io
b64_file = list(uploaded.keys())[0]
with open(b64_file, 'rb') as f:
    raw = base64.b64decode(f.read())

with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
    tar.extractall('/content/')

print("✅ Scripts extraídos:")
import glob
for f in glob.glob('/content/xplagiax_ensemble_sota_v7/*.py'):
    print(f"   {f}")
```

### 4.2 Opción B — Pegar el código directamente (si no tienes el .b64)

```python
# CELDA 9B — Crear el directorio y copiar scripts
import os
os.makedirs('/content/xplagiax_ensemble_sota_v7', exist_ok=True)
# Pega el contenido de detector_final.py, trainer.py, test.py aquí
# O usa %%writefile:

# %%writefile /content/xplagiax_ensemble_sota_v7/detector_final.py
# [pega el contenido aquí]
```

### 4.3 Verifica que los scripts están bien

```python
# CELDA 10 — Verificar imports
import sys
sys.path.insert(0, '/content/xplagiax_ensemble_sota_v7')

try:
    from detector_final import SOTADetector, canonicalize_text, LABEL_ORDER
    print("✅ detector_final.py OK")
    print(f"   Clases: {LABEL_ORDER}")
except Exception as e:
    print(f"❌ Error: {e}")

try:
    import trainer  # solo verifica que importa sin errores
    print("✅ trainer.py OK")
except Exception as e:
    print(f"❌ Error en trainer.py: {e}")
```

---

## 5. Stage 1 — Limpieza y split

```python
# CELDA 11 — Stage 1: dedup + split group-aware
import sys, os, pickle, json
sys.path.insert(0, '/content/xplagiax_ensemble_sota_v7')

from trainer import load_rows, dedup_and_split

OUT_DIR = '/content/artifacts'
os.makedirs(OUT_DIR, exist_ok=True)

# Carga el dataset unificado
print("⏳ Cargando datos...")
rows = load_rows(
    '/content/data_unificado.parquet',
    text_col='text',
    label_col='label',
)
print(f"   {len(rows):,} filas cargadas")

# Split con deduplicación MinHash
print("\n⏳ Deduplicando y dividiendo (puede tardar 5-15 min con 4M filas)...")
train, val, test = dedup_and_split(
    rows,
    test_size=0.10,   # 10% para test final (NUNCA lo toques durante entrenamiento)
    val_size=0.10,    # 10% para calibración y meta-learner
)

# Guarda el test set (SELLADO — no lo uses para nada más)
with open(f'{OUT_DIR}/test_rows.pkl', 'wb') as f:
    pickle.dump(test, f)

# Guarda también train y val para poder reanudar
with open(f'{OUT_DIR}/train_rows.pkl', 'wb') as f:
    pickle.dump(train, f)
with open(f'{OUT_DIR}/val_rows.pkl', 'wb') as f:
    pickle.dump(val, f)

json.dump(
    {'n_train': len(train), 'n_val': len(val), 'n_test': len(test)},
    open(f'{OUT_DIR}/split_sizes.json', 'w'), indent=2
)

print("\n✅ Split listo:")
print(f"   Train : {len(train):,} filas  (80%)")
print(f"   Val   : {len(val):,} filas  (10%) — calibración y meta-learner")
print(f"   Test  : {len(test):,} filas  (10%) — evaluación final SELLADA")
print(f"\n⚠️  El test set está sellado en {OUT_DIR}/test_rows.pkl")
print("    NO lo uses hasta la evaluación final.")
```

---

## 6. Stage 2-3 — Entrenamiento

> **Tiempo estimado**: 4.46M filas × 2 epochs × 2 modelos ≈ **6-10 horas** en A100.
> Usa checkpointing de Drive para no perderlo si Colab desconecta.

### 6.1 Configura backup automático en Drive

```python
# CELDA 12 — Backup automático a Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

DRIVE_BACKUP = '/content/drive/MyDrive/xota_v7_artifacts'
os.makedirs(DRIVE_BACKUP, exist_ok=True)

def backup_a_drive(msg="checkpoint"):
    """Copia los artefactos actuales a Drive."""
    import shutil
    shutil.copytree(OUT_DIR, DRIVE_BACKUP, dirs_exist_ok=True)
    print(f"  💾 Backup a Drive: {msg}")

backup_a_drive("inicial")
```

### 6.2 Entrena ModernBERT (modelo principal)

```python
# CELDA 13 — Entrenar ModernBERT
# ⏱ ~3-4 horas en A100 con 4.46M filas, 2 epochs
from trainer import train_supervised, SupervisedBranch
import pickle

# Carga train/val (si reinicias el runtime, carga desde pkl)
with open(f'{OUT_DIR}/train_rows.pkl', 'rb') as f:
    train = pickle.load(f)
with open(f'{OUT_DIR}/val_rows.pkl', 'rb') as f:
    val = pickle.load(f)

print(f"Train: {len(train):,} | Val: {len(val):,}")
print("\n⏳ Entrenando ModernBERT...")
print("   (verás 'step XXX loss=X.XXXX' cada 200 pasos)")
print("   Presiona el botón ■ para parar si hay un error")

# Sin length experts (más rápido, buena base)
save_dir = train_supervised(
    train_rows=train,
    val_rows=val,
    out_dir=OUT_DIR,
    model_key='modernbert',
    epochs=2,
    length_expert=None,   # None = entrena en todo el dataset
    max_length=512,
)
print(f"\n✅ ModernBERT guardado en: {save_dir}")
backup_a_drive("modernbert_done")
```

### 6.3 (Opcional) Entrena con expertos de longitud — +0.3% F1

```python
# CELDA 14 — Length experts (solo si quieres máximo rendimiento)
# ⏱ Duplica el tiempo de entrenamiento
# Entrena un modelo para textos cortos (<60 palabras) y otro para largos

print("⏳ Entrenando ModernBERT SHORT expert...")
train_supervised(train, val, OUT_DIR, 'modernbert', epochs=2, length_expert='short')
backup_a_drive("modernbert_short")

print("⏳ Entrenando ModernBERT LONG expert...")
train_supervised(train, val, OUT_DIR, 'modernbert', epochs=2, length_expert='long')
backup_a_drive("modernbert_long")
```

### 6.4 Entrena DeBERTa (segundo modelo del ensemble)

```python
# CELDA 15 — Entrenar DeBERTa
# ⏱ ~2-3 horas adicionales en A100
print("⏳ Entrenando DeBERTa-v3...")
save_dir = train_supervised(
    train_rows=train,
    val_rows=val,
    out_dir=OUT_DIR,
    model_key='deberta',
    epochs=2,
    length_expert=None,
    max_length=512,
)
print(f"✅ DeBERTa guardado en: {save_dir}")
backup_a_drive("deberta_done")
```

### 6.5 Verifica que los modelos se guardaron

```python
# CELDA 16 — Verificar modelos
import glob
modelos = glob.glob(f'{OUT_DIR}/*/config.json')
print(f"Modelos entrenados: {len(modelos)}")
for m in modelos:
    folder = os.path.dirname(m)
    size_mb = sum(
        os.path.getsize(f) for f in glob.glob(f'{folder}/**', recursive=True)
        if os.path.isfile(f)
    ) / 1024**2
    print(f"  {os.path.basename(folder):25s}  {size_mb:.0f} MB")
```

---

## 7. Stage 4 — Calibración de temperatura

La temperatura escala los logits del modelo para que las probabilidades sean más precisas.
**Se ajusta en VAL, no en test.**

```python
# CELDA 17 — Calibración de temperatura
from trainer import fit_temperature, SupervisedBranch

print("⏳ Cargando modelos entrenados...")
branch = SupervisedBranch(
    ensemble_dir=OUT_DIR,
    model_keys=('modernbert', 'deberta'),
)

print("⏳ Calibrando temperatura en val set...")
temps = fit_temperature(branch, val, OUT_DIR)

print("\n✅ Temperaturas calibradas:")
for nombre, t in temps.items():
    interpretacion = "↑ logits comprimidos (modelo overconfident)" if t > 1.2 else \
                     "↓ logits expandidos (modelo underconfident)" if t < 0.8 else \
                     "≈ bien calibrado"
    print(f"   {nombre:20s}  T={t:.4f}  {interpretacion}")

backup_a_drive("temperatura_calibrada")
```

---

## 8. Stage 5-6 — Meta-learner y threshold conformal

### ¿Qué hace cada stage?
- **Meta-learner**: combina las 3 branches (supervisionada + cero-shot + estilométrica) en una sola puntuación P(IA)
- **Conformal threshold**: fija el umbral de decisión con una **garantía estadística** de que el FPR en humanos ≤ 1% (o el target que elijas)

```python
# CELDA 18 — Meta-learner + threshold conformal
from trainer import fit_meta_and_conformal

print("⏳ Ensamblando features del val set...")
print("   (Branch A: supervised | Branch Z: zero-shot | Branch F: stylometric)")
print("   Esto puede tardar 20-40 minutos con 400K filas de val...")

fit_meta_and_conformal(
    out_dir=OUT_DIR,
    val_rows=val,
    target_fpr=0.01,          # garantía: máximo 1% de falsos positivos en humanos
    enable_zeroshot=False,    # pon True si tienes GPU libre y quieres Binoculars
)

# Muestra el threshold aprendido
import json
conf = json.load(open(f'{OUT_DIR}/conformal.json'))
print(f"\n✅ Threshold conformal calculado:")
print(f"   tau_ai    = {conf['tau_ai']:.4f}  → P(IA) ≥ este valor → AI_DETECTED")
print(f"   tau_human = {conf['tau_human']:.4f}  → P(IA) ≤ este valor → HUMAN")
print(f"   Entre ambos → INCONCLUSIVE (abstención segura)")
print(f"\n   FPR garantizado: ≤ {conf['target_fpr']*100:.1f}% de humanos marcados como IA")

backup_a_drive("meta_y_conformal")
```

### 8.1 Calibración conformal POR IDIOMA (blindaje legal multiidioma)

`fit_meta_and_conformal` ya genera **automáticamente** un segundo archivo,
`conformal_grouped.json`, con un umbral independiente por cada idioma de tu
columna `idiom`. Esto cierra el agujero de *intercambiabilidad*: la garantía
de FPR ≤ 1% se mantiene **dentro de cada idioma**, no solo en promedio.

> **Por qué importa**: con un único umbral global, un idioma no-nativo
> infra-representado puede sufrir FPR del 2-5% aunque el promedio diga 1%.
> Ese es exactamente el escándalo de los detectores comerciales con
> estudiantes no-nativos. Un umbral por idioma lo elimina.

```python
# CELDA 18.1 — Inspeccionar los umbrales por idioma
import json
gc = json.load(open(f'{OUT_DIR}/conformal_grouped.json'))

print(f"FPR objetivo: {gc['target_fpr']*100:.1f}%")
print(f"Mínimo de humanos para calibrar un idioma: {gc['min_group_calib']}")
print(f"\nUmbral GLOBAL (fallback): tau_ai = {gc['global']['tau_ai']:.4f}\n")

print(f"{'Idioma':12s} {'N_humanos':>10s} {'tau_ai':>8s} {'Estado'}")
print("-" * 50)
for idioma, size in sorted(gc['group_sizes'].items(), key=lambda x: -x[1]):
    if idioma in gc['groups']:
        tau = gc['groups'][idioma]['tau_ai']
        estado = "✅ calibrado propio"
    else:
        tau = gc['global']['tau_ai']
        estado = f"⚠️ usa global (n<{gc['min_group_calib']})"
    print(f"{idioma:12s} {size:>10,} {tau:>8.4f}  {estado}")

print("\n💡 Idiomas con 'usa global' necesitan más muestras humanas")
print("   para tener garantía propia de FPR. Recoge más datos de esos idiomas.")
```

**Cómo lo usa el detector en producción** — pásale el idioma del texto:

```python
# El detector enruta al umbral del idioma automáticamente
v = detector.classify(texto, group='es')   # usa el umbral calibrado para español
v = detector.classify(texto, group='fr')   # si 'fr' no se calibró → usa global
                                            # y añade flag 'group_fallback_global'

# Si el flag aparece, NO tienes garantía de FPR para ese idioma:
if 'group_fallback_global' in v.flags:
    print("⚠️ Idioma sin calibración propia — revisar acusación con cuidado")
```

> **Regla legal**: solo acusa (`AI_DETECTED`) cuando el veredicto NO lleve el
> flag `group_fallback_global`. Si lo lleva, trata el resultado como
> `INCONCLUSIVE` para ese idioma hasta tener ≥100 muestras humanas y recalibrar.

---

## 9. Evaluación y métricas

**REGLA DE ORO**: el test set solo se abre UNA VEZ — aquí.

```python
# CELDA 19 — Evaluación final (ejecutar SOLO al final)
import subprocess
result = subprocess.run(
    [
        'python3', '/content/xplagiax_ensemble_sota_v7/test.py',
        '--artifacts', OUT_DIR,
        '--target-fpr', '0.01',
        '--no-zeroshot',   # quitar si entrenaste con zeroshot
    ],
    capture_output=True, text=True
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[:2000])
```

### ¿Cómo interpretar los resultados?

```
MCC                  → principal métrica (0=azar, 1=perfecto). Target: > 0.80
AUROC                → ranking puro. Target: > 0.95
TPR@1%FPR            → recall cuando solo el 1% de humanos son falsos positivos. Target: > 0.70
FPR_human_deployed   → FPR real con tu threshold. Debe ser ≤ 0.01
ECE                  → calibración (0=perfecta). Target: < 0.05

Robustez bajo ataques (mcc_drop_under_attack):
  homoglyph  → debe ser ≈ 0.00  (canonicalizador lo defiende)
  zero_width → debe ser ≈ 0.00
  char_noise → debe ser < 0.05
  whitespace → debe ser < 0.02
```

### Evaluación por idioma (audit de falsos positivos no-nativos)

```python
# CELDA 20 — Audit por idioma (importante para textos no en inglés)
import pickle
from detector_final import SOTADetector
import numpy as np

detector = SOTADetector(OUT_DIR, target_fpr=0.01, enable_zeroshot=False)

with open(f'{OUT_DIR}/test_rows.pkl', 'rb') as f:
    test = pickle.load(f)

# Separa por idioma
from collections import defaultdict
por_idioma = defaultdict(list)
for r in test:
    por_idioma[r.get('domain', 'unknown')].append(r)

print(f"{'Idioma':15s} {'N_human':>8} {'FPR':>8} {'N_ai':>8} {'TPR':>8}")
print("-" * 55)
for idioma, filas in sorted(por_idioma.items()):
    humanos = [r for r in filas if r['label'] == 'Human']
    ias     = [r for r in filas if r['label'] != 'Human']
    if len(humanos) < 10:
        continue

    fp_h = sum(1 for r in humanos
               if detector.classify(r['text']).verdict == 'AI_DETECTED')
    tp_a = sum(1 for r in ias
               if detector.classify(r['text']).verdict == 'AI_DETECTED') if ias else 0

    fpr = fp_h / len(humanos)
    tpr = tp_a / len(ias) if ias else float('nan')

    alerta = "⚠️" if fpr > 0.05 else ""
    print(f"{idioma:15s} {len(humanos):>8,} {fpr:>8.3f} {len(ias):>8,} {tpr:>8.3f} {alerta}")
```

---

## 10. Guardar y reutilizar artefactos

### 10.1 Guarda todo en Drive (backup final)

```python
# CELDA 21 — Backup final completo
import shutil

DRIVE_FINAL = '/content/drive/MyDrive/xota_v7_final'
shutil.copytree(OUT_DIR, DRIVE_FINAL, dirs_exist_ok=True)
print(f"✅ Artefactos guardados en: {DRIVE_FINAL}")

# Lista de lo que necesitas para producción:
archivos_produccion = [
    'modernbert_final/',       # (o _short/ y _long/ si usaste length experts)
    'deberta_final/',
    'calibration.json',        # temperaturas
    'meta_learner.pkl',        # meta-learner
    'conformal.json',          # threshold global
    'conformal_grouped.json',  # threshold POR IDIOMA (garantía multiidioma)
    'test_rows.pkl',           # test set sellado (para re-evaluar)
    'split_sizes.json',
]
print("\nArchivos necesarios para producción:")
for f in archivos_produccion:
    path = f'{OUT_DIR}/{f}'
    existe = os.path.exists(path)
    print(f"  {'✅' if existe else '❌'} {f}")
```

### 10.2 Prueba el detector en producción

```python
# CELDA 22 — Prueba de inferencia
from detector_final import SOTADetector
import json

detector = SOTADetector(
    ensemble_dir=OUT_DIR,
    target_fpr=0.01,
    enable_zeroshot=False,
)

textos_prueba = [
    ("HUMANO", "Mi perro se llama Firulais y le gusta jugar en el parque los domingos."),
    ("IA-GPT", "The implementation of machine learning algorithms requires careful consideration of hyperparameters and optimization strategies to achieve optimal performance."),
    ("ATAQUE",  "The mіtосhоndrіа іs thе рowerhоusе оf thе cеll."),  # Cyrillic lookalikes
]

print(f"{'TIPO':10s} {'VEREDICTO':15s} {'P(IA)':8s} {'CONFIANZA':10s} {'TAMPER'}")
print("-" * 65)
for tipo, texto in textos_prueba:
    v = detector.classify(texto)
    tamper = "🚨 OBFUSCADO" if v.tamper.get('suspicious') else ""
    print(f"{tipo:10s} {v.verdict:15s} {v.p_ai:8.4f} {v.confidence:10.4f} {tamper}")
```

---

## 11. Solución de problemas frecuentes

### ❌ "CUDA out of memory"
```python
# Reduce el batch size en MODEL_CONFIGS en trainer.py
# Cambia "bs": 32 a "bs": 16 para modernbert
# Cambia "bs": 16 a "bs": 8 para deberta
```

### ❌ "already trained, skipping"
```python
# El trainer detecta que ya existe el modelo y lo salta (es correcto)
# Si quieres re-entrenar desde cero:
import shutil
shutil.rmtree(f'{OUT_DIR}/modernbert_final', ignore_errors=True)
# Luego vuelve a correr la celda de entrenamiento
```

### ❌ Runtime desconectado a mitad del entrenamiento
```python
# Los modelos se guardan al FINAL de cada epoch completo
# Si desconecta en medio de un epoch, hay que re-empezar ese modelo
# La celda de entrenamiento salta automáticamente lo que ya está guardado

# Para reanudar, carga train/val desde pkl y vuelve a ejecutar la celda
with open(f'{OUT_DIR}/train_rows.pkl', 'rb') as f:
    train = pickle.load(f)
with open(f'{OUT_DIR}/val_rows.pkl', 'rb') as f:
    val = pickle.load(f)
# → vuelve a correr CELDA 13, 15, etc.
```

### ❌ "No module named 'lightgbm'"
```python
!pip install -q lightgbm
# El meta-learner hace fallback a LogisticRegression si no está lightgbm
```

### ❌ Muchos valores de 'model' no mapeados
```python
# Agrega las entradas que faltan a MI_MAPA_MODELOS en CELDA 7
# Ejemplo: si tienes 'llama3', 'llama-3-8b', etc.
MI_MAPA_MODELOS['llama3']    = 'GPT'   # o crea clase nueva si tienes suficientes muestras
MI_MAPA_MODELOS['llama-3-8b'] = 'GPT'
```

### ❌ FPR alto en un idioma específico
```python
# El campo 'idiom' se usa como 'domain' para el group-aware split
# Asegúrate de que cada idioma está representado en TODOS los splits
# Revisa el output de Stage 1 (dedup_and_split)
# Si un idioma tiene muy pocas muestras, se puede sesgar el split
```

---

## 12. Tiempos y costes estimados

### Hardware: Google Colab A100 (~$2.5/hora con Colab Pro+)

| Stage                        | 4.46M filas | 1M filas  |
|------------------------------|-------------|-----------|
| 1. Dedup + split             | ~15 min     | ~4 min    |
| 3. ModernBERT (2 epochs)     | ~3-4 horas  | ~45 min   |
| 3. DeBERTa (2 epochs)        | ~2-3 horas  | ~30 min   |
| 3. Length experts (opcional) | ×2 tiempo   | ×2 tiempo |
| 4. Calibración temperatura   | ~10 min     | ~3 min    |
| 5-6. Meta + conformal        | ~30 min     | ~8 min    |
| **TOTAL (sin experts)**      | **~7 horas**| **~90 min**|
| **TOTAL (con experts)**      | **~13 horas**| **~3 horas**|

### Coste estimado con A100

| Configuración                | Tiempo   | Coste aprox. |
|------------------------------|----------|--------------|
| Sin length experts (rápido)  | ~7 horas | ~$17         |
| Con length experts (SOTA)    | ~13 horas| ~$32         |

### Precisión esperada

| Métrica      | Sin experts | Con experts | Meta-ensemble |
|--------------|-------------|-------------|---------------|
| MCC          | ~0.78       | ~0.82       | ~0.85         |
| AUROC        | ~0.94       | ~0.96       | ~0.97         |
| TPR@1%FPR    | ~0.65       | ~0.72       | ~0.75         |
| FPR humanos  | ≤ 1.0%      | ≤ 1.0%      | ≤ 1.0%        |

> Los scores con expertos de longitud y meta-learner sobre el dataset de 4.46M son estimados basados en los benchmarks de SzegedAI (2025.genaidetect-1.15) escalados. Los números reales dependerán de la calidad y distribución de tu dataset.

---

## CHECKLIST FINAL

Antes de usar el detector en producción, confirma:

- [ ] `split_sizes.json` — train/val/test > 0
- [ ] `modernbert_final/config.json` — modelo guardado
- [ ] `deberta_final/config.json` — modelo guardado
- [ ] `calibration.json` — temperaturas entre 0.5 y 2.0
- [ ] `meta_learner.pkl` — tipo `lightgbm` o `logreg`
- [ ] `conformal.json` — `tau_ai` entre 0.5 y 0.99
- [ ] MCC > 0.75 en test set
- [ ] FPR_human_deployed ≤ 0.01 en test set
- [ ] mcc_drop_under_attack (homoglyph) < 0.05

---

*Guía para xplagiax_ensemble_sota_v7 — basada en GenAIDetect 2025 y SzegedAI*
