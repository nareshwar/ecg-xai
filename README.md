# ECG-XAI

Explainability toolkit for 12-lead ECG classifiers. The repo bundles LIME-style perturbation explanations, a TimeSHAP-inspired variant, fusion logic to combine them, plotting utilities, and automatic evaluation metrics (token-level AttAUC and deletion-based faithfulness). It targets PhysioNet-style ECG records (.mat/.hea) and Keras/TF models that output SNOMED-coded probabilities.

# ECG‑XAI notebook guides

This repo has two “runner” notebooks that cover the same end‑to‑end workflow (select → explain → fuse → evaluate → stability),
but in two different environments:

- **`ecg-xai-kaggle.ipynb`**: a Kaggle‑friendly demo/eval runner (uses Kaggle Inputs + `/kaggle/working` caches).
- **`ecg-xai-local.ipynb`**: a local runner (assumes you’re running from `notebook/` inside the repo).

---

## Notebook 1 — `ecg-xai-kaggle.ipynb` (Kaggle runner)

### What it’s for
Run the ECG‑XAI pipeline on Kaggle with minimal setup:
1) clone + install the repo,
2) load a pretrained model + precomputed arrays from Kaggle Inputs,
3) build/load `sel_df`,
4) generate **LIME**, **TimeSHAP**, and **fused** explanations,
5) compute evaluation metrics + **extra‑beat stability**,
6) write outputs to `/kaggle/working/outputs`.

### Inputs you must provide (Kaggle “Datasets” as Inputs)
The notebook expects **two Kaggle Inputs** mounted under `/kaggle/input/`:

**Also required:** the raw ECG record dataset (PhysioNet‑style `.hea`/`.mat`). `ecg_filenames.npy` contains file paths that should resolve to those records on Kaggle, so you must attach the dataset that provides the `.hea`/`.mat` files as an Input as well (see the companion Kaggle notebook for how the datasets are wired).

Companion Kaggle notebook (shows the configured dataset Inputs):
- `https://www.kaggle.com/code/nareshuhull/ecg-xai`

#### 1) Input: `metadata`  → `/kaggle/input/metadata/`
Required files:
- `ecg_filenames.npy` — list/array of paths to PhysioNet records (usually `.mat` paths).
- `ecg_model_probs.npy` — model probabilities, shape `(N, C)`.
- `ecg_y_true.npy` — ground truth multi‑hot labels, shape `(N, C)`.
- `snomed_classes.npy` — class codes in the same order as model outputs.

Optional:
- `sel_df_cache.npz` — prebuilt selection cache (speeds up reruns).

#### 2) Input: `resnet` → `/kaggle/input/resnet/`
Required file (either is fine):
- `keras/default/1/resnet_final.keras` **or**
- `keras/default/1/resnet_final.h5`

> The notebook includes a small compatibility helper that renames a “mislabelled `.keras` that is actually HDF5” into a `.h5`
> file before calling `tf.keras.models.load_model(..., compile=False)`.

### Outputs (written to `/kaggle/working`)
- `/kaggle/working/cache/`
  - `sel_df_cache.npz` (selection cache)
  - `ecg_durations.npy` (duration cache for filtering)
- `/kaggle/working/outputs/`
  - `ecg_xai_sel_meta.csv` (the selected examples per class)
  - `df_eval_attauc_deletion.csv` (token‑level plausibility + deletion faithfulness)
  - `df_eval_stability.csv` (extra‑beat stability metrics)
  - `extra_beat_aug/` (augmented `.mat` files and any experiment artifacts)

### The workflow (matches the notebook section headers)

#### 0) Configuration (the only cell you usually edit)
`RunConfig` supports:
- `run_mode`: `"demo"` or `"eval"`
- `seed`: base seed (also used to derive per‑record stability seeds)
- `force_recompute_sel_df`: rebuild selection even if cache exists
- `max_examples_per_class`: demo cap (auto‑set to 50 in eval mode)
- `plot`: plot fused payloads inline (auto‑off in eval mode)

#### 1) Clone & install
- Clones the repo to `/kaggle/working/ecg-xai`
- Installs the project with `pip install . --no-deps` to avoid Kaggle image dependency churn
- Installs `ecg-plot` + `kaleido` for plotting/export

#### 2) Imports + GPU sanity check
- Imports TensorFlow with init logs redirected to `/kaggle/working/tf_init.log`
- Prints TF version + visible GPUs

#### 3) Paths + input validation
- Defines all paths for:
  - Kaggle inputs (`/kaggle/input/metadata`, `/kaggle/input/resnet`)
  - Working caches (`/kaggle/working/cache`)
  - Outputs (`/kaggle/working/outputs`)
- Fails fast with a clear `FileNotFoundError` if required input files are missing.

#### 4) Load model + metadata arrays
- Loads the model (`compile=False`)
- Loads arrays: filenames / probs / y_true / class_names

#### 5) Build or load `sel_df`
`sel_df` is the “selection dataframe” that drives the rest of the notebook.
- If a working cache exists: load `sel_df_cache.npz`
- Else if an input cache exists: load it and copy to working
- Else: build from scratch using:
  - `build_selection_df_with_aliases(..., k_per_class=50, min_prob=0.85, max_duration_sec=20.0, duration_cache_path=...)`
- Saves `ecg_xai_sel_meta.csv` for inspection.

#### 5.1) “Single example” demo helper
Defines a `run_demo_for_class(<SNOMED_CODE>)` helper that:
- picks a representative record for a given class (e.g., AF or sinus rhythm),
- runs LIME + TimeSHAP on that record,
- fuses the results,
- optionally plots,
- returns a small one‑row summary you can display.

#### 6) Run LIME + TimeSHAP + fused explanations (full run)
- **6.1 LIME**: loops over `target_classes = list(TARGET_META.keys())`,
  uses `default_explainer_config(cls)` to get per‑class windows/priors,
  then calls `run_lime_for_one_class_from_sel(...)`.
- **6.2 TimeSHAP**: same pattern via `run_timeshap_for_one_class_from_sel(...)` (uses `link="logit"`).
- **6.3 Fusion**:
  - aligns LIME and TimeSHAP rows by `val_idx`,
  - builds payloads with `payload_from_lime_row(...)` and `payload_from_timeshap_row(...)`,
  - fuses via `fuse_lime_timeshap_payload(...)` (default in notebook: `agg="geomean", beta=1.0, tau=0.02, topk=5`)
  - applies optional per‑class `method_weights` (example: sinus rhythm gets `(0.7, 0.3)`).

> Tip: Kaggle Inputs are read‑only. If you want to **save** `df_lime_all.pkl`, `df_ts_all.pkl`, or fused payloads,
> write them under `/kaggle/working/cache` or `/kaggle/working/outputs` (not under `/kaggle/input/...`).

#### 7) Evaluate explanations
Runs:
- `evaluate_all_payloads(all_payloads=all_fused_payloads, method_label="LIME+TimeSHAP", model=model, class_names=class_names)`
and writes:
- `df_eval_attauc_deletion.csv`

#### 8) Extra‑beat stability experiment
For each evaluated record:
- creates a deterministic seed using `CFG.seed XOR crc32(mat_path)`
- calls `run_extra_beat_stability_experiment(...)` which:
  - generates an “extra heartbeat” augmented record,
  - reruns explanations,
  - compares explanation stability (e.g., Spearman / Jaccard / RBO variants)
- writes:
  - `df_eval_stability.csv`

#### 9) Per‑class summary
Aggregates stability metrics per `(meta_code, class_name)`:
- count + mean/std for key metrics.

### Common tweaks (quick and safe)
- **Faster demo:** `run_mode="demo"`, `max_examples_per_class=1`, `plot=False`
- **More coverage:** `run_mode="eval"` (auto: 50 per class, plots off)
- **Selection strictness:** raise `min_prob` (e.g., 0.90) or reduce `max_duration_sec`
- **Fusion behavior:** change `agg`, `beta`, `tau`, `topk`, and per‑class `method_weights`

---

## Notebook 2 — `ecg-xai-local.ipynb` (Local runner)

### What it’s for
Run the same pipeline locally (or in a local Jupyter environment) against either:
- the repo’s **sample data** (`data/sample`) — includes **50 AF** + **50 SNR** records for quick local runs, or
- your own PhysioNet‑style dataset (by pointing `ECGXAI_DATA_ROOT` to it).

It includes an optional “precompute arrays” stage (probabilities + labels) so you can reproduce the `metadata` artifacts that
the Kaggle notebook expects.

### Assumed working directory
This notebook is written as if you open it from the repo’s **`notebook/`** folder:

- It does `sys.path.append('../src')`
- It uses `ROOT = Path.cwd().parent` (so `ROOT` becomes the repo root)

### Key paths / artifacts
The notebook defines:
- **Data root**
  - `SAMPLE_ROOT = ROOT / "data" / "sample"`
  - `os.environ["ECGXAI_DATA_ROOT"] = str(SAMPLE_ROOT)`
  - then reloads `ecgxai.config` so it picks up the env var.
- **Model**
  - `MODEL_PATH = ROOT / "model" / "resnet_final.h5"`
- **Metadata artifacts (saved locally)**
  - `data/ecg_filenames.npy`
  - `data/ecg_model_probs.npy`
  - `data/ecg_y_true.npy`
  - `data/snomed_classes.npy`
  - `data/ecg_durations.npy`
- **Outputs**
  - `outputs/<run_mode>/...` (cached runs)
  - plus summary CSVs under `outputs/<run_mode>/`

### The workflow (matches the notebook cells)

#### 0) Install in editable mode
Runs:
- `%pip install -e ..`
so your notebook always uses your current working tree.

#### 1) Imports
Imports the same “runner” utilities as the Kaggle notebook:
- selection builder (`build_selection_df_with_aliases`)
- fused pipeline (`run_fused_pipeline_for_classes`)
- evaluation (`evaluate_all_payloads`)
- extra‑beat stability (`run_extra_beat_stability_experiment`)
- plus `save_run` / `load_run` helpers

#### 2) Point the project at the data + load the model
- sets `ECGXAI_DATA_ROOT`
- reloads config
- loads the Keras model (`compile=False`)
- loads `class_names`

#### 3) (Optional) Build `y_true` + run model inference locally
If you want to regenerate the `metadata` arrays from raw sample data:
- `import_key_data(DATA_ROOT)` loads demographics + labels + filenames
- `build_y_true_from_labels(labels, class_names)` makes multi‑hot ground truth
- `batched_predict_all(model, ecg_filenames, maxlen=MAXLEN, batch_size=32)` computes probabilities
- saves `.npy` files into `data/`

> If you already have `ecg_model_probs.npy` and `ecg_y_true.npy`, you can skip this section.

#### 4) Build `sel_df`
Loads arrays from disk and builds `sel_df` via:
- `build_selection_df_with_aliases(..., k_per_class=50, min_prob=0.85, max_duration_sec=20.0, duration_cache_path=...)`
Then saves:
- `outputs/ecg_xai_sel_meta_p0.85_k5.csv` (path name reflects the selection params)

#### 5) Choose run mode + caching
Two modes:
- `run_mode="demo"` → `max_examples_per_class=3`, `plot=True`
- `run_mode="eval"` → `max_examples_per_class=50`, `plot=False`

Then:
- checks for cached assets under `outputs/<run_mode>/`
- if present: `load_run(RUN_DIR)`
- else: runs `run_fused_pipeline_for_classes(...)` and saves via `save_run(...)`

Finally it caps `sel_df` to `max_examples_per_class` per class (so later steps stay consistent).

#### 6) LIME / TimeSHAP / Fusion
Same pattern as Kaggle:
- `default_explainer_config(cls)` provides per‑class settings
- runs LIME and TimeSHAP stages
- fuses with `fuse_lime_timeshap_payload(...)`
- plots with `plot_from_payload(...)` if enabled

#### 7) Evaluate
Writes:
- `outputs/<run_mode>/df_eval_attauc_deletion.csv`

#### 8) Extra‑beat stability
Writes:
- `outputs/<run_mode>/df_eval_stability.csv`
and generates augmented records under:
- `outputs/extra_beat_aug/` (or the configured stability output dir)

#### 9) Per‑class stability summary
Aggregates stability metrics (count + means/stats) per class.

### Common local tweaks
- Use your own dataset:
  - set `os.environ["ECGXAI_DATA_ROOT"]` to your PhysioNet‑style root
  - ensure the loader can find `.mat` + `.hea` pairs
- Speed up:
  - reduce `batch_size` in `batched_predict_all`
  - reduce `k_per_class`, increase `min_prob`, lower `max_examples_per_class`
- Plot control:
  - `MODE_CFG["demo"]["plot"] = False` for faster/cleaner runs

---

## Glossary (quick)
- **`sel_df`**: selection DataFrame containing the subset of records you’ll explain (typically high‑confidence picks per target class).
- **`TARGET_META`**: mapping of target SNOMED codes to human names + aliases (meta‑classes).
- **Payload**: a plotting/evaluation‑friendly dict containing lead×time spans + weights (from LIME/TimeSHAP or fused).
- **AttAUC / F1 (token‑level)**: plausibility metrics comparing “what the explainer highlights” vs clinically plausible lead/window tokens.
- **Deletion AUC**: faithfulness metric based on removing highlighted regions and measuring prediction drop.
- **Extra‑beat stability**: how consistent explanations remain under a controlled augmentation that inserts a heartbeat.


