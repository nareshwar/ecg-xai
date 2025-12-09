# ECG-XAI

Explainability toolkit for 12-lead ECG classifiers. The repo bundles LIME-style perturbation explanations, a TimeSHAP-inspired variant, fusion logic to combine them, plotting utilities, and automatic evaluation metrics (token-level AttAUC/F1 and deletion-based faithfulness). It targets PhysioNet-style ECG records (.mat/.hea) and Keras/TF models that output SNOMED-coded probabilities.

## Repository Layout
- src/: core library
  - preprocessing.py: load PhysioNet records, pad/truncate to MAXLEN
  - ecg_predict.py: batched model inference
  - selection.py: pick high-confidence examples per diagnosis (with duration filter)
  - ecg_lime.py, ecg_timeshap.py: event + per-lead attributions
  - explainer.py: class-aware pipelines (LIME-only or fused LIME+TimeSHAP)
  - fusion.py: merge two payloads lead-wise and time-wise
  - payload.py: convert rows -> plotting payloads
  - plot.py: render ECG plus shaded spans using ecg_plot
  - eval.py: ground-truth windows, AttAUC/F1, deletion AUC
  - config.py, config_targets.py: global constants and meta-class definitions
- data/: example metadata (ecg_model_pred_data.csv, snomed_classes.npy)
- model/: example Keras model weights (resnet_final.keras, resnet_final_weights.h5)
- notebook/ecg-xai.ipynb: interactive walkthrough/staging area
- README.md: (this file)

## Requirements
- Python >= 3.10
- Packages: numpy, pandas, scipy, scikit-learn, tensorflow/keras, tqdm, matplotlib, ecg_plot
- Data format: PhysioNet-style .mat with val key (12xT) plus matching .hea
- Defaults: MAXLEN=5000, RANDOM_SEED=42, DATA_ROOT='C:/data/'

Install (example):
```bash
python -m venv .venv
.\.venv\Scripts\activate   # on Windows; use source .venv/bin/activate on *nix
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn tensorflow keras tqdm matplotlib ecg_plot
```

## Quickstart Workflow
1) Load class names and model
```python
import numpy as np
from tensorflow import keras

class_names = np.load("data/snomed_classes.npy", allow_pickle=True)
model = keras.models.load_model("model/resnet_final.keras")
```

2) Predict on ECGs
```python
from ecg_predict import batched_predict_all

ecg_files = [...]  # list of .mat paths
probs = batched_predict_all(model, ecg_files, batch_size=16, maxlen=5000)
```

3) Select examples per target diagnosis
```python
from selection import build_selection_df_with_aliases
from config_targets import TARGET_META  # meta-classes and aliases

sel_df = build_selection_df_with_aliases(
    ecg_filenames=ecg_files,
    probs=probs,
    class_names=class_names,
    target_meta=TARGET_META,   # e.g., AF group
    k_per_class=5,
    min_prob=0.85,
    max_duration_sec=20.0,
    duration_cache_path="durations.npy",
)
```
sel_df needs at least group_class (target code), filename (.mat path), and sel_idx (index).

4) Run explanations
- LIME-only:
```python
from explainer import run_pipeline_for_classes

target_classes = list(TARGET_META.keys())  # e.g., ["164889003"]
payloads_by_class, df_lime = run_pipeline_for_classes(
    target_classes,
    sel_df,
    model=model,
    class_names=class_names,
    max_examples_per_class=3,
    plot=False,
)
```
- Fused LIME + TimeSHAP:
```python
from explainer import run_fused_pipeline_for_classes

fused_payloads, df_lime, df_ts = run_fused_pipeline_for_classes(
    target_classes,
    sel_df,
    model=model,
    class_names=class_names,
    max_examples_per_class=3,
    plot=False,
)
```
Each payload is keyed by val_idx/sel_idx and stores time/lead spans plus scores.

5) Plot a payload
```python
from plot import plot_from_payload

payload = list(fused_payloads["164889003"].values())[0]
plot_from_payload(payload, topk=5, show_all_leads=False)
```

6) Evaluate explanations
```python
from eval import evaluate_all_payloads

df_metrics = evaluate_all_payloads(
    fused_payloads,
    method_label="fused",
    model=model,
    class_names=class_names,
    debug=False,
)
```
Metrics reported per ECG: strict/lenient AttAUC, strict/lenient F1, deletion AUC, faithfulness gain, token count.

## Payload Format (plotting and evaluation)
A payload dict contains:
- mat_path: path to the ECG record (.mat)
- target_label: text for titles (for example, "atrial fibrillation")
- method_label: "LIME", "TimeSHAP", or "fused"
- page_seconds: duration plotted
- perlead_spans: {lead: [(start_sec, end_sec, weight), ...]}
- lead_scores: optional aggregate per lead
- top5_leads: ordered lead list used for plotting focus

payload_from_lime_row / payload_from_timeshap_row build these from the per-ECG rows returned by the explainers. fuse_lime_timeshap_payload merges two payloads with overlap-aware weighting (agg, beta, tau, sign_policy, method_weights, etc.).

## Customization
- Lead priors and windowing: tweak explainer.py (LEAD_PRIOR_BY_SNOMED, WINDOW_SEC_BY_SNOMED, BASE_WINDOW_SEC). Lower window_sec to focus on PR/QRS; raise to cover longer beats.
- Event segmentation: ecg_lime.make_event_segments currently supports uniform windows; extend for beat/QRS families if needed.
- Class registry and aliases: edit config_targets.py to add meta-classes and SNOMED aliases; expand TARGET_META and the evaluator REGISTRY (in eval.py) if you add diagnoses.
- Preprocessing: adjust MAXLEN/DATA_ROOT in config.py to match your model training setup.
- Model I/O: preprocess_for_model pads/truncates to (T=MAXLEN, F=12) using keras.preprocessing.sequence.pad_sequences; replace if your model expects a different input format.

## Data Notes
- Expects 12-lead signals with sampling frequency in the .hea first line; falls back to 500 Hz if missing.
- selection.build_selection_df_with_aliases can enforce a duration cap via the header nsamp/fs.
- Example metadata (data/ecg_model_pred_data.csv) shows the expected columns (group_class, sel_idx, filename, prob_group_class, etc.).

## Evaluation Details
- Uses heart-rate-adaptive diagnostic windows around detected R-peaks (BeatWindows).
- Per-class ground truth windows/leads live in eval.REGISTRY; strict vs lenient lead sets are supported.
- AttAUC: ranking-based area under attention-vs-ground-truth curve (strict/lenient by lead set).
- F1: thresholded token-level precision/recall.
- Deletion AUC: faithfulness metric by iteratively masking high-importance spans and re-predicting.

## Troubleshooting
- Missing or mismatched .hea files -> defaults to 500 Hz and may skip duration filtering.
- Low-segmentation density -> increase m_event / m_feat (more masks) or raise topk_events.
- Plotting issues -> ensure ecg_plot is installed and the .mat contains a val key shaped (12, T).

## Notebook
Use notebook/ecg-xai.ipynb to prototype end-to-end: load a model, run selection, generate explanations, plot, and score metrics.
# ECG-XAI

Explainability toolkit for 12-lead ECG classifiers. The repo bundles LIME-style perturbation explanations, a TimeSHAP-inspired variant, fusion logic to combine them, plotting utilities, and automatic evaluation metrics (token-level AttAUC/F1 and deletion-based faithfulness). It targets PhysioNet-style ECG records (.mat/.hea) and Keras/TF models that output SNOMED-coded probabilities.

## Repository Layout
- src/: core library
  - preprocessing.py: load PhysioNet records, pad/truncate to MAXLEN
  - ecg_predict.py: batched model inference
  - selection.py: pick high-confidence examples per diagnosis (with duration filter)
  - ecg_lime.py, ecg_timeshap.py: event + per-lead attributions
  - explainer.py: class-aware pipelines (LIME-only or fused LIME+TimeSHAP)
  - fusion.py: merge two payloads lead-wise and time-wise
  - payload.py: convert rows -> plotting payloads
  - plot.py: render ECG plus shaded spans using ecg_plot
  - eval.py: ground-truth windows, AttAUC/F1, deletion AUC
  - config.py, config_targets.py: global constants and meta-class definitions
- data/: example metadata (ecg_model_pred_data.csv, snomed_classes.npy)
- model/: example Keras model weights (resnet_final.keras, resnet_final_weights.h5)
- notebook/ecg-xai.ipynb: interactive walkthrough/staging area
- README.md: (this file)

## Requirements
- Python >= 3.10
- Packages: numpy, pandas, scipy, scikit-learn, tensorflow/keras, tqdm, matplotlib, ecg_plot
- Data format: PhysioNet-style .mat with val key (12xT) plus matching .hea
- Defaults: MAXLEN=5000, RANDOM_SEED=42, DATA_ROOT='C:/data/'

Install (example):
```bash
python -m venv .venv
.\.venv\Scripts\activate   # on Windows; use source .venv/bin/activate on *nix
pip install --upgrade pip
pip install numpy pandas scipy scikit-learn tensorflow keras tqdm matplotlib ecg_plot
```

## Quickstart Workflow
1) Load class names and model
```python
import numpy as np
from tensorflow import keras

class_names = np.load("data/snomed_classes.npy", allow_pickle=True)
model = keras.models.load_model("model/resnet_final.keras")
```

2) Predict on ECGs
```python
from ecg_predict import batched_predict_all

ecg_files = [...]  # list of .mat paths
probs = batched_predict_all(model, ecg_files, batch_size=16, maxlen=5000)
```

3) Select examples per target diagnosis
```python
from selection import build_selection_df_with_aliases
from config_targets import TARGET_META  # meta-classes and aliases

sel_df = build_selection_df_with_aliases(
    ecg_filenames=ecg_files,
    probs=probs,
    class_names=class_names,
    target_meta=TARGET_META,   # e.g., AF group
    k_per_class=5,
    min_prob=0.85,
    max_duration_sec=20.0,
    duration_cache_path="durations.npy",
)
```
sel_df needs at least group_class (target code), filename (.mat path), and sel_idx (index).

4) Run explanations
- LIME-only:
```python
from explainer import run_pipeline_for_classes

target_classes = list(TARGET_META.keys())  # e.g., ["164889003"]
payloads_by_class, df_lime = run_pipeline_for_classes(
    target_classes,
    sel_df,
    model=model,
    class_names=class_names,
    max_examples_per_class=3,
    plot=False,
)
```
- Fused LIME + TimeSHAP:
```python
from explainer import run_fused_pipeline_for_classes

fused_payloads, df_lime, df_ts = run_fused_pipeline_for_classes(
    target_classes,
    sel_df,
    model=model,
    class_names=class_names,
    max_examples_per_class=3,
    plot=False,
)
```
Each payload is keyed by val_idx/sel_idx and stores time/lead spans plus scores.

5) Plot a payload
```python
from plot import plot_from_payload

payload = list(fused_payloads["164889003"].values())[0]
plot_from_payload(payload, topk=5, show_all_leads=False)
```

6) Evaluate explanations
```python
from eval import evaluate_all_payloads

df_metrics = evaluate_all_payloads(
    fused_payloads,
    method_label="fused",
    model=model,
    class_names=class_names,
    debug=False,
)
```
Metrics reported per ECG: strict/lenient AttAUC, strict/lenient F1, deletion AUC, faithfulness gain, token count.

## Payload Format (plotting and evaluation)
A payload dict contains:
- mat_path: path to the ECG record (.mat)
- target_label: text for titles (for example, "atrial fibrillation")
- method_label: "LIME", "TimeSHAP", or "fused"
- page_seconds: duration plotted
- perlead_spans: {lead: [(start_sec, end_sec, weight), ...]}
- lead_scores: optional aggregate per lead
- top5_leads: ordered lead list used for plotting focus

payload_from_lime_row / payload_from_timeshap_row build these from the per-ECG rows returned by the explainers. fuse_lime_timeshap_payload merges two payloads with overlap-aware weighting (agg, beta, tau, sign_policy, method_weights, etc.).

## Customization
- Lead priors and windowing: tweak explainer.py (LEAD_PRIOR_BY_SNOMED, WINDOW_SEC_BY_SNOMED, BASE_WINDOW_SEC). Lower window_sec to focus on PR/QRS; raise to cover longer beats.
- Event segmentation: ecg_lime.make_event_segments currently supports uniform windows; extend for beat/QRS families if needed.
- Class registry and aliases: edit config_targets.py to add meta-classes and SNOMED aliases; expand TARGET_META and the evaluator REGISTRY (in eval.py) if you add diagnoses.
- Preprocessing: adjust MAXLEN/DATA_ROOT in config.py to match your model training setup.
- Model I/O: preprocess_for_model pads/truncates to (T=MAXLEN, F=12) using keras.preprocessing.sequence.pad_sequences; replace if your model expects a different input format.

## Data Notes
- Expects 12-lead signals with sampling frequency in the .hea first line; falls back to 500 Hz if missing.
- selection.build_selection_df_with_aliases can enforce a duration cap via the header nsamp/fs.
- Example metadata (data/ecg_model_pred_data.csv) shows the expected columns (group_class, sel_idx, filename, prob_group_class, etc.).

## Evaluation Details
- Uses heart-rate-adaptive diagnostic windows around detected R-peaks (BeatWindows).
- Per-class ground truth windows/leads live in eval.REGISTRY; strict vs lenient lead sets are supported.
- AttAUC: ranking-based area under attention-vs-ground-truth curve (strict/lenient by lead set).
- F1: thresholded token-level precision/recall.
- Deletion AUC: faithfulness metric by iteratively masking high-importance spans and re-predicting.

## Troubleshooting
- Missing or mismatched .hea files -> defaults to 500 Hz and may skip duration filtering.
- Low-segmentation density -> increase m_event / m_feat (more masks) or raise topk_events.
- Plotting issues -> ensure ecg_plot is installed and the .mat contains a val key shaped (12, T).

## Notebook
Use notebook/ecg-xai.ipynb to prototype end-to-end: load a model, run selection, generate explanations, plot, and score metrics.
