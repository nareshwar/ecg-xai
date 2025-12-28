"""ecgxai.ecg_predict

Batch prediction utilities for ECG classification models.

This module provides a thin convenience wrapper around:
- preprocessing.preprocess_for_model(...) -> (T, F)
- model.predict(...) -> (B, C)

Conventions:
- Input records are file paths (e.g., .mat). Each record is preprocessed into a fixed-length tensor of shape (T=maxlen, F=num_leads).
- Output is a numpy array of shape (N, C) where C is the number of classes.

Typical usage:
    probs = batched_predict_all(model, ecg_files, maxlen=5000, batch_size=32)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Union, Optional

import numpy as np
from tqdm import tqdm

from .preprocessing import preprocess_for_model

PathLike = Union[str, Path]

def _infer_num_classes(model: Any) -> Optional[int]:
    """Best-effort inference of number of output classes from a Keras-like model."""
    out_shape = getattr(model, "output_shape", None)
    # Keras can return (None, C) or a list/tuple of shapes for multi-output models.
    try:
        if isinstance(out_shape, (list, tuple)) and out_shape and isinstance(out_shape[0], (list, tuple)):
            # Multi-output: not supported by this helper
            return None
        if isinstance(out_shape, (list, tuple)) and len(out_shape) >= 2:
            c = out_shape[-1]
            return int(c) if c is not None else None
    except Exception:
        return None
    return None

def batched_predict_all(
    model: Any,
    ecg_filenames: Sequence[PathLike],
    *,
    maxlen: int = 5000,
    batch_size: int = 32,
    show_progress: bool = True,
    desc: str = "Predicting ECGs",
) -> np.ndarray:
    """Run model predictions over a list of ECG files.

    Args:
        model: Keras/TF-like model with `predict(X, verbose=0) -> (B, C)`.
        ecg_filenames: Sequence of ECG file paths (e.g., .mat).
        maxlen: Fixed time length used in `preprocess_for_model`.
        batch_size: Number of ECGs per batch.
        show_progress: If True, show a tqdm progress bar.
        desc: Label for the progress bar.

    Returns:
        probs: Float32 array of shape (N, C) with class probabilities.

    Raises:
        ValueError: If maxlen/batch_size are invalid.
        RuntimeError: If preprocessing fails for a specific file.
    """
    if maxlen <= 0:
        raise ValueError(f"maxlen must be > 0, got {maxlen}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    files = list(ecg_filenames)
    n = len(files)

    if n == 0:
        c = _infer_num_classes(model)
        return np.zeros((0, c if c is not None else 0), dtype=np.float32)

    all_probs: list[np.ndarray] = []

    indices = range(0, n, batch_size)
    if show_progress:
        indices = tqdm(indices, desc=desc)

    for start in indices:
        end = min(n, start + batch_size)
        batch_files = files[start:end]

        # Preprocess each file into (T, F) and stack -> (B, T, F)
        X_list = []
        for f in batch_files:
            try:
                x = preprocess_for_model(f, maxlen=maxlen)
            except Exception as e:
                raise RuntimeError(f"preprocess_for_model failed for file: {f}") from e
            X_list.append(x)

        X_batch = np.stack(X_list, axis=0).astype(np.float32, copy=False)

        # Predict -> (B, C)
        probs_batch = model.predict(X_batch, verbose=0)
        probs_batch = np.asarray(probs_batch, dtype=np.float32)

        all_probs.append(probs_batch)

    probs = np.vstack(all_probs).astype(np.float32, copy=False)  # (N, C)
    return probs
