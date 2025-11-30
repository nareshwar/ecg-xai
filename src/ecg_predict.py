from __future__ import annotations
from typing import Sequence

import numpy as np
from tqdm import tqdm

from preprocessing import preprocess_for_model


def batched_predict_all(
    model,
    ecg_filenames: Sequence[str],
    *,
    maxlen: int = 5000,
    batch_size: int = 32,
    show_progress: bool = True,
    desc: str = "Predicting ECGs",
) -> np.ndarray:
    """
    Run model predictions over all ECGs in ecg_filenames.

    Args
    ----
    model          : Keras/TF model with .predict(X) -> (B, C)
    ecg_filenames  : iterable of .mat paths
    maxlen         : time length for preprocess_for_model
    batch_size     : number of ECGs per prediction batch
    show_progress  : if True, show a tqdm progress bar
    desc           : label for the progress bar

    Returns
    -------
    probs : (N, C) array of class probabilities.
    """
    ecg_filenames = np.asarray(ecg_filenames, dtype=object)
    N = len(ecg_filenames)

    if N == 0:
        return np.zeros((0, 0), dtype=np.float32)

    all_probs = []

    # create an index range then optionally wrap with tqdm
    indices = range(0, N, batch_size)
    if show_progress:
        indices = tqdm(indices, desc=desc)

    for start in indices:
        end = min(N, start + batch_size)
        batch_files = ecg_filenames[start:end]

        # Preprocess this batch
        X_batch = np.stack([
            preprocess_for_model(f, maxlen=maxlen)
            for f in batch_files
        ], axis=0)  # (B, T, F)

        # Predict
        probs_batch = model.predict(X_batch, verbose=0)
        all_probs.append(probs_batch)

    probs = np.vstack(all_probs)  # (N, C)
    return probs
