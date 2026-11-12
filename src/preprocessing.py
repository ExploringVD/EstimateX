"""Shared helpers: one-hot encoding, scaling, and saving/loading fitted artifacts.

Pass a fitted encoder/scaler to transform only (the app); omit it to fit a new one (training).
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def scale_numeric_features(
    df: pd.DataFrame,
    columns: list[str],
    scaler: Optional[StandardScaler] = None,
) -> tuple[pd.DataFrame, StandardScaler]:
    df = df.copy()
    if scaler is None:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df[columns] = scaler.transform(df[columns])
    return df, scaler


def encode_onehot(
    df: pd.DataFrame,
    columns: list[str],
    encoder: Optional[OneHotEncoder] = None,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """Replaces `columns` with one-hot columns; unseen categories encode as all zeros."""
    df = df.copy()
    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[columns])
    else:
        encoded = encoder.transform(df[columns])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns), index=df.index)
    return pd.concat([df.drop(columns=columns), encoded_df], axis=1), encoder


def save_artifact(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str):
    return joblib.load(path)
