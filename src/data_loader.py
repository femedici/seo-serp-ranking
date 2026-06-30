"""Carregamento e validacao do conjunto de dados ``seo_dataset.csv``."""

from __future__ import annotations

import pandas as pd

from . import config


EXPECTED_COLUMNS = config.all_features() + [config.TARGET]


def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Carrega o CSV e valida o schema esperado (14 colunas, sem ausentes)."""
    path = path or config.DATA_PATH
    df = pd.read_csv(path)

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Colunas ausentes no dataset: {sorted(missing_cols)}")

    # Reordena para a ordem canonica (grupos de fatores + alvo)
    df = df[EXPECTED_COLUMNS].copy()
    return df


def split_xy(df: pd.DataFrame):
    """Separa preditoras (X) e alvo (y)."""
    X = df[config.all_features()].copy()
    y = df[config.TARGET].astype(int).copy()
    return X, y


def dataset_summary(df: pd.DataFrame) -> dict:
    """Resumo descritivo usado na EDA e nos testes."""
    counts = df[config.TARGET].value_counts().sort_index()
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "n_features": len(config.all_features()),
        "missing_total": int(df.isna().sum().sum()),
        "class_0": int(counts.get(0, 0)),
        "class_1": int(counts.get(1, 0)),
        "positive_rate": float(counts.get(1, 0) / df.shape[0]),
    }
