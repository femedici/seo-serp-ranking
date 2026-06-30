"""
Testes basicos do pipeline (pytest).

Cobrem: integridade do dataset conforme a especificacao, ausencia de vazamento
no pipeline (scaler dentro do Pipeline) e sanidade dos modelos (campeao supera
o baseline em PR-AUC numa CV curta).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline

from src import config
from src.data_loader import load_dataset, split_xy, dataset_summary
from src.preprocessing import build_pipeline
from src.models import _logreg, _rf


@pytest.fixture(scope="module")
def df():
    return load_dataset()


def test_dataset_shape(df):
    assert df.shape == (500, 14)


def test_expected_columns(df):
    expected = set(config.all_features() + [config.TARGET])
    assert set(df.columns) == expected
    assert len(config.all_features()) == 13


def test_no_missing(df):
    assert int(df.isna().sum().sum()) == 0


def test_target_distribution(df):
    s = dataset_summary(df)
    assert s["class_0"] == 312
    assert s["class_1"] == 188
    assert abs(s["positive_rate"] - 0.376) < 1e-6


def test_binary_columns(df):
    for col in config.BINARY_FEATURES + [config.TARGET]:
        assert set(df[col].unique()).issubset({0, 1})


def test_weak_linear_correlation(df):
    # Especificacao: correlacoes lineares fracas com o alvo
    corr = df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET).abs()
    assert corr.max() < 0.30


def test_pipeline_has_scaler():
    pipe = build_pipeline(_logreg(), needs_scaling=True)
    assert isinstance(pipe, SkPipeline)
    assert "scaler" in dict(pipe.steps)


def test_pipeline_no_scaler_for_trees():
    pipe = build_pipeline(_rf(), needs_scaling=False)
    assert "scaler" not in dict(pipe.steps)


def test_model_beats_baseline(df):
    X, y = split_xy(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)
    rf = build_pipeline(_rf(), needs_scaling=False)
    pr_auc = cross_val_score(rf, X, y, scoring="average_precision", cv=cv).mean()
    # PR-AUC do baseline equivale a prevalencia da classe positiva (~0.376)
    assert pr_auc >= y.mean()
