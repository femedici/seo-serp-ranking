"""
Definicao dos modelos, espacos de busca de hiperparametros e combinacoes.

Progressao recomendada (Materiais e Metodos, secao 3):
baseline -> linear -> arvore -> ensemble, cobrindo familias distintas para
testar H1 (ensembles superam lineares). O desbalanceamento leve (62/38) e
tratado por class_weight='balanced' (modelos sklearn) e scale_pos_weight
(XGBoost), sem reamostragem sintetica na configuracao principal.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from . import config
from .preprocessing import build_pipeline

# Razao de classes para tratar desbalanceamento no XGBoost (~312/188)
SCALE_POS_WEIGHT = 312 / 188


def _logreg() -> LogisticRegression:
    return LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=config.SEED
    )


def _rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=config.SEED,
        n_jobs=-1,
    )


def _xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=SCALE_POS_WEIGHT,
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.SEED,
        n_jobs=-1,
    )


def _svm() -> SVC:
    return SVC(
        kernel="rbf", probability=True, class_weight="balanced", random_state=config.SEED
    )


def _knn() -> KNeighborsClassifier:
    return KNeighborsClassifier(n_neighbors=15)


def get_models() -> dict:
    """
    Dicionario nome -> Pipeline pronto para CV. Cada modelo ja embute o
    escalonamento adequado a sua familia (anti-vazamento por fold).
    """
    voting = VotingClassifier(
        estimators=[
            ("logreg", build_pipeline(_logreg(), needs_scaling=True)),
            ("rf", _rf()),
            ("xgb", _xgb()),
        ],
        voting="soft",
        n_jobs=-1,
    )
    stacking = StackingClassifier(
        estimators=[
            ("rf", _rf()),
            ("xgb", _xgb()),
            ("svm", build_pipeline(_svm(), needs_scaling=True)),
        ],
        final_estimator=LogisticRegression(max_iter=5000, random_state=config.SEED),
        cv=config.CV_INNER_SPLITS,
        n_jobs=-1,
    )

    return {
        "Dummy (majoritária)": build_pipeline(
            DummyClassifier(strategy="most_frequent"), needs_scaling=False
        ),
        "Regressão Logística": build_pipeline(_logreg(), needs_scaling=True),
        "k-NN": build_pipeline(_knn(), needs_scaling=True),
        "SVM-RBF": build_pipeline(_svm(), needs_scaling=True),
        "Random Forest": build_pipeline(_rf(), needs_scaling=False),
        "XGBoost": build_pipeline(_xgb(), needs_scaling=False),
        "Soft Voting": voting,
        "Stacking": stacking,
    }


def get_search_spaces() -> dict:
    """
    Espacos de busca (RandomizedSearchCV) para a CV aninhada dos modelos
    principais. Chaves usam o prefixo 'model__' do passo do Pipeline.
    """
    return {
        "Regressão Logística": {
            "estimator": build_pipeline(_logreg(), needs_scaling=True),
            "params": {
                "model__C": loguniform(1e-3, 1e3),
                "model__penalty": ["l1", "l2"],
                "model__solver": ["liblinear", "saga"],
            },
        },
        "Random Forest": {
            "estimator": build_pipeline(_rf(), needs_scaling=False),
            "params": {
                "model__n_estimators": randint(200, 800),
                "model__max_depth": [None, 5, 10, 20],
                "model__min_samples_leaf": [1, 2, 5, 10],
                "model__max_features": ["sqrt", "log2", 0.5],
            },
        },
        "XGBoost": {
            "estimator": build_pipeline(_xgb(), needs_scaling=False),
            "params": {
                "model__n_estimators": randint(200, 800),
                "model__max_depth": randint(3, 8),
                "model__learning_rate": loguniform(1e-2, 3e-1),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.6, 0.4),
                "model__min_child_weight": randint(1, 7),
                "model__reg_lambda": [0, 1, 5],
            },
        },
    }
