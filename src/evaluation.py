"""
Protocolo de avaliacao: metricas, validacao cruzada repetida, CV aninhada e
holdout de confirmacao (Materiais e Metodos, secao 4).

Metrica primaria: PR-AUC (average_precision), foco na classe minoritaria
ranking_improved=1. Reportam-se ainda F1-macro, Balanced Accuracy, ROC-AUC,
Precision/Recall/F1 da classe 1 e acuracia. Acuracia isolada nao e criterio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    make_scorer,
)

from . import config


# Conjunto de scorers (PR-AUC primaria)
SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "f1_macro": "f1_macro",
    "balanced_acc": "balanced_accuracy",
    "precision_pos": make_scorer(precision_score, pos_label=1, zero_division=0),
    "recall_pos": make_scorer(recall_score, pos_label=1, zero_division=0),
    "f1_pos": make_scorer(f1_score, pos_label=1, zero_division=0),
    "accuracy": "accuracy",
}

METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "f1_macro": "F1-macro",
    "balanced_acc": "Balanced Acc.",
    "precision_pos": "Precision (1)",
    "recall_pos": "Recall (1)",
    "f1_pos": "F1 (1)",
    "accuracy": "Acuracia",
}


def make_repeated_cv() -> RepeatedStratifiedKFold:
    """CV repetida estratificada (5-fold x 3 repeticoes = 15 medidas por modelo)."""
    return RepeatedStratifiedKFold(
        n_splits=config.CV_OUTER_SPLITS,
        n_repeats=config.CV_REPEATS,
        random_state=config.SEED,
    )


def evaluate_models(models: dict, X, y) -> tuple[pd.DataFrame, dict]:
    """
    Avalia todos os modelos na MESMA particao repetida (folds alinhados), o que
    permite testes estatisticos pareados. Retorna:
      - resumo (media +/- desvio por metrica e modelo)
      - per_fold: dict modelo -> array de PR-AUC por fold (para Friedman/Nemenyi)
    """
    cv = make_repeated_cv()
    summary_rows = []
    per_fold_prauc = {}

    for name, model in models.items():
        res = cross_validate(
            model, X, y, scoring=SCORING, cv=cv, n_jobs=-1, return_train_score=False
        )
        per_fold_prauc[name] = res["test_pr_auc"]
        row = {"Modelo": name}
        for key in SCORING:
            scores = res[f"test_{key}"]
            row[METRIC_LABELS[key]] = scores.mean()
            row[METRIC_LABELS[key] + " (dp)"] = scores.std()
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("Modelo")
    return summary, per_fold_prauc


def nested_cv(search_spaces: dict, X, y) -> pd.DataFrame:
    """
    CV aninhada estratificada: externa 5-fold (estimativa nao enviesada) +
    interna 3-fold (RandomizedSearchCV). Retorna media/dp da PR-AUC e F1-macro
    da CV externa por modelo, alem dos melhores hiperparametros por fold.
    """
    outer = StratifiedKFold(
        n_splits=config.CV_OUTER_SPLITS, shuffle=True, random_state=config.SEED
    )
    rows = []
    for name, spec in search_spaces.items():
        fold_prauc, fold_f1 = [], []
        for tr_idx, te_idx in outer.split(X, y):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            search = RandomizedSearchCV(
                spec["estimator"],
                spec["params"],
                n_iter=config.SEARCH_ITER,
                scoring="average_precision",
                cv=StratifiedKFold(
                    n_splits=config.CV_INNER_SPLITS, shuffle=True, random_state=config.SEED
                ),
                random_state=config.SEED,
                n_jobs=-1,
            )
            search.fit(X_tr, y_tr)
            best = search.best_estimator_
            proba = best.predict_proba(X_te)[:, 1]
            pred = best.predict(X_te)
            fold_prauc.append(average_precision_score(y_te, proba))
            fold_f1.append(f1_score(y_te, pred, average="macro"))
        rows.append(
            {
                "Modelo": name,
                "PR-AUC (aninhada)": np.mean(fold_prauc),
                "PR-AUC dp": np.std(fold_prauc),
                "F1-macro (aninhada)": np.mean(fold_f1),
                "F1-macro dp": np.std(fold_f1),
            }
        )
    return pd.DataFrame(rows).set_index("Modelo")


def holdout_evaluation(model, X, y):
    """
    Treina no conjunto de treino (80%) e avalia no holdout estratificado (20%).
    Retorna metricas, matriz de confusao, relatorio e curvas (y_true, proba).
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.HOLDOUT_SIZE, stratify=y, random_state=config.SEED
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)

    metrics = {
        "PR-AUC": average_precision_score(y_te, proba),
        "ROC-AUC": roc_auc_score(y_te, proba),
        "F1-macro": f1_score(y_te, pred, average="macro"),
        "Balanced Acc.": balanced_accuracy_score(y_te, pred),
        "Precision (1)": precision_score(y_te, pred, pos_label=1, zero_division=0),
        "Recall (1)": recall_score(y_te, pred, pos_label=1, zero_division=0),
        "F1 (1)": f1_score(y_te, pred, pos_label=1, zero_division=0),
        "Acuracia": accuracy_score(y_te, pred),
    }
    cm = confusion_matrix(y_te, pred)
    report = classification_report(y_te, pred, digits=3, zero_division=0)
    return metrics, cm, report, (y_te.to_numpy(), proba)


def champion_name(summary: pd.DataFrame) -> str:
    """Campeao = maior PR-AUC media; desempate por F1-macro (exclui o Dummy)."""
    candidates = summary.drop(index=[i for i in summary.index if "Dummy" in i])
    ranked = candidates.sort_values(
        by=["PR-AUC", "F1-macro"], ascending=False
    )
    return ranked.index[0]
