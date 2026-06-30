"""
Interpretabilidade e importancia de atributos (Materiais e Metodos, secao 4).

Reporta-se a importancia por PERMUTACAO (model-agnostica, menos enviesada),
complementada por informacao mutua e pela importancia nativa de arvore quando
disponivel. A importancia e ainda agregada por grupo de fatores (on-page,
autoridade, comportamental, contextual), conectando-se a H2 e H5. Inclui a
analise de ablacao do confundidor ``serp_position_before`` (lacuna D7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config
from .evaluation import make_repeated_cv, SCORING


def permutation_importances(model, X, y, n_splits: int = 5) -> pd.DataFrame:
    """
    Importancia por permutacao AGREGADA sobre folds de CV estratificada, mais
    robusta que o calculo em um unico holdout. Reporta media e desvio entre
    folds (o desvio mede a estabilidade do ranking de importancia, lacuna D6).
    PR-AUC e a metrica de referencia.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    per_fold = []
    for tr_idx, te_idx in cv.split(X, y):
        est = clone(model)
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        result = permutation_importance(
            est, X.iloc[te_idx], y.iloc[te_idx],
            scoring="average_precision",
            n_repeats=20,
            random_state=config.SEED,
            n_jobs=-1,
        )
        per_fold.append(result.importances_mean)
    arr = np.vstack(per_fold)  # folds x features
    return (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": arr.mean(axis=0),
                "std": arr.std(axis=0),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def mutual_information(X, y) -> pd.DataFrame:
    """Informacao mutua (captura relacao nao linear)."""
    discrete = [X.columns.get_loc(c) for c in config.BINARY_FEATURES]
    mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=config.SEED)
    return (
        pd.DataFrame({"feature": X.columns, "mutual_info": mi})
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )


def group_importance(imp_df: pd.DataFrame) -> pd.DataFrame:
    """Soma a importancia por grupo de fatores de SEO."""
    feat_to_group = {}
    for group, cols in config.FEATURE_GROUPS.items():
        for c in cols:
            feat_to_group[c] = group
    df = imp_df.copy()
    df["grupo"] = df["feature"].map(feat_to_group)
    agg = (
        df.groupby("grupo")["importance"].sum().sort_values(ascending=False)
        .rename("importancia_total")
        .reset_index()
    )
    return agg


def ablation_serp_before(model_factory, X, y) -> pd.DataFrame:
    """
    Ablacao do confundidor com efeito-teto ``serp_position_before``.
    Compara a PR-AUC (CV repetida) com e sem a feature.
    """
    cv = make_repeated_cv()
    rows = []
    for label, cols in [
        ("Com serp_position_before", list(X.columns)),
        ("Sem serp_position_before", [c for c in X.columns if c != config.LEAKAGE_CANDIDATE]),
    ]:
        res = cross_validate(
            model_factory(), X[cols], y, scoring=SCORING, cv=cv, n_jobs=-1
        )
        rows.append(
            {
                "Cenario": label,
                "PR-AUC": res["test_pr_auc"].mean(),
                "PR-AUC dp": res["test_pr_auc"].std(),
                "F1-macro": res["test_f1_macro"].mean(),
                "Balanced Acc.": res["test_balanced_acc"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("Cenario")
