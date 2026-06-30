"""
Comparacao estatistica entre modelos (Materiais e Metodos, secao 4).

  * Friedman: testa se ha diferenca global entre os modelos (PR-AUC por fold);
  * Nemenyi: pos-teste pareado quando Friedman e significativo;
  * Wilcoxon signed-rank: cada modelo contra o baseline (Dummy), pareado por fold.

Nivel de significancia alpha = 0,05.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
import scikit_posthocs as sp

from . import config


def friedman_test(per_fold_prauc: dict) -> dict:
    """Aplica o teste de Friedman sobre a PR-AUC por fold de todos os modelos."""
    names = list(per_fold_prauc.keys())
    matrix = np.column_stack([per_fold_prauc[n] for n in names])  # folds x modelos
    stat, p = friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
    return {"statistic": float(stat), "p_value": float(p), "models": names, "matrix": matrix}


def nemenyi_test(friedman_result: dict) -> pd.DataFrame:
    """Pos-teste de Nemenyi (matriz de p-valores pareados)."""
    matrix = friedman_result["matrix"]
    names = friedman_result["models"]
    pvals = sp.posthoc_nemenyi_friedman(matrix)
    pvals.index = names
    pvals.columns = names
    return pvals


def average_ranks(friedman_result: dict) -> pd.Series:
    """Rank medio de cada modelo (1 = melhor) ao longo dos folds, por PR-AUC."""
    matrix = friedman_result["matrix"]
    names = friedman_result["models"]
    # ranks por fold: maior PR-AUC recebe rank 1
    ranks = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        order = (-matrix[i]).argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, matrix.shape[1] + 1)
        ranks[i] = r
    return pd.Series(ranks.mean(axis=0), index=names).sort_values()


def wilcoxon_vs_baseline(per_fold_prauc: dict, baseline_key: str) -> pd.DataFrame:
    """Wilcoxon signed-rank de cada modelo contra o baseline, pareado por fold."""
    base = per_fold_prauc[baseline_key]
    rows = []
    for name, scores in per_fold_prauc.items():
        if name == baseline_key:
            continue
        diff = np.array(scores) - np.array(base)
        if np.allclose(diff, 0):
            stat, p = np.nan, 1.0
        else:
            stat, p = wilcoxon(scores, base)
        rows.append(
            {
                "Modelo": name,
                "PR-AUC media": float(np.mean(scores)),
                "Delta vs baseline": float(np.mean(diff)),
                "Wilcoxon p": float(p),
                "Significativo (a=0.05)": bool(p < config.ALPHA),
            }
        )
    return pd.DataFrame(rows).set_index("Modelo")
