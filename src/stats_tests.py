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


def _holm_adjust(pvalues: list[float]) -> list[float]:
    """Correção de Holm-Bonferroni para múltiplas comparações."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvalues[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return adj.tolist()


def wilcoxon_vs_baseline(per_fold_prauc: dict, baseline_key: str) -> pd.DataFrame:
    """
    Wilcoxon signed-rank de cada modelo contra o baseline, pareado por fold,
    com correção de Holm-Bonferroni para o conjunto de comparações.
    """
    base = per_fold_prauc[baseline_key]
    names, means, deltas, pvals = [], [], [], []
    for name, scores in per_fold_prauc.items():
        if name == baseline_key:
            continue
        diff = np.array(scores) - np.array(base)
        if np.allclose(diff, 0):
            p = 1.0
        else:
            _, p = wilcoxon(scores, base)
        names.append(name)
        means.append(float(np.mean(scores)))
        deltas.append(float(np.mean(diff)))
        pvals.append(float(p))

    holm = _holm_adjust(pvals)
    df = pd.DataFrame(
        {
            "Modelo": names,
            "PR-AUC media": means,
            "Delta vs baseline": deltas,
            "Wilcoxon p": pvals,
            "Holm p": holm,
            "Significativo Holm (a=0.05)": [bool(h < config.ALPHA) for h in holm],
        }
    ).set_index("Modelo")
    return df.sort_values("Wilcoxon p")
