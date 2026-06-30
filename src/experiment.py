"""
Orquestrador dos experimentos. Executa o protocolo completo e salva tabelas
(CSV) e figuras (PNG) em ``results/`` para uso no artigo.

Execucao:
    python -m src.experiment
"""

from __future__ import annotations

import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from . import config
from .data_loader import load_dataset, split_xy, dataset_summary
from . import eda
from .models import get_models, get_search_spaces
from .evaluation import (
    evaluate_models,
    nested_cv,
    holdout_evaluation,
    champion_name,
)
from . import stats_tests
from .preprocessing import build_pipeline, build_resampling_pipeline
from .models import _logreg, _rf
from . import interpretability as interp

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def _save_table(df: pd.DataFrame, name: str, index: bool = True) -> None:
    path = os.path.join(config.TABLES_DIR, name)
    df.to_csv(path, index=index)
    print(f"  [tabela] {name}")


def _fig_path(name: str) -> str:
    return os.path.join(config.FIGURES_DIR, name)


def balancing_comparison(X, y) -> pd.DataFrame:
    """Compara class_weight='balanced' vs SMOTE para LogReg e RF (lacuna D3)."""
    from .evaluation import make_repeated_cv, SCORING
    from sklearn.model_selection import cross_validate

    cv = make_repeated_cv()
    configs = {
        "LogReg + class_weight": build_pipeline(_logreg(), needs_scaling=True),
        "LogReg + SMOTE": build_resampling_pipeline(
            _logreg().set_params(class_weight=None), needs_scaling=True
        ),
        "RF + class_weight": build_pipeline(_rf(), needs_scaling=False),
        "RF + SMOTE": build_resampling_pipeline(
            _rf().set_params(class_weight=None), needs_scaling=False
        ),
    }
    rows = []
    for name, model in configs.items():
        res = cross_validate(model, X, y, scoring=SCORING, cv=cv, n_jobs=-1)
        rows.append(
            {
                "Estrategia": name,
                "PR-AUC": res["test_pr_auc"].mean(),
                "Recall (1)": res["test_recall_pos"].mean(),
                "F1 (1)": res["test_f1_pos"].mean(),
                "F1-macro": res["test_f1_macro"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("Estrategia")


def plot_model_boxplot(per_fold_prauc: dict, path: str) -> None:
    order = sorted(per_fold_prauc, key=lambda k: np.mean(per_fold_prauc[k]))
    data = [per_fold_prauc[k] for k in order]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, vert=False, labels=order, showmeans=True)
    ax.set_xlabel("PR-AUC por fold (validação cruzada 5 x 3)")
    ax.set_title("Comparação de modelos: PR-AUC")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, title: str, path: str) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["0", "1"], yticklabels=["0", "1"], ax=ax)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(y_true, proba, title: str, path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(rec, prec)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, color="#4C72B0", label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_xlabel("Taxa de falsos positivos")
    axes[0].set_ylabel("Taxa de verdadeiros positivos")
    axes[0].set_title("Curva ROC")
    axes[0].legend(loc="lower right")
    axes[1].plot(rec, prec, color="#DD8452", label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].axhline(y_true.mean(), ls="--", color="gray",
                    label=f"Linha de base = {y_true.mean():.3f}")
    axes[1].set_xlabel("Revocação")
    axes[1].set_ylabel("Precisão")
    axes[1].set_title("Curva Precisão-Revocação")
    axes[1].legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(imp_df: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    d = imp_df.sort_values("importance")
    ax.barh(d["feature"], d["importance"], xerr=d["std"], color="#4C72B0")
    ax.set_xlabel("Importância por permutação (queda média de PR-AUC)")
    ax.set_title("Importância dos atributos: modelo selecionado")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_group_importance(group_df: pd.DataFrame, path: str) -> None:
    rotulos = {
        "behavioral": "Comportamental",
        "authority": "Autoridade",
        "contextual": "Contextual",
        "on_page": "On-page",
    }
    nomes = [rotulos.get(g, g) for g in group_df["grupo"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(nomes, group_df["importancia_total"],
           color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(group_df)])
    ax.set_ylabel("Importância total (informação mútua)")
    ax.set_title("Importância por grupo de fatores de SEO")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cd_diagram(ranks, nemenyi, path: str) -> None:
    try:
        import scikit_posthocs as sp
        fig, ax = plt.subplots(figsize=(9, 3))
        sp.critical_difference_diagram(ranks, nemenyi, ax=ax)
        ax.set_title("Diagrama de diferença crítica (Nemenyi)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("  [figura] critical_difference.png")
    except Exception as exc:  # pragma: no cover
        print(f"  [aviso] diagrama CD nao gerado: {exc}")


def main() -> None:
    config.ensure_dirs()
    np.random.seed(config.SEED)
    results_json: dict = {}

    print("== 1. Carga e EDA ==")
    df = load_dataset()
    summary = dataset_summary(df)
    results_json["dataset"] = summary
    _save_table(pd.DataFrame([summary]), "dataset_summary.csv", index=False)

    corr_tbl = eda.correlation_table(df)
    _save_table(corr_tbl, "target_correlation.csv")
    eda.plot_class_distribution(df, _fig_path("class_distribution.png"))
    eda.plot_correlation_heatmap(df, _fig_path("correlation_heatmap.png"))
    eda.plot_target_correlation(corr_tbl, _fig_path("target_correlation.png"))
    print("  [figuras] class_distribution / correlation_heatmap / target_correlation")

    X, y = split_xy(df)
    mi_tbl = interp.mutual_information(X, y)
    _save_table(mi_tbl, "mutual_information.csv", index=False)

    print("== 2. Comparacao de modelos (CV repetida 5x3) ==")
    models = get_models()
    model_summary, per_fold = evaluate_models(models, X, y)
    model_summary_round = model_summary.round(4)
    _save_table(model_summary_round, "model_comparison.csv")
    _save_table(pd.DataFrame(per_fold), "per_fold_prauc.csv", index=False)
    plot_model_boxplot(per_fold, _fig_path("model_comparison_prauc.png"))
    print("  [figura] model_comparison_prauc.png")
    print(model_summary_round[["PR-AUC", "F1-macro", "Balanced Acc.", "ROC-AUC", "Recall (1)"]].to_string())

    print("== 3. Comparacao de balanceamento (class_weight vs SMOTE) ==")
    bal = balancing_comparison(X, y).round(4)
    _save_table(bal, "balancing_comparison.csv")
    print(bal.to_string())

    print("== 4. Testes estatisticos ==")
    fr = stats_tests.friedman_test(per_fold)
    ranks = stats_tests.average_ranks(fr)
    nemenyi = stats_tests.nemenyi_test(fr)
    wilco = stats_tests.wilcoxon_vs_baseline(per_fold, baseline_key=[k for k in per_fold if "Dummy" in k][0])
    results_json["friedman"] = {"statistic": fr["statistic"], "p_value": fr["p_value"]}
    _save_table(ranks.to_frame("rank_medio"), "average_ranks.csv")
    _save_table(nemenyi.round(4), "nemenyi_pvalues.csv")
    _save_table(wilco.round(4), "wilcoxon_vs_baseline.csv")
    plot_cd_diagram(ranks, nemenyi, _fig_path("critical_difference.png"))
    print(f"  Friedman: chi2={fr['statistic']:.3f}, p={fr['p_value']:.4g}")
    print(wilco.round(4).to_string())

    print("== 5. CV aninhada (hiperparametros ajustados) ==")
    nested = nested_cv(get_search_spaces(), X, y).round(4)
    _save_table(nested, "nested_cv.csv")
    print(nested.to_string())

    print("== 6. Modelo selecionado e holdout de confirmacao ==")
    champ = champion_name(model_summary)
    results_json["champion"] = champ
    print(f"  Modelo selecionado (maior PR-AUC medio): {champ}")
    champ_model = clone(models[champ])
    metrics, cm, report, (y_true, proba) = holdout_evaluation(champ_model, X, y)
    results_json["holdout_metrics"] = {k: float(v) for k, v in metrics.items()}
    _save_table(pd.DataFrame([metrics], index=[champ]), "holdout_metrics.csv")
    plot_confusion(cm, f"Matriz de confusão: {champ}", _fig_path("confusion_matrix.png"))
    plot_curves(y_true, proba, f"Curvas: {champ}", _fig_path("roc_pr_curves.png"))
    with open(os.path.join(config.TABLES_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Modelo selecionado: {champ}\n\n{report}\n")
    print("  [figuras] confusion_matrix.png / roc_pr_curves.png")
    print(report)

    print("== 7. Interpretabilidade e ablacao ==")
    imp = interp.permutation_importances(clone(models[champ]), X, y)
    _save_table(imp.round(5), "permutation_importance.csv", index=False)
    plot_importance(imp, _fig_path("feature_importance.png"))
    # Importância por grupo baseada em informação mútua (consistente com a tabela de grupos)
    mi_for_group = mi_tbl.rename(columns={"mutual_info": "importance"})
    grp = interp.group_importance(mi_for_group)
    _save_table(grp.round(5), "group_importance.csv", index=False)
    plot_group_importance(grp, _fig_path("group_importance.png"))
    print("  [figuras] feature_importance.png / group_importance.png")
    print(imp.round(4).to_string())

    abl = interp.ablation_serp_before(lambda: clone(models[champ]), X, y).round(4)
    _save_table(abl, "ablation_serp_position.csv")
    print(abl.to_string())

    with open(os.path.join(config.RESULTS_DIR, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print("\nConcluido. Tabelas em results/tables, figuras em results/figures.")


if __name__ == "__main__":
    main()
