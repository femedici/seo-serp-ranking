"""Analise exploratoria: distribuicao do alvo, correlacoes e figuras (secao 1 do protocolo)."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config
from .data_loader import split_xy


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Correlacao de Pearson de cada preditora com o alvo (ordenada por |r|)."""
    corr = df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET)
    out = corr.to_frame("pearson_r")
    out["abs_r"] = out["pearson_r"].abs()
    return out.sort_values("abs_r", ascending=False)


def plot_class_distribution(df: pd.DataFrame, path: str) -> None:
    counts = df[config.TARGET].value_counts().sort_index()
    labels = ["0 - Nao melhorou", "1 - Melhorou"]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#DD8452"])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v}\n({v/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Quantidade de instancias")
    ax.set_title("Distribuicao da variavel-alvo ranking_improved")
    ax.set_ylim(0, counts.max() * 1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, path: str) -> None:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Matriz de correlacao de Pearson")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_target_correlation(corr_table: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    data = corr_table.sort_values("pearson_r")
    colors = ["#C44E52" if v < 0 else "#55A868" for v in data["pearson_r"]]
    ax.barh(data.index, data["pearson_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlacao de Pearson com o alvo")
    ax.set_title("Correlacao linear (fraca) das preditoras com ranking_improved")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
