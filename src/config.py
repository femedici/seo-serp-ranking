"""Configuracao central do projeto: caminhos, semente, alvo e grupos de fatores."""

from __future__ import annotations

import os

# ----------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "seo_dataset.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# ----------------------------------------------------------------------
# Reprodutibilidade
# ----------------------------------------------------------------------
SEED = 42

# ----------------------------------------------------------------------
# Variavel-alvo e particionamento
# ----------------------------------------------------------------------
TARGET = "ranking_improved"
HOLDOUT_SIZE = 0.20          # holdout estratificado de confirmacao final
CV_OUTER_SPLITS = 5          # validacao cruzada externa
CV_INNER_SPLITS = 3          # ajuste de hiperparametros (CV interna)
CV_REPEATS = 3               # repeticoes da Stratified K-Fold (mais medidas)
SEARCH_ITER = 25             # n_iter do RandomizedSearchCV
ALPHA = 0.05                 # nivel de significancia dos testes estatisticos

# ----------------------------------------------------------------------
# Grupos de fatores de SEO (conforme levantamento bibliografico, secao 5.4)
# ----------------------------------------------------------------------
FEATURE_GROUPS = {
    "on_page": [
        "content_length",
        "keyword_density",
        "has_meta_description",
        "has_alt_text",
        "num_internal_links",
    ],
    "authority": [
        "domain_authority",
        "page_authority",
        "backlink_count",
        "num_external_links",
    ],
    "behavioral": [
        "avg_time_on_page_sec",
        "bounce_rate",
        "scroll_depth_percent",
    ],
    "contextual": [
        "serp_position_before",
    ],
}

# Atributos binarios (ja em {0,1}) e numericos continuos/discretos
BINARY_FEATURES = ["has_meta_description", "has_alt_text"]

# Feature confundidora com efeito-teto (analise de ablacao / leakage, lacuna D7)
LEAKAGE_CANDIDATE = "serp_position_before"


def all_features() -> list[str]:
    """Lista plana com as 13 preditoras na ordem dos grupos."""
    feats: list[str] = []
    for cols in FEATURE_GROUPS.values():
        feats.extend(cols)
    return feats


def ensure_dirs() -> None:
    """Garante a existencia dos diretorios de saida."""
    for d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR):
        os.makedirs(d, exist_ok=True)
