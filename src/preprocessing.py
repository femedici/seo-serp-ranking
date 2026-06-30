"""
Pipelines de pre-processamento anti-vazamento.

Principio (Materiais e Metodos, secao 2): escalonamento e reamostragem sao
ajustados SOMENTE no fold de treino, encapsulados em Pipeline. Nada e ajustado
sobre o dado completo antes da particao. Arvores nao recebem escalonamento
(invariantes a escala); modelos lineares/de distancia/margem usam StandardScaler
(z-score) ou RobustScaler (mediana + IQR) quando ha caudas pesadas.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from . import config


def make_scaler(kind: str = "standard"):
    """Retorna o escalonador escolhido."""
    if kind == "robust":
        return RobustScaler()
    return StandardScaler()


def build_pipeline(estimator, *, needs_scaling: bool, scaler: str = "standard") -> SkPipeline:
    """
    Monta um Pipeline scikit-learn. O escalonador, quando necessario, e o
    primeiro passo e e ajustado por fold (sem vazamento).
    """
    steps = []
    if needs_scaling:
        steps.append(("scaler", make_scaler(scaler)))
    steps.append(("model", estimator))
    return SkPipeline(steps)


def build_resampling_pipeline(estimator, *, needs_scaling: bool, scaler: str = "standard") -> ImbPipeline:
    """
    Pipeline com SMOTE aplicado APENAS ao fold de treino (imblearn.Pipeline).
    Usado na comparacao de estrategias de balanceamento (lacuna D3).
    """
    steps = []
    if needs_scaling:
        steps.append(("scaler", make_scaler(scaler)))
    steps.append(("smote", SMOTE(random_state=config.SEED)))
    steps.append(("model", estimator))
    return ImbPipeline(steps)
