"""Pilar 2 — Detecção multivariada de anomalias com Isolation Forest.

Um modelo por pet, treinado só com o histórico dele. O Isolation Forest isola pontos
raros com poucos cortes aleatórios; combinações incomuns de sinais (ex.: FC alta em
repouso de madrugada) ficam isoladas rapidamente mesmo quando cada sinal, sozinho,
parece normal.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from app.ml.preparacao import Serie

COLUNAS = ("frequencia_cardiaca", "temperatura", "pressao", "atividade", "hora_sen", "hora_cos")


def montar_matriz(serie: Serie) -> np.ndarray:
    angulo = 2 * np.pi * serie.hora_dia / 24
    return np.column_stack(
        [
            serie.valores["frequencia_cardiaca"],
            serie.valores["temperatura"],
            serie.valores["pressao"],
            serie.valores["atividade"],
            np.sin(angulo),
            np.cos(angulo),
        ]
    )


class DetectorMultivariado:
    def __init__(self, contaminacao: float = 0.02, arvores: int = 150, semente: int = 42):
        self.contaminacao = contaminacao
        self.arvores = arvores
        self.semente = semente

    def treinar(self, X: np.ndarray) -> "DetectorMultivariado":
        # Colunas sem nenhum dado (ex.: temperatura ainda não enviada) ficam de fora.
        self.colunas = np.flatnonzero(~np.all(np.isnan(X), axis=0))
        X = X[:, self.colunas]
        self.medianas = np.nanmedian(X, axis=0)
        X = self._imputar(X)
        self.escalonador = RobustScaler().fit(X)
        self.modelo = IsolationForest(
            n_estimators=self.arvores,
            contamination=self.contaminacao,
            random_state=self.semente,
        ).fit(self.escalonador.transform(X))
        self.scores_treino = self.modelo.score_samples(self.escalonador.transform(X))
        return self

    def _imputar(self, X: np.ndarray) -> np.ndarray:
        X = X.copy()
        faltantes = np.isnan(X)
        X[faltantes] = np.take(self.medianas, np.nonzero(faltantes)[1])
        return X

    def _transformar(self, X: np.ndarray) -> np.ndarray:
        return self.escalonador.transform(self._imputar(X[:, self.colunas]))

    def anomalo(self, X: np.ndarray) -> np.ndarray:
        return self.modelo.predict(self._transformar(X)) == -1

    def percentil_anomalia(self, X: np.ndarray) -> np.ndarray:
        """0 a 1: fração do histórico do pet que é *menos* anômala que cada leitura."""
        scores = self.modelo.score_samples(self._transformar(X))
        return np.array([float(np.mean(self.scores_treino > s)) for s in scores])
