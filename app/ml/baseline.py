"""Pilar 1 — Baseline individual dinâmica.

Para cada sinal vital aprende o que é "normal para este pet" em cada contexto
(repouso diurno, repouso noturno, caminhada, corrida). Dentro do contexto, o valor
esperado acompanha o nível de atividade do momento (regressão linear por pet), e a
dispersão é medida com estatística robusta (mediana e MAD), que não é distorcida por
poucos valores extremos já presentes no histórico.
"""

from dataclasses import dataclass

import numpy as np

from app.ml.preparacao import Serie

CONTEXTOS = {-1: "geral", 0: "repouso", 1: "caminhada", 2: "corrida", 3: "repouso_noturno"}
CONTEXTOS_REPOUSO = (-1, 0, 3)
SINAIS_PONTUAIS = ("frequencia_cardiaca", "temperatura", "pressao")

# Escala mínima evita z enorme quando o histórico é quase constante.
ESCALA_MINIMA = {"frequencia_cardiaca": 3.0, "temperatura": 0.15, "pressao": 4.0}
ESCALA_RELATIVA = {"frequencia_cardiaca": 0.03, "temperatura": 0.0, "pressao": 0.03}
MIN_LEITURAS_CONTEXTO = 12
MIN_LEITURAS_REGRESSAO = 30
MIN_DIAS_ATIVIDADE = 3


def contexto_atividade(atividade: np.ndarray, hora_dia: np.ndarray) -> np.ndarray:
    """Mesma classificação da coleira (<=30% repouso, 31–60% caminhada, >60% corrida),
    separando o repouso noturno (22h–6h), quando a FC basal é naturalmente mais baixa."""
    contexto = np.full(atividade.shape, -1, dtype=int)
    noite = (hora_dia >= 22) | (hora_dia < 6)
    with np.errstate(invalid="ignore"):
        contexto[(atividade <= 30) & ~noite] = 0
        contexto[(atividade <= 30) & noite] = 3
        contexto[(atividade > 30) & (atividade <= 60)] = 1
        contexto[atividade > 60] = 2
    return contexto


def em_repouso(contexto: np.ndarray) -> np.ndarray:
    return np.isin(contexto, CONTEXTOS_REPOUSO)


@dataclass(frozen=True)
class Estatistica:
    mediana: float
    escala: float
    n: int


def estatistica_robusta(valores: np.ndarray, escala_minima: float) -> Estatistica | None:
    valores = valores[~np.isnan(valores)]
    if len(valores) == 0:
        return None
    mediana = float(np.median(valores))
    mad = float(np.median(np.abs(valores - mediana)))
    return Estatistica(mediana=mediana, escala=max(1.4826 * mad, escala_minima), n=len(valores))


@dataclass(frozen=True)
class ModeloSinal:
    """Valor esperado = intercepto + inclinacao * atividade; z usa a dispersão dos resíduos."""

    intercepto: float
    inclinacao: float
    residuo: Estatistica
    nivel_tipico: float

    def esperado(self, atividade: float) -> float:
        if self.inclinacao == 0:
            return self.intercepto + self.residuo.mediana
        if np.isnan(atividade):
            return self.nivel_tipico
        return self.intercepto + self.inclinacao * atividade + self.residuo.mediana


def ajustar_modelo(sinal: str, valores: np.ndarray, atividade: np.ndarray) -> ModeloSinal | None:
    validos = ~np.isnan(valores)
    if validos.sum() < MIN_LEITURAS_CONTEXTO:
        return None
    nivel = float(np.median(valores[validos]))
    escala_minima = max(ESCALA_MINIMA[sinal], ESCALA_RELATIVA[sinal] * abs(nivel))

    com_atividade = validos & ~np.isnan(atividade)
    if com_atividade.sum() >= MIN_LEITURAS_REGRESSAO and np.ptp(atividade[com_atividade]) >= 5:
        x, y = atividade[com_atividade], valores[com_atividade]
        inclinacao, intercepto = np.polyfit(x, y, 1)
        # Reajuste sem os outliers: mínimos quadrados puro é puxado por episódios antigos.
        residuos = y - (intercepto + inclinacao * x)
        desvio = np.abs(residuos - np.median(residuos))
        normais = desvio <= 3.5 * max(1.4826 * float(np.median(desvio)), escala_minima)
        if MIN_LEITURAS_REGRESSAO <= normais.sum() < len(y):
            inclinacao, intercepto = np.polyfit(x[normais], y[normais], 1)
        residuos = y - (intercepto + inclinacao * x)
    else:
        inclinacao, intercepto = 0.0, 0.0
        residuos = valores[validos]

    est = estatistica_robusta(residuos, escala_minima)
    return ModeloSinal(float(intercepto), float(inclinacao), est, nivel)


class BaselineIndividual:
    def __init__(self, treino: Serie):
        atividade = treino.valores["atividade"]
        contexto = contexto_atividade(atividade, treino.hora_dia)
        self.modelos: dict[str, dict[int, ModeloSinal]] = {}
        for sinal in SINAIS_PONTUAIS:
            valores = treino.valores[sinal]
            por_contexto: dict[int, ModeloSinal] = {}
            sem_atividade = np.full(len(valores), np.nan)
            geral = ajustar_modelo(sinal, valores, sem_atividade)
            if geral is not None:
                por_contexto[-1] = geral
            for codigo in (0, 1, 2, 3):
                mascara = contexto == codigo
                modelo = ajustar_modelo(sinal, valores[mascara], atividade[mascara])
                if modelo is not None:
                    por_contexto[codigo] = modelo
            self.modelos[sinal] = por_contexto

        self.atividade_diaria = self._baseline_atividade_diaria(treino)

    @staticmethod
    def _baseline_atividade_diaria(treino: Serie) -> Estatistica | None:
        """Média diária de atividade, só com dias de cobertura completa."""
        atividade = treino.valores["atividade"]
        dias: dict[object, list[float]] = {}
        for data, valor in zip(treino.datas, atividade):
            if not np.isnan(valor):
                dias.setdefault(data.date(), []).append(valor)
        if not dias:
            return None
        maior_cobertura = max(len(v) for v in dias.values())
        medias = np.array(
            [np.mean(v) for v in dias.values() if len(v) >= 0.6 * maior_cobertura], dtype=float
        )
        if len(medias) < MIN_DIAS_ATIVIDADE:
            return None
        return estatistica_robusta(medias, escala_minima=max(1.0, 0.08 * float(np.median(medias))))

    def modelo(self, sinal: str, contexto: int) -> ModeloSinal | None:
        por_contexto = self.modelos.get(sinal, {})
        return por_contexto.get(contexto) or por_contexto.get(-1)

    def z(
        self, sinal: str, valores: np.ndarray, contextos: np.ndarray, atividade: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (z robusto, valor esperado para o pet) leitura a leitura."""
        z = np.full(valores.shape, np.nan)
        esperado = np.full(valores.shape, np.nan)
        for i, (valor, contexto, ativ) in enumerate(zip(valores, contextos, atividade)):
            modelo = self.modelo(sinal, int(contexto))
            if modelo is None or np.isnan(valor):
                continue
            esperado[i] = modelo.esperado(ativ)
            z[i] = (valor - esperado[i]) / modelo.residuo.escala
        return z, esperado
