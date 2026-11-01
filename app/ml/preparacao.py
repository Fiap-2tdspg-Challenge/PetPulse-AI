"""Preparação dos dados: validação, limpeza, normalização e ordenação das leituras."""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from app.schemas import Leitura

EPOCA = datetime(1970, 1, 1)

SINAIS = ("frequencia_cardiaca", "temperatura", "pressao", "atividade")

# Valores fisicamente impossíveis para o sensor são ruído (não fisiologia) e viram NaN.
LIMITES_SENSOR = {
    "frequencia_cardiaca": (20.0, 350.0),
    "temperatura": (30.0, 45.0),
    "pressao": (40.0, 300.0),
    "atividade": (0.0, 100.0),
}


@dataclass
class Serie:
    """Série temporal de um pet, ordenada por data, com NaN onde não há dado válido."""

    datas: list[datetime]
    horas: np.ndarray
    valores: dict[str, np.ndarray]
    descartados: int = 0
    hora_dia: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.hora_dia = np.array([d.hour + d.minute / 60 for d in self.datas], dtype=float)

    def __len__(self) -> int:
        return len(self.datas)

    def recorte(self, mascara: np.ndarray) -> "Serie":
        indices = np.flatnonzero(mascara)
        return Serie(
            datas=[self.datas[i] for i in indices],
            horas=self.horas[indices],
            valores={nome: v[indices] for nome, v in self.valores.items()},
        )


def normalizar_data(data: datetime) -> datetime:
    if data.tzinfo is not None:
        return data.astimezone(timezone.utc).replace(tzinfo=None)
    return data


def normalizar_pressao(valor: float) -> float:
    """A API Java aceitava pressão em cmHg nos exemplos (13 = 130 mmHg); a coleira envia mmHg."""
    return valor * 10 if valor < 30 else valor


def preparar(leituras: list[Leitura]) -> Serie:
    # Ordena e remove leituras duplicadas no mesmo instante (fica a última recebida).
    por_data: dict[datetime, Leitura] = {}
    for leitura in leituras:
        por_data[normalizar_data(leitura.data_leitura)] = leitura
    datas = sorted(por_data)

    brutos = {
        "frequencia_cardiaca": [por_data[d].frequencia_cardiaca for d in datas],
        "temperatura": [por_data[d].temperatura for d in datas],
        "pressao": [
            None if por_data[d].pressao is None else normalizar_pressao(por_data[d].pressao)
            for d in datas
        ],
        "atividade": [por_data[d].nivel_atividade for d in datas],
    }

    valores: dict[str, np.ndarray] = {}
    descartados = 0
    for nome, lista in brutos.items():
        arr = np.array([np.nan if v is None else float(v) for v in lista], dtype=float)
        minimo, maximo = LIMITES_SENSOR[nome]
        fora = ~np.isnan(arr) & ((arr < minimo) | (arr > maximo))
        descartados += int(fora.sum())
        arr[fora] = np.nan
        valores[nome] = arr

    # Leituras sem nenhum sinal válido não acrescentam informação.
    algum_valido = np.zeros(len(datas), dtype=bool)
    for arr in valores.values():
        algum_valido |= ~np.isnan(arr)

    horas = np.array([(d - EPOCA).total_seconds() / 3600 for d in datas], dtype=float)
    serie = Serie(datas=datas, horas=horas, valores=valores)
    serie = serie.recorte(algum_valido)
    serie.descartados = descartados
    return serie
