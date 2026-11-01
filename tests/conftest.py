import os
from datetime import datetime, timedelta

# Banco em memória nos testes (antes de qualquer import de app.main).
os.environ.setdefault("PETPULSE_DB", ":memory:")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from app.ml.motor import MotorAnomalias  # noqa: E402
from app.schemas import Leitura, PerfilPet  # noqa: E402

INICIO = datetime(2026, 10, 1)


def gerar_leituras(
    dias: float = 5,
    passo_min: int = 10,
    fc_repouso: float = 80,
    temperatura: float = 38.5,
    pressao: float = 130,
    com_temperatura: bool = True,
    semente: int = 7,
    alteracao=None,
) -> list[Leitura]:
    """Série sintética simples de um cão saudável; ``alteracao(i, data, valores)`` injeta anomalias."""
    rng = np.random.default_rng(semente)
    n = int(dias * 24 * 60 / passo_min)
    leituras = []
    ruido = 0.0
    for i in range(n):
        data = INICIO + timedelta(minutes=passo_min * i)
        noite = data.hour >= 22 or data.hour < 6
        passeio = data.hour in (7, 18) and data.minute < 40
        atividade = rng.uniform(40, 55) if passeio else rng.uniform(0, 8) if noite else rng.uniform(5, 22)
        ruido = 0.7 * ruido + rng.normal(0, 2.5)
        valores = {
            "fc": fc_repouso - (12 if noite else 0) + 0.8 * atividade + ruido,
            "atividade": atividade,
            "temperatura": temperatura + 0.003 * atividade + rng.normal(0, 0.04),
            "pressao": pressao + 0.2 * atividade + rng.normal(0, 2.5),
        }
        if alteracao:
            alteracao(i, data, valores)
        leituras.append(
            Leitura(
                data_leitura=data,
                frequencia_cardiaca=round(valores["fc"]),
                nivel_atividade=round(float(valores["atividade"]), 1),
                pressao=round(valores["pressao"]),
                temperatura=round(valores["temperatura"], 2) if com_temperatura else None,
            )
        )
    return leituras


@pytest.fixture
def pet() -> PerfilPet:
    return PerfilPet(id=10, nome="Rex", especie="cao", raca="SRD", porte="medio", idade_anos=4, sexo="M")


@pytest.fixture
def motor() -> MotorAnomalias:
    return MotorAnomalias()
