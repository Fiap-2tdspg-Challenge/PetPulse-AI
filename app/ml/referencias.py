"""Faixas clínicas populacionais.

Usadas em dois momentos:
1. *Cold start*: enquanto o pet ainda não tem histórico suficiente para uma baseline
   individual, a calibração inicial é feita por espécie, porte e idade (Sprint 3, seção 4).
2. *Rede de segurança*: limites críticos que disparam alerta ALTO independentemente da
   baseline (desvio agudo e severo). São os mesmos valores do firmware da coleira.
"""

from app.schemas import PerfilPet

# Faixa de normalidade em repouso (mesmos valores do firmware PetPulse-Iot para cães)
FAIXAS = {
    "cao": {
        "frequencia_cardiaca": (60.0, 140.0),
        "temperatura": (37.5, 39.2),
        "pressao": (100.0, 160.0),
    },
    "gato": {
        "frequencia_cardiaca": (140.0, 220.0),
        "temperatura": (37.8, 39.2),
        "pressao": (110.0, 160.0),
    },
}

# Limites críticos (alerta ALTO imediato, confirmado em 2 das 3 últimas leituras)
CRITICOS = {
    "cao": {"fc_repouso": (40.0, 180.0), "temperatura": (36.0, 40.5), "pressao_max": 180.0},
    "gato": {"fc_repouso": (100.0, 260.0), "temperatura": (36.0, 40.5), "pressao_max": 180.0},
}

TEMPERATURA_FEBRE = 39.5


def faixas_referencia(pet: PerfilPet) -> dict[str, tuple[float, float]]:
    """Ajusta a faixa populacional ao porte e à idade do pet."""
    faixas = dict(FAIXAS[pet.especie])
    minimo, maximo = faixas["frequencia_cardiaca"]

    if pet.especie == "cao":
        if pet.porte == "pequeno":
            minimo, maximo = minimo + 10, maximo + 20
        elif pet.porte == "grande":
            minimo, maximo = minimo - 10, maximo - 20

    if pet.idade_anos is not None and pet.idade_anos < 1:
        maximo += 20

    faixas["frequencia_cardiaca"] = (minimo, maximo)
    return faixas
