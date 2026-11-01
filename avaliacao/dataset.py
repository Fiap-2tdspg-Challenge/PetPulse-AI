"""Carrega o dataset sintético (dados/) nos mesmos modelos que o serviço recebe pela API."""

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

from app.schemas import EventoClinico, Leitura, PerfilPet

PASTA_DADOS = Path(__file__).resolve().parent.parent / "dados"


def _numero(valor: str) -> float | None:
    return float(valor) if valor not in ("", None) else None


def carregar_pets(pasta: Path = PASTA_DADOS) -> dict[int, tuple[PerfilPet, list[EventoClinico]]]:
    pets = json.loads((pasta / "pets.json").read_text(encoding="utf-8"))
    return {
        p["id"]: (
            PerfilPet.model_validate(p),
            [EventoClinico.model_validate(e) for e in p["historicoClinico"]],
        )
        for p in pets
    }


def carregar_leituras(pasta: Path = PASTA_DADOS) -> dict[int, list[tuple[Leitura, str]]]:
    """Leituras por pet, cada uma com o id do episódio rotulado ('' = normal)."""
    por_pet: dict[int, list[tuple[Leitura, str]]] = {}
    with open(pasta / "leituras.csv", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            leitura = Leitura(
                data_leitura=datetime.fromisoformat(linha["dataLeitura"]),
                frequencia_cardiaca=_numero(linha["frequenciaCardiaca"]),
                nivel_atividade=_numero(linha["nivelAtividade"]),
                pressao=_numero(linha["pressao"]),
                temperatura=_numero(linha["temperatura"]),
            )
            por_pet.setdefault(int(linha["petId"]), []).append((leitura, linha["episodio"]))
    return por_pet


def carregar_episodios(pasta: Path = PASTA_DADOS) -> list[dict]:
    episodios = json.loads((pasta / "episodios.json").read_text(encoding="utf-8"))
    for ep in episodios:
        ep["inicio"] = datetime.fromisoformat(ep["inicio"])
        ep["fim"] = ep["inicio"] + timedelta(hours=ep["horas"])
    return episodios
