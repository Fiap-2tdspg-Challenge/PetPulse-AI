"""Configuração lida de variáveis de ambiente (nenhuma chave fica no código)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Configuracao:
    gemini_api_key: str | None
    gemini_modelo: str
    gemini_timeout_s: float
    chave_servico: str | None
    caminho_banco: str


def carregar_configuracao() -> Configuracao:
    return Configuracao(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_modelo=os.getenv("GEMINI_MODEL", "gemini-3.5-flash"),
        gemini_timeout_s=float(os.getenv("GEMINI_TIMEOUT_S", "20")),
        chave_servico=os.getenv("AI_SERVICE_KEY") or None,
        caminho_banco=os.getenv("PETPULSE_DB", "petpulse_ai.db"),
    )
