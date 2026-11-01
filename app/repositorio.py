"""Persistência das análises em SQLite (Lab 3.5: resultados e logs sobrevivem a um restart).

O alerta que o tutor vê é gravado no Oracle pela API Java. Aqui o serviço de IA guarda o
seu próprio histórico: todas as análises, inclusive as de nível BAIXO (que não viram alerta),
com a justificativa do motor e se o texto veio do Gemini ou do template.
"""

import json
import sqlite3
import threading
from datetime import datetime

from app.schemas import AnaliseResponse, RegistroAnalise

CRIAR_TABELA = """
CREATE TABLE IF NOT EXISTS analises (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    criado_em        TEXT    NOT NULL,
    pet_id           INTEGER NOT NULL,
    data_referencia  TEXT    NOT NULL,
    nivel_risco      TEXT,
    gerar_alerta     INTEGER NOT NULL,
    tipo_alerta      TEXT,
    modo             TEXT    NOT NULL,
    score_anomalia   REAL    NOT NULL,
    texto_gerado_por TEXT,
    mensagem         TEXT,
    justificativa    TEXT    NOT NULL,
    duracao_ms       REAL    NOT NULL
)
"""


class RepositorioAnalises:
    def __init__(self, caminho: str):
        self._conexao = sqlite3.connect(caminho, check_same_thread=False)
        self._conexao.row_factory = sqlite3.Row
        self._trava = threading.Lock()
        with self._trava:
            self._conexao.execute(CRIAR_TABELA)
            self._conexao.commit()

    def salvar(self, analise: AnaliseResponse, duracao_ms: float) -> int:
        with self._trava:
            cursor = self._conexao.execute(
                """INSERT INTO analises (criado_em, pet_id, data_referencia, nivel_risco, gerar_alerta,
                       tipo_alerta, modo, score_anomalia, texto_gerado_por, mensagem, justificativa, duracao_ms)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(timespec="seconds"),
                    analise.pet_id,
                    analise.data_referencia.isoformat(),
                    analise.nivel_risco,
                    int(analise.gerar_alerta),
                    analise.tipo_alerta,
                    analise.modo,
                    analise.score_anomalia,
                    analise.alerta.gerado_por if analise.alerta else None,
                    analise.alerta.mensagem if analise.alerta else None,
                    json.dumps(analise.justificativa, ensure_ascii=False),
                    round(duracao_ms, 1),
                ),
            )
            self._conexao.commit()
            return cursor.lastrowid

    def listar(self, pet_id: int | None = None, limite: int = 50) -> list[RegistroAnalise]:
        sql = "SELECT * FROM analises"
        parametros: tuple = ()
        if pet_id is not None:
            sql += " WHERE pet_id = ?"
            parametros = (pet_id,)
        sql += " ORDER BY id DESC LIMIT ?"
        with self._trava:
            linhas = self._conexao.execute(sql, (*parametros, limite)).fetchall()
        return [
            RegistroAnalise(
                **{k: linha[k] for k in linha.keys() if k not in ("justificativa", "gerar_alerta")},
                gerar_alerta=bool(linha["gerar_alerta"]),
                justificativa=json.loads(linha["justificativa"]),
            )
            for linha in linhas
        ]
