import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.genai.alerta import RedatorAlertas, montar_fatos, texto_template, validar_texto
from app.ml.motor import ResultadoMotor
from app.schemas import Desvio, PerfilPet

MEL = PerfilPet(id=2, nome="Mel", especie="cao", raca="Chihuahua", porte="pequeno", idade_anos=11, sexo="F")


@pytest.fixture
def resultado_medio() -> ResultadoMotor:
    return ResultadoMotor(
        pet_id=2,
        data_referencia=datetime(2026, 10, 13, 9, 0),
        nivel_risco="MEDIO",
        modo="baseline_individual",
        score_anomalia=0.97,
        desvios=[
            Desvio(
                sinal="frequencia_cardiaca", direcao="acima", tipo="persistente", valor_atual=136,
                referencia=106.6, desvio_percentual=27.6, z_robusto=6.1, contexto="repouso",
                persistencia_horas=5.8,
            )
        ],
        contexto_clinico=["Vacina aplicada em 09/10: Antirrábica"],
        justificativa=[],
        leituras_validas=1000,
        valores_descartados=0,
        tipo_alerta="FREQUENCIA_CARDIACA",
    )


class GeminiFalso:
    """Imita client.interactions.create devolvendo um texto fixo (ou lançando erro)."""

    def __init__(self, resposta=None, erro: Exception | None = None):
        self.chamadas = []
        self.interactions = SimpleNamespace(create=self._create)
        self._resposta = resposta
        self._erro = erro

    def _create(self, **kwargs):
        self.chamadas.append(kwargs)
        if self._erro:
            raise self._erro
        texto = self._resposta if isinstance(self._resposta, str) else json.dumps(self._resposta)
        return SimpleNamespace(output_text=texto)


RESPOSTA_BOA = {
    "titulo": "Frequência cardíaca da Mel acima do padrão",
    "mensagem": "A frequência cardíaca da Mel está 28% acima do padrão dela em repouso há cerca de 6 horas "
                "(136 bpm, quando o habitual é 107 bpm).",
    "recomendacao": "Observe se ela está ofegante ou sem apetite e agende uma consulta de rotina. "
                    "Ela tomou a vacina antirrábica em 09/10.",
}


def test_template_cita_pet_valores_e_contexto_clinico(resultado_medio):
    alerta = texto_template(MEL, resultado_medio)
    assert "Mel" in alerta.mensagem
    assert "28%" in alerta.mensagem and "136 bpm" in alerta.mensagem
    assert "consulta de rotina" in alerta.recomendacao
    assert "Antirrábica" in alerta.recomendacao
    assert validar_texto(alerta, montar_fatos(MEL, resultado_medio), MEL) is None


def test_usa_gemini_quando_o_texto_passa_nos_guardrails(resultado_medio):
    gemini = GeminiFalso(RESPOSTA_BOA)
    texto, motivo = RedatorAlertas(cliente=gemini).redigir(MEL, resultado_medio)
    assert motivo is None
    assert texto.gerado_por == "gemini"
    assert texto.mensagem == RESPOSTA_BOA["mensagem"]
    chamada = gemini.chamadas[0]
    assert chamada["response_format"]["mime_type"] == "application/json"
    assert "NUNCA" in chamada["system_instruction"].upper()


@pytest.mark.parametrize(
    ("resposta", "trecho_motivo"),
    [
        ({**RESPOSTA_BOA, "mensagem": "A FC da Mel subiu 45% nas últimas 12 horas."}, "alucinação"),
        ({**RESPOSTA_BOA, "recomendacao": "Dê meio comprimido de dipirona à Mel."}, "medicamento"),
        ({**RESPOSTA_BOA, "mensagem": "A frequência cardíaca está 28% acima do padrão em repouso."}, "nome do pet"),
        ("isto não é json", "schema"),
        ({"titulo": "x"}, "schema"),
    ],
)
def test_texto_do_gemini_rejeitado_cai_no_template(resultado_medio, resposta, trecho_motivo):
    texto, motivo = RedatorAlertas(cliente=GeminiFalso(resposta)).redigir(MEL, resultado_medio)
    assert texto.gerado_por == "template"
    assert trecho_motivo in motivo


def test_gemini_fora_do_ar_nao_impede_o_alerta(resultado_medio):
    texto, motivo = RedatorAlertas(cliente=GeminiFalso(erro=TimeoutError())).redigir(MEL, resultado_medio)
    assert texto.gerado_por == "template"
    assert "TimeoutError" in motivo


def test_sem_chave_usa_template_sem_motivo(resultado_medio):
    texto, motivo = RedatorAlertas(cliente=None).redigir(MEL, resultado_medio)
    assert texto.gerado_por == "template"
    assert motivo is None


def test_textos_cabem_nas_colunas_do_oracle(resultado_medio):
    resultado_medio.desvios = resultado_medio.desvios * 5
    resultado_medio.contexto_clinico = ["Medicação registrada em 01/10: " + "x" * 900]
    alerta = texto_template(MEL, resultado_medio)
    assert len(alerta.mensagem) <= 500
    assert len(alerta.recomendacao) <= 1000
