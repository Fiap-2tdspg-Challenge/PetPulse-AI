"""API HTTP do PetPulse-AI (FastAPI).

Fluxo: a API Java recebe uma leitura da coleira, envia a janela recente de leituras +
perfil do pet + histórico clínico para POST /v1/analises e, se ``gerarAlerta`` vier
verdadeiro, grava o SmartAlert no Oracle — que o app mobile já exibe e notifica.
Cada análise também fica registrada no SQLite do serviço (GET /v1/analises).
"""

import logging
import time

from fastapi import Depends, FastAPI, Header, HTTPException, Query

from app import VERSAO
from app.config import carregar_configuracao
from app.genai.alerta import RedatorAlertas
from app.ml.motor import MotorAnomalias
from app.repositorio import RepositorioAnalises
from app.schemas import AnaliseRequest, AnaliseResponse, RegistroAnalise

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("petpulse-ai")

configuracao = carregar_configuracao()
motor = MotorAnomalias()
redator = RedatorAlertas.da_configuracao(configuracao)
repositorio = RepositorioAnalises(configuracao.caminho_banco)

app = FastAPI(
    title="PetPulse-AI",
    version=VERSAO,
    description=(
        "Detecção de anomalias na saúde do pet (baseline individual + Isolation Forest) "
        "e alertas contextuais redigidos com Gemini."
    ),
)


def verificar_chave(x_api_key: str | None = Header(default=None)) -> None:
    """Autenticação serviço-a-serviço: só a API Java conhece a AI_SERVICE_KEY."""
    if configuracao.chave_servico and x_api_key != configuracao.chave_servico:
        raise HTTPException(status_code=401, detail="Chave de serviço inválida")


@app.get("/health")
def health() -> dict:
    return {
        "status": "UP",
        "versao": VERSAO,
        "textoDosAlertas": configuracao.gemini_modelo if redator.llm_ativo else "template",
    }


@app.post("/v1/analises", response_model=AnaliseResponse, dependencies=[Depends(verificar_chave)])
def analisar(requisicao: AnaliseRequest) -> AnaliseResponse:
    inicio = time.perf_counter()
    try:
        resultado = motor.analisar(requisicao.pet, requisicao.leituras, requisicao.historico_clinico)
    except ValueError as erro:
        raise HTTPException(status_code=422, detail=str(erro)) from erro

    alerta = None
    if resultado.gerar_alerta:
        alerta, motivo = redator.redigir(requisicao.pet, resultado, usar_llm=requisicao.gerar_texto)
        if motivo:
            resultado.justificativa.append(motivo)

    resposta = AnaliseResponse(
        pet_id=resultado.pet_id,
        data_referencia=resultado.data_referencia,
        nivel_risco=resultado.nivel_risco,
        gerar_alerta=resultado.gerar_alerta,
        tipo_alerta=resultado.tipo_alerta,
        modo=resultado.modo,
        score_anomalia=resultado.score_anomalia,
        desvios=resultado.desvios,
        contexto_clinico=resultado.contexto_clinico,
        justificativa=resultado.justificativa,
        alerta=alerta,
        leituras_validas=resultado.leituras_validas,
        valores_descartados=resultado.valores_descartados,
        versao_modelo=VERSAO,
    )

    duracao_ms = (time.perf_counter() - inicio) * 1000
    repositorio.salvar(resposta, duracao_ms)
    log.info(
        "pet=%s nivel=%s alerta=%s texto=%s modo=%s leituras=%s %.0fms",
        requisicao.pet.id,
        resultado.nivel_risco,
        resultado.gerar_alerta,
        alerta.gerado_por if alerta else "-",
        resultado.modo,
        resultado.leituras_validas,
        duracao_ms,
    )
    return resposta


@app.get("/v1/analises", response_model=list[RegistroAnalise], dependencies=[Depends(verificar_chave)])
def historico(
    pet_id: int | None = Query(default=None, alias="petId"),
    limite: int = Query(default=50, ge=1, le=500),
) -> list[RegistroAnalise]:
    """Histórico das análises feitas pelo serviço (mais recentes primeiro)."""
    return repositorio.listar(pet_id, limite)
