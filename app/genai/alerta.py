"""Pilar 3 — Alertas contextuais inteligentes.

O Gemini recebe apenas os fatos já calculados pelo motor e devolve JSON validado com
Pydantic. Antes de aceitar o texto, a aplicação confere guardrails (números inventados,
menção a remédios/doses, tamanho, nome do pet). Qualquer falha — sem chave, timeout,
JSON inválido ou guardrail violado — cai num template determinístico, então o alerta
nunca deixa de ser gerado.
"""

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as TempoEsgotado

from pydantic import BaseModel, Field, ValidationError

from app.config import Configuracao
from app.genai.prompts import SYSTEM_PROMPT, montar_entrada
from app.ml.motor import ResultadoMotor
from app.schemas import Desvio, PerfilPet, TextoAlerta

log = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gemini")

NOME_SINAL = {
    "frequencia_cardiaca": "frequência cardíaca",
    "temperatura": "temperatura",
    "pressao": "pressão arterial",
    "atividade": "movimentação",
}
UNIDADE = {"frequencia_cardiaca": "bpm", "temperatura": "°C", "pressao": "mmHg", "atividade": "%"}
TEXTO_CONTEXTO = {
    "repouso": "em repouso",
    "repouso_noturno": "durante o repouso noturno",
    "caminhada": "durante caminhadas",
    "corrida": "durante atividade intensa",
    "geral": "",
    "24h": "nas últimas 24 horas",
}
CONDUTA = {
    "BAIXO": "Informar que nenhuma ação é necessária agora e que o monitoramento continua.",
    "MEDIO": "Recomendar observar o pet e agendar consulta de rotina com o veterinário.",
    "ALTO": "Recomendar procurar atendimento veterinário imediato.",
}
TERMOS_PROIBIDOS = re.compile(
    r"\b(dose|dosagem|mg|ml|comprimidos?|gotas|administr\w+|medique|dipirona|paracetamol|"
    r"ibuprofeno|antibi[oó]tico\w*|anti-?inflamat[oó]ri\w*|diagn[oó]stico confirmado)\b",
    re.IGNORECASE,
)
NUMERO = re.compile(r"\d+(?:[.,]\d+)?")


class AlertaContextual(BaseModel):
    """Schema enviado ao Gemini como saída estruturada."""

    titulo: str = Field(min_length=5, max_length=80)
    mensagem: str = Field(min_length=20, max_length=480)
    recomendacao: str = Field(min_length=10, max_length=950)


def _artigo(pet: PerfilPet) -> tuple[str, str, str]:
    """('do', 'dele', 'o') para macho, ('da', 'dela', 'a') para fêmea."""
    if pet.sexo == "F":
        return "da", "dela", "a"
    if pet.sexo == "M":
        return "do", "dele", "o"
    return "de", "do pet", ""


def _fmt(valor: float, sinal: str) -> str:
    return f"{valor:.1f}".replace(".", ",") if sinal == "temperatura" else f"{valor:.0f}"


def _variacao(d: Desvio) -> str:
    """Temperatura é comparada em graus (1 °C já é muito); os demais sinais em %."""
    if d.sinal == "temperatura":
        return f"{_fmt(abs(d.valor_atual - d.referencia), d.sinal)} °C"
    return f"{abs(d.desvio_percentual):.0f}%"


def _contexto_relevante(contexto_clinico: list[str]) -> str | None:
    for item in contexto_clinico:
        if item.startswith(("Vacina", "Medicação", "Condição")):
            return item
    return None


def montar_fatos(pet: PerfilPet, resultado: ResultadoMotor) -> dict:
    artigo, _, _ = _artigo(pet)
    desvios = []
    for d in resultado.desvios[:3]:
        un = UNIDADE[d.sinal]
        desvios.append({
            "sinal": NOME_SINAL[d.sinal],
            "tipo": d.tipo,
            "direcao": d.direcao,
            "valor_atual": f"{_fmt(d.valor_atual, d.sinal)} {un}",
            "padrao_do_pet" if d.tipo != "critico" else "limite_de_seguranca":
                f"{_fmt(d.referencia, d.sinal)} {un}",
            "variacao": _variacao(d),
            "contexto": TEXTO_CONTEXTO[d.contexto],
            "duracao_horas": round(d.persistencia_horas),
        })
    return {
        "pet": {
            "nome": pet.nome,
            "especie": "gato" if pet.especie == "gato" else "cão",
            "raca": pet.raca,
            "idade_anos": pet.idade_anos,
            "artigo": artigo,
        },
        "nivel_risco": resultado.nivel_risco,
        "conduta_obrigatoria": CONDUTA[resultado.nivel_risco or "BAIXO"],
        "desvios": desvios,
        "contexto_clinico": resultado.contexto_clinico,
    }


def _numeros(texto: str) -> list[float]:
    return [float(n.replace(",", ".")) for n in NUMERO.findall(texto)]


def validar_texto(alerta: AlertaContextual, fatos: dict, pet: PerfilPet) -> str | None:
    """Retorna o motivo da rejeição, ou None se o texto passou nos guardrails."""
    texto = f"{alerta.titulo} {alerta.mensagem} {alerta.recomendacao}"
    if pet.nome.lower() not in alerta.mensagem.lower():
        return "mensagem não cita o nome do pet"
    if TERMOS_PROIBIDOS.search(texto):
        return f"menção proibida a medicamento/dose: '{TERMOS_PROIBIDOS.search(texto).group(0)}'"
    permitidos = _numeros(str(fatos))
    for n in _numeros(texto):
        if not any(abs(n - p) <= 0.5 for p in permitidos):
            return f"número {n:g} não existe nos fatos (possível alucinação)"
    return None


def texto_template(pet: PerfilPet, resultado: ResultadoMotor) -> AlertaContextual:
    artigo, possessivo, definido = _artigo(pet)
    nome_com_artigo = f"{definido} {pet.nome}".strip()
    frases = []
    for d in resultado.desvios[:2]:
        nome_sinal = NOME_SINAL[d.sinal]
        un = UNIDADE[d.sinal]
        if d.tipo == "critico":
            frases.append(
                f"{pet.nome} registrou {nome_sinal} de {_fmt(d.valor_atual, d.sinal)} {un}, "
                f"além do limite de segurança de {_fmt(d.referencia, d.sinal)} {un}."
            )
        elif d.tipo == "tendencia":
            frases.append(
                f"A {nome_sinal} {artigo} {pet.nome} caiu {_variacao(d)} nas últimas 24 horas "
                f"em relação à média diária {possessivo}."
            )
        else:
            sentido = "acima" if d.direcao == "acima" else "abaixo"
            contexto = TEXTO_CONTEXTO[d.contexto]
            frases.append(
                f"A {nome_sinal} {artigo} {pet.nome} está {_variacao(d)} {sentido} do padrão {possessivo}"
                f"{' ' + contexto if contexto else ''} há cerca de {d.persistencia_horas:.0f} h "
                f"({_fmt(d.valor_atual, d.sinal)} {un}; o habitual é {_fmt(d.referencia, d.sinal)} {un})."
            )

    sinais = {d.sinal for d in resultado.desvios}
    if resultado.nivel_risco == "ALTO":
        titulo = f"Atenção: {pet.nome} precisa de avaliação veterinária"
        recomendacao = (
            "Procure atendimento veterinário o quanto antes e leve o histórico do PetPulse "
            "para agilizar a avaliação."
        )
    elif resultado.nivel_risco == "MEDIO":
        titulo = (
            f"Alterações combinadas na saúde {artigo} {pet.nome}" if len(sinais) > 1
            else f"Mudança no padrão de {NOME_SINAL[next(iter(sinais))]} {artigo} {pet.nome}"
        )
        recomendacao = (
            f"Observe {nome_com_artigo} nas próximas horas (apetite, respiração e disposição) "
            "e agende uma consulta de rotina com o veterinário."
        )
    else:
        titulo = f"Pequena variação nos sinais {artigo} {pet.nome}"
        recomendacao = "Nenhuma ação é necessária agora; o PetPulse continua monitorando."

    contexto = _contexto_relevante(resultado.contexto_clinico)
    if contexto:
        recomendacao += f" Informe ao veterinário: {contexto}."

    return AlertaContextual(
        titulo=titulo[:80], mensagem=" ".join(frases)[:480], recomendacao=recomendacao[:950]
    )


class RedatorAlertas:
    def __init__(self, cliente=None, modelo: str = "gemini-3.5-flash", timeout_s: float = 20.0):
        self.cliente = cliente
        self.modelo = modelo
        self.timeout_s = timeout_s

    @classmethod
    def da_configuracao(cls, cfg: Configuracao) -> "RedatorAlertas":
        cliente = None
        if cfg.gemini_api_key:
            from google import genai
            from google.genai import types

            # Sem novas tentativas: com a cota esgotada (429) é melhor cair logo no template.
            cliente = genai.Client(
                api_key=cfg.gemini_api_key,
                http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
            )
        return cls(cliente=cliente, modelo=cfg.gemini_modelo, timeout_s=cfg.gemini_timeout_s)

    @property
    def llm_ativo(self) -> bool:
        return self.cliente is not None

    def _gerar_com_gemini(self, fatos: dict) -> AlertaContextual:
        interaction = self.cliente.interactions.create(
            model=self.modelo,
            system_instruction=SYSTEM_PROMPT,
            input=montar_entrada(fatos),
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": AlertaContextual.model_json_schema(),
            },
            timeout=self.timeout_s,
        )
        return AlertaContextual.model_validate_json(interaction.output_text)

    def redigir(
        self, pet: PerfilPet, resultado: ResultadoMotor, usar_llm: bool = True
    ) -> tuple[TextoAlerta, str | None]:
        """Retorna o texto final e, se houve fallback, o motivo (vai para a justificativa)."""
        fatos = montar_fatos(pet, resultado)
        motivo = None
        if usar_llm and self.llm_ativo:
            try:
                # O timeout do SDK nem sempre é respeitado (ex.: cota esgotada), então o
                # limite é garantido aqui: estourou, o alerta sai na hora com o template.
                alerta = _EXECUTOR.submit(self._gerar_com_gemini, fatos).result(timeout=self.timeout_s)
                motivo = validar_texto(alerta, fatos, pet)
                if motivo is None:
                    return TextoAlerta(**alerta.model_dump(), gerado_por="gemini"), None
                motivo = f"Texto do Gemini rejeitado pelos guardrails: {motivo}; usado template."
            except ValidationError as erro:
                motivo = f"Gemini devolveu JSON fora do schema ({erro.error_count()} erro(s)); usado template."
            except TempoEsgotado:
                motivo = f"Gemini não respondeu em {self.timeout_s:.0f} s; usado template."
            except Exception as erro:  # rede, cota (429), chave inválida
                motivo = f"Gemini indisponível ({type(erro).__name__}); usado template."
            log.warning(motivo)

        alerta = texto_template(pet, resultado)
        return TextoAlerta(**alerta.model_dump(), gerado_por="template"), motivo
