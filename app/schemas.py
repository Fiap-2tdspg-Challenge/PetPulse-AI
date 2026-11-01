"""Contratos de entrada e saída do serviço (validados com Pydantic).

O JSON trafega em camelCase (``frequenciaCardiaca``, ``dataLeitura``), o mesmo padrão
do payload da coleira e da API Java. No código Python os campos ficam em snake_case.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

Especie = Literal["cao", "gato"]
Porte = Literal["pequeno", "medio", "grande"]
NivelRisco = Literal["BAIXO", "MEDIO", "ALTO"]
Sinal = Literal["frequencia_cardiaca", "temperatura", "pressao", "atividade"]


class ModeloBase(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Leitura(ModeloBase):
    """Uma leitura da coleira. Campos ausentes são aceitos (sensor sem dado naquele ciclo)."""

    data_leitura: datetime
    frequencia_cardiaca: float | None = Field(default=None, ge=0, description="bpm")
    nivel_atividade: float | None = Field(default=None, ge=0, description="0 a 100 (%)")
    pressao: float | None = Field(default=None, ge=0, description="sistólica em mmHg (cmHg é convertido)")
    temperatura: float | None = Field(default=None, ge=0, description="°C")


class PerfilPet(ModeloBase):
    id: int
    nome: str = Field(min_length=1, max_length=100)
    especie: Especie = "cao"
    raca: str | None = None
    porte: Porte | None = None
    idade_anos: float | None = Field(default=None, ge=0, le=40)
    peso_kg: float | None = Field(default=None, gt=0)
    sexo: Literal["M", "F"] | None = None


class EventoClinico(ModeloBase):
    """Registro do histórico clínico (mesmos tipos de ``RecordType`` da API Java)."""

    tipo: Literal["VACINA", "CONSULTA", "DOENCA", "MEDICAMENTO", "OBSERVACAO", "EXAME"]
    descricao: str | None = None
    data: date
    data_retorno: date | None = None


class AnaliseRequest(ModeloBase):
    pet: PerfilPet
    leituras: list[Leitura] = Field(min_length=1, max_length=20000)
    historico_clinico: list[EventoClinico] = Field(default_factory=list)
    gerar_texto: bool = Field(default=True, description="False desliga o LLM e usa o template")


class Desvio(ModeloBase):
    sinal: Sinal
    direcao: Literal["acima", "abaixo"]
    tipo: Literal["persistente", "tendencia", "critico", "pontual"]
    valor_atual: float
    referencia: float = Field(description="baseline individual (mediana) ou limite clínico")
    desvio_percentual: float
    z_robusto: float | None = None
    contexto: Literal["repouso", "repouso_noturno", "caminhada", "corrida", "geral", "24h"]
    persistencia_horas: float


class TextoAlerta(ModeloBase):
    titulo: str
    mensagem: str
    recomendacao: str
    gerado_por: Literal["gemini", "template"]


class AnaliseResponse(ModeloBase):
    pet_id: int
    data_referencia: datetime
    nivel_risco: NivelRisco | None
    gerar_alerta: bool
    tipo_alerta: str | None
    modo: Literal["baseline_individual", "referencia_populacional"]
    score_anomalia: float = Field(ge=0, le=1)
    desvios: list[Desvio]
    contexto_clinico: list[str]
    justificativa: list[str]
    alerta: TextoAlerta | None
    leituras_validas: int
    valores_descartados: int
    versao_modelo: str


class RegistroAnalise(ModeloBase):
    """Uma análise gravada no SQLite do serviço (histórico/log consultável)."""

    id: int
    criado_em: datetime
    pet_id: int
    data_referencia: datetime
    nivel_risco: NivelRisco | None
    gerar_alerta: bool
    tipo_alerta: str | None
    modo: str
    score_anomalia: float
    texto_gerado_por: str | None
    mensagem: str | None
    justificativa: list[str]
    duracao_ms: float
