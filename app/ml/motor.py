"""Motor de análise: junta baseline individual, Isolation Forest, matriz de severidade e
contexto clínico em um único resultado técnico (sem texto em linguagem natural).

Princípio: o ML e as regras clínicas decidem o nível de risco; o LLM só redige o texto.
"""

import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, time

import numpy as np

from app.ml.baseline import (
    CONTEXTOS,
    SINAIS_PONTUAIS,
    BaselineIndividual,
    contexto_atividade,
    em_repouso,
)
from app.ml.detector import DetectorMultivariado, montar_matriz
from app.ml.preparacao import Serie, preparar
from app.ml.referencias import CRITICOS, TEMPERATURA_FEBRE, faixas_referencia
from app.schemas import Desvio, EventoClinico, Leitura, NivelRisco, PerfilPet

NIVEIS: list[NivelRisco] = ["BAIXO", "MEDIO", "ALTO"]

TIPOS_ALERTA = {
    "frequencia_cardiaca": "FREQUENCIA_CARDIACA",
    "pressao": "PRESSAO",
    "atividade": "ATIVIDADE",
    "temperatura": "TEMPERATURA",
}
TIPO_ALERTA_MULTIPLO = "ANOMALIA_MULTIVARIADA"


@dataclass(frozen=True)
class ConfigMotor:
    janela_persistencia_h: float = 6.0
    # A baseline só aprende com dados de mais de 24 h atrás, para que um episódio longo
    # (ex.: febre há 20 h) não seja incorporado como "normal" do próprio pet.
    defasagem_baseline_h: float = 24.0
    janela_historico_dias: float = 14.0
    z_limiar: float = 3.0
    z_limiar_com_if: float = 2.0
    z_forte: float = 5.0
    fracao_persistencia: float = 0.5
    min_leituras_janela: int = 6
    min_horas_baseline: float = 24.0
    min_leituras_baseline: int = 72
    min_leituras_if: int = 100
    queda_atividade: float = 0.30
    z_atividade_diaria: float = 2.0
    horas_reacao_vacinal: float = 72.0
    contaminacao_if: float = 0.02
    arvores_if: int = 150
    retreino_if_horas: float = 1.0
    usar_baseline_individual: bool = True
    usar_isolation_forest: bool = True


@dataclass
class ResultadoMotor:
    pet_id: int
    data_referencia: datetime
    nivel_risco: NivelRisco | None
    modo: str
    score_anomalia: float
    desvios: list[Desvio]
    contexto_clinico: list[str]
    justificativa: list[str]
    leituras_validas: int
    valores_descartados: int
    tipo_alerta: str | None = field(default=None)

    @property
    def gerar_alerta(self) -> bool:
        # Severidade baixa = registro em log e monitoramento passivo (Sprint 3, seção 8).
        return self.nivel_risco in ("MEDIO", "ALTO")


class MotorAnomalias:
    def __init__(self, config: ConfigMotor | None = None, max_modelos_cache: int = 256):
        self.config = config or ConfigMotor()
        self._cache: OrderedDict[int, tuple[float, DetectorMultivariado]] = OrderedDict()
        self._max_cache = max_modelos_cache
        self._trava = threading.Lock()

    # ------------------------------------------------------------------ entrada

    def analisar(
        self,
        pet: PerfilPet,
        leituras: list[Leitura],
        historico_clinico: list[EventoClinico] | None = None,
    ) -> ResultadoMotor:
        serie = preparar(leituras)
        if len(serie) == 0:
            raise ValueError("Nenhuma leitura válida após a limpeza dos dados.")
        return self.analisar_serie(pet, serie, historico_clinico or [])

    def analisar_serie(
        self, pet: PerfilPet, serie: Serie, historico_clinico: list[EventoClinico]
    ) -> ResultadoMotor:
        cfg = self.config
        descartados = serie.descartados
        agora_h = float(serie.horas[-1])
        data_ref = serie.datas[-1]

        serie = serie.recorte(serie.horas >= agora_h - cfg.janela_historico_dias * 24)
        treino = serie.recorte(serie.horas <= agora_h - cfg.defasagem_baseline_h)
        recente = serie.recorte(serie.horas > agora_h - cfg.janela_persistencia_h)

        justificativa: list[str] = []
        desvios: list[Desvio] = []
        score = 0.0

        cobertura = float(treino.horas[-1] - treino.horas[0]) if len(treino) > 1 else 0.0
        individual = (
            cfg.usar_baseline_individual
            and len(treino) >= cfg.min_leituras_baseline
            and cobertura >= cfg.min_horas_baseline
        )
        contexto = contexto_atividade(recente.valores["atividade"], recente.hora_dia)

        if individual:
            modo = "baseline_individual"
            baseline = BaselineIndividual(treino)
            flags_if = np.zeros(len(recente), dtype=bool)
            if cfg.usar_isolation_forest and len(treino) >= cfg.min_leituras_if:
                detector = self._detector(pet.id, treino)
                X = montar_matriz(recente)
                flags_if = detector.anomalo(X)
                score = float(detector.percentil_anomalia(X[-1:])[0])
            pontuais = self._desvios_individuais(baseline, recente, contexto, flags_if, agora_h)
            tendencia = self._tendencia_atividade(baseline, serie, agora_h)
            if tendencia:
                pontuais.append(tendencia)
            if flags_if[-1]:
                justificativa.append(
                    f"Isolation Forest: leitura atual mais incomum que {score:.0%} do histórico do pet."
                )
        else:
            modo = "referencia_populacional"
            justificativa.append(
                f"Histórico insuficiente ({len(treino)} leituras, {cobertura:.0f} h): "
                "usando faixas de referência por espécie, porte e idade."
            )
            pontuais = self._desvios_populacionais(pet, recente, contexto, agora_h)

        desvios.extend(pontuais)
        desvios.extend(self._criticos(pet, recente, contexto, agora_h))

        nivel = self._severidade(desvios, justificativa)
        contexto_clinico, nivel = self._contexto_clinico(
            historico_clinico, data_ref, desvios, nivel, justificativa
        )

        # Em MEDIO/ALTO o desvio pontual de outro sinal só polui a explicação.
        relevantes = desvios if nivel == "BAIXO" else [d for d in desvios if d.tipo != "pontual"]
        # O mais grave primeiro: é o que aparece na mensagem ao tutor.
        relevantes.sort(key=lambda d: (d.tipo != "critico", -abs(d.z_robusto or 0)))

        return ResultadoMotor(
            pet_id=pet.id,
            data_referencia=data_ref,
            nivel_risco=nivel,
            modo=modo,
            score_anomalia=round(score, 4),
            desvios=relevantes,
            contexto_clinico=contexto_clinico,
            justificativa=justificativa,
            leituras_validas=len(serie),
            valores_descartados=descartados,
            tipo_alerta=self._tipo_alerta(relevantes) if nivel else None,
        )

    # ------------------------------------------------------------ Isolation Forest

    def _detector(self, pet_id: int, treino: Serie) -> DetectorMultivariado:
        fim_treino = float(treino.horas[-1])
        with self._trava:
            em_cache = self._cache.get(pet_id)
            if em_cache and abs(fim_treino - em_cache[0]) < self.config.retreino_if_horas:
                self._cache.move_to_end(pet_id)
                return em_cache[1]

        detector = DetectorMultivariado(
            contaminacao=self.config.contaminacao_if, arvores=self.config.arvores_if
        ).treinar(montar_matriz(treino))

        with self._trava:
            self._cache[pet_id] = (fim_treino, detector)
            self._cache.move_to_end(pet_id)
            while len(self._cache) > self._max_cache:
                self._cache.popitem(last=False)
        return detector

    # ------------------------------------------------------------ desvios

    def _desvios_individuais(
        self,
        baseline: BaselineIndividual,
        recente: Serie,
        contexto: np.ndarray,
        flags_if: np.ndarray,
        agora_h: float,
    ) -> list[Desvio]:
        cfg = self.config
        desvios = []
        for sinal in SINAIS_PONTUAIS:
            valores = recente.valores[sinal]
            z, referencia = baseline.z(sinal, valores, contexto, recente.valores["atividade"])
            with np.errstate(invalid="ignore"):
                desviado = (np.abs(z) >= cfg.z_limiar) | ((np.abs(z) >= cfg.z_limiar_com_if) & flags_if)
            desvio = self._avaliar_janela(
                sinal, valores, referencia, z, desviado, contexto, recente.horas, agora_h
            )
            if desvio:
                desvios.append(desvio)
        return desvios

    def _desvios_populacionais(
        self, pet: PerfilPet, recente: Serie, contexto: np.ndarray, agora_h: float
    ) -> list[Desvio]:
        faixas = faixas_referencia(pet)
        desvios = []
        for sinal in SINAIS_PONTUAIS:
            valores = recente.valores[sinal]
            minimo, maximo = faixas[sinal]
            # FC em exercício é naturalmente alta: a faixa populacional só vale em repouso.
            aplicavel = em_repouso(contexto) if sinal == "frequencia_cardiaca" else np.ones(len(valores), bool)
            with np.errstate(invalid="ignore"):
                acima = aplicavel & (valores > maximo)
                abaixo = aplicavel & (valores < minimo)
            referencia = np.where(acima, maximo, np.where(abaixo, minimo, (minimo + maximo) / 2))
            desvio = self._avaliar_janela(
                sinal, valores, referencia, None, acima | abaixo, contexto, recente.horas, agora_h
            )
            if desvio:
                desvios.append(desvio)
        return desvios

    def _avaliar_janela(
        self,
        sinal: str,
        valores: np.ndarray,
        referencia: np.ndarray,
        z: np.ndarray | None,
        desviado: np.ndarray,
        contexto: np.ndarray,
        horas: np.ndarray,
        agora_h: float,
    ) -> Desvio | None:
        """Classifica o desvio de um sinal na janela recente como persistente ou pontual."""
        cfg = self.config
        validos = ~np.isnan(valores) & ~np.isnan(referencia)
        if not validos.any():
            return None

        diferenca = np.where(validos, valores - referencia, 0.0)
        acima = desviado & validos & (diferenca > 0)
        abaixo = desviado & validos & (diferenca < 0)
        selecionados = acima if acima.sum() >= abaixo.sum() else abaixo
        if not selecionados.any():
            return None

        fracao = selecionados.sum() / validos.sum()
        ultima_hora = validos & (horas > agora_h - 1.0)
        fracao_recente = selecionados[ultima_hora].sum() / max(ultima_hora.sum(), 1)
        persistente = (
            validos.sum() >= cfg.min_leituras_janela
            and fracao >= cfg.fracao_persistencia
            and fracao_recente >= 0.5
        )
        atual_desviado = bool(selecionados[np.flatnonzero(validos)[-1]])
        if not persistente and not atual_desviado:
            return None

        valor = float(np.median(valores[selecionados]))
        ref = float(np.median(referencia[selecionados]))
        contexto_dominante = Counter(contexto[selecionados].tolist()).most_common(1)[0][0]
        z_mediano = float(np.median(z[selecionados])) if z is not None else None
        return Desvio(
            sinal=sinal,
            direcao="acima" if selecionados is acima else "abaixo",
            tipo="persistente" if persistente else "pontual",
            valor_atual=round(valor, 1),
            referencia=round(ref, 1),
            desvio_percentual=round((valor - ref) / ref * 100, 1),
            z_robusto=None if z_mediano is None else round(z_mediano, 2),
            contexto=CONTEXTOS[int(contexto_dominante)],
            persistencia_horas=round(float(agora_h - horas[selecionados].min()), 1)
            if persistente
            else 0.0,
        )

    def _tendencia_atividade(
        self, baseline: BaselineIndividual, serie: Serie, agora_h: float
    ) -> Desvio | None:
        """Queda sustentada de movimentação em 24 h comparada à média diária habitual."""
        cfg = self.config
        est = baseline.atividade_diaria
        if est is None or est.mediana <= 0:
            return None
        ultimas_24h = serie.valores["atividade"][serie.horas > agora_h - 24]
        ultimas_24h = ultimas_24h[~np.isnan(ultimas_24h)]
        if len(ultimas_24h) < cfg.min_leituras_janela:
            return None
        media = float(np.mean(ultimas_24h))
        variacao = (media - est.mediana) / est.mediana
        z = (media - est.mediana) / est.escala
        if variacao > -cfg.queda_atividade or z > -cfg.z_atividade_diaria:
            return None
        return Desvio(
            sinal="atividade",
            direcao="abaixo",
            tipo="tendencia",
            valor_atual=round(media, 1),
            referencia=round(est.mediana, 1),
            desvio_percentual=round(variacao * 100, 1),
            z_robusto=round(z, 2),
            contexto="24h",
            persistencia_horas=24.0,
        )

    def _criticos(
        self, pet: PerfilPet, recente: Serie, contexto: np.ndarray, agora_h: float
    ) -> list[Desvio]:
        """Rede de segurança: limite crítico confirmado em 2 das 3 últimas leituras
        (uma leitura isolada é tratada como possível artefato do sensor)."""
        limites = CRITICOS[pet.especie]
        ultimas = slice(-3, None)
        regras = [
            ("temperatura", recente.valores["temperatura"][ultimas], *limites["temperatura"], np.ones(3, bool)),
            ("pressao", recente.valores["pressao"][ultimas], -np.inf, limites["pressao_max"], np.ones(3, bool)),
            (
                "frequencia_cardiaca",
                recente.valores["frequencia_cardiaca"][ultimas],
                *limites["fc_repouso"],
                em_repouso(contexto[ultimas]),
            ),
        ]
        desvios = []
        for sinal, valores, minimo, maximo, aplicavel in regras:
            aplicavel = aplicavel[-len(valores):]
            with np.errstate(invalid="ignore"):
                acima = aplicavel & (valores > maximo)
                abaixo = aplicavel & (valores < minimo)
            for mascara, direcao, limite in ((acima, "acima", maximo), (abaixo, "abaixo", minimo)):
                if mascara.sum() >= 2:
                    valor = float(np.nanmedian(valores[mascara]))
                    desvios.append(
                        Desvio(
                            sinal=sinal,
                            direcao=direcao,
                            tipo="critico",
                            valor_atual=round(valor, 1),
                            referencia=float(limite),
                            desvio_percentual=round((valor - limite) / limite * 100, 1),
                            contexto="repouso" if sinal == "frequencia_cardiaca" else "geral",
                            persistencia_horas=0.0,
                        )
                    )
        return desvios

    # ------------------------------------------------------------ severidade

    def _severidade(self, desvios: list[Desvio], justificativa: list[str]) -> NivelRisco | None:
        """Matriz de priorização da Sprint 3 (seção 8B)."""
        cfg = self.config
        criticos = [d for d in desvios if d.tipo == "critico"]
        sustentados = [d for d in desvios if d.tipo in ("persistente", "tendencia")]
        pontuais = [d for d in desvios if d.tipo == "pontual"]
        sinais = sorted({d.sinal for d in sustentados})

        forte = any(
            d.z_robusto is not None and abs(d.z_robusto) >= cfg.z_forte for d in sustentados
        ) or any(
            d.sinal == "temperatura" and d.direcao == "acima" and d.valor_atual >= TEMPERATURA_FEBRE
            for d in sustentados
        )

        if criticos:
            justificativa.append(
                "Limite crítico ultrapassado em leituras consecutivas: "
                + ", ".join(d.sinal for d in criticos)
                + " → ALTO."
            )
            return "ALTO"
        if len(sinais) >= 2 and forte:
            justificativa.append(f"Desvios sustentados combinados e severos ({', '.join(sinais)}) → ALTO.")
            return "ALTO"
        if len(sinais) >= 2:
            justificativa.append(f"Desvios sustentados combinados ({', '.join(sinais)}) → MEDIO.")
            return "MEDIO"
        if len(sinais) == 1:
            justificativa.append(f"Desvio persistente em {sinais[0]} → MEDIO.")
            return "MEDIO"
        if pontuais:
            justificativa.append(
                "Desvio isolado sem persistência ("
                + ", ".join(d.sinal for d in pontuais)
                + ") → BAIXO (registro e monitoramento passivo)."
            )
            return "BAIXO"
        return None

    def _contexto_clinico(
        self,
        historico: list[EventoClinico],
        data_ref: datetime,
        desvios: list[Desvio],
        nivel: NivelRisco | None,
        justificativa: list[str],
    ) -> tuple[list[str], NivelRisco | None]:
        cfg = self.config
        contexto: list[str] = []
        vacina_recente = None

        for evento in sorted(historico, key=lambda e: e.data, reverse=True):
            dias = (data_ref.date() - evento.data).days
            if dias < 0:
                continue
            descricao = (evento.descricao or evento.tipo.lower()).strip()
            data_txt = evento.data.strftime("%d/%m")
            if evento.tipo == "VACINA" and dias <= 30:
                contexto.append(f"Vacina aplicada em {data_txt}: {descricao}")
                horas = (data_ref - datetime.combine(evento.data, time(0))).total_seconds() / 3600
                if horas <= cfg.horas_reacao_vacinal and vacina_recente is None:
                    vacina_recente = descricao
            elif evento.tipo == "MEDICAMENTO" and (
                dias <= 30 or (evento.data_retorno and evento.data_retorno >= data_ref.date())
            ):
                contexto.append(f"Medicação registrada em {data_txt}: {descricao}")
            elif evento.tipo == "DOENCA" and dias <= 180:
                contexto.append(f"Condição clínica registrada em {data_txt}: {descricao}")
            elif evento.tipo in ("CONSULTA", "EXAME") and dias <= 15:
                contexto.append(f"{evento.tipo.capitalize()} em {data_txt}: {descricao}")

        sustentados = [d for d in desvios if d.tipo in ("persistente", "tendencia")]
        compativel_com_vacina = (
            vacina_recente is not None
            and sustentados
            and not any(d.tipo == "critico" for d in desvios)
            and all(self._compativel_com_reacao_vacinal(d) for d in sustentados)
        )
        if compativel_com_vacina and nivel in ("MEDIO", "ALTO"):
            novo = NIVEIS[NIVEIS.index(nivel) - 1]
            justificativa.append(
                f"Vacina recente ({vacina_recente}): alterações leves de temperatura e atividade são "
                f"esperadas até {cfg.horas_reacao_vacinal:.0f} h após a aplicação → rebaixado de {nivel} para {novo}."
            )
            nivel = novo

        return contexto[:4], nivel

    @staticmethod
    def _compativel_com_reacao_vacinal(desvio: Desvio) -> bool:
        if desvio.sinal == "temperatura":
            return desvio.direcao == "acima" and desvio.valor_atual < TEMPERATURA_FEBRE
        if desvio.sinal == "atividade":
            return desvio.direcao == "abaixo"
        if desvio.sinal == "frequencia_cardiaca":
            return desvio.direcao == "acima" and desvio.desvio_percentual <= 20
        return False

    @staticmethod
    def _tipo_alerta(desvios: list[Desvio]) -> str | None:
        sinais = []
        for d in sorted(desvios, key=lambda d: d.tipo != "critico"):
            if d.sinal not in sinais:
                sinais.append(d.sinal)
        if not sinais:
            return None
        return TIPO_ALERTA_MULTIPLO if len(sinais) > 1 else TIPOS_ALERTA[sinais[0]]
