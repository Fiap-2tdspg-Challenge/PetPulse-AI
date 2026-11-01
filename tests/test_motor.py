from datetime import date, datetime, timedelta

import pytest

from app.ml.motor import ConfigMotor, MotorAnomalias
from app.schemas import Desvio, EventoClinico
from tests.conftest import INICIO, gerar_leituras

ULTIMO_DIA = INICIO + timedelta(days=4)


def janela(inicio_h: float, fim_h: float):
    """Retorna um teste de 'data está entre ULTIMO_DIA+inicio_h e ULTIMO_DIA+fim_h'."""
    inicio = ULTIMO_DIA + timedelta(hours=inicio_h)
    fim = ULTIMO_DIA + timedelta(hours=fim_h)
    return lambda data: inicio <= data < fim


def ate(leituras, horas_no_ultimo_dia: float):
    limite = ULTIMO_DIA + timedelta(hours=horas_no_ultimo_dia)
    return [l for l in leituras if l.data_leitura <= limite]


def test_pet_saudavel_nao_gera_alerta(motor, pet):
    resultado = motor.analisar(pet, gerar_leituras())
    assert resultado.modo == "baseline_individual"
    assert resultado.nivel_risco is None
    assert not resultado.gerar_alerta


def test_taquicardia_em_repouso_persistente_gera_alerta_medio(motor, pet):
    no_episodio = janela(12, 20)

    def taquicardia(i, data, v):
        if no_episodio(data):
            v["atividade"] = 8
            v["fc"] = (80 + 0.8 * 8) * 1.3

    resultado = motor.analisar(pet, ate(gerar_leituras(alteracao=taquicardia), 17))
    assert resultado.nivel_risco == "MEDIO"
    assert resultado.gerar_alerta
    assert resultado.tipo_alerta == "FREQUENCIA_CARDIACA"
    desvio = resultado.desvios[0]
    assert desvio.tipo == "persistente" and desvio.direcao == "acima"
    assert 20 <= desvio.desvio_percentual <= 40


def test_pico_isolado_e_tratado_como_artefato(motor, pet):
    pico = ULTIMO_DIA + timedelta(hours=14)

    def artefato(i, data, v):
        if data == pico:
            v["fc"] = 210

    resultado = motor.analisar(pet, ate(gerar_leituras(alteracao=artefato), 14))
    assert resultado.nivel_risco == "BAIXO"
    assert not resultado.gerar_alerta


def test_febre_com_queda_de_atividade_gera_alerta_alto(motor, pet):
    no_episodio = janela(-14, 24)

    def febre(i, data, v):
        if no_episodio(data):
            v["atividade"] = min(v["atividade"] * 0.3, 8)
            v["temperatura"] = 39.9

    resultado = motor.analisar(pet, ate(gerar_leituras(alteracao=febre), 12))
    assert resultado.nivel_risco == "ALTO"
    assert {d.sinal for d in resultado.desvios} >= {"temperatura", "atividade"}


def test_reacao_vacinal_leve_e_rebaixada(motor, pet):
    no_episodio = janela(-14, 24)

    def reacao(i, data, v):
        if no_episodio(data):
            v["atividade"] *= 0.6
            v["temperatura"] += 0.5

    vacina = EventoClinico(tipo="VACINA", descricao="V10", data=date(2026, 10, 4))
    leituras = ate(gerar_leituras(alteracao=reacao), 12)

    sem_contexto = motor.analisar(pet, leituras)
    com_contexto = MotorAnomalias().analisar(pet, leituras, [vacina])

    assert sem_contexto.gerar_alerta
    assert not com_contexto.gerar_alerta
    assert any("Vacina recente" in j for j in com_contexto.justificativa)
    assert com_contexto.contexto_clinico[0].startswith("Vacina aplicada em 04/10")


def test_limite_critico_confirmado_gera_alto_imediato(motor, pet):
    def hipertensao(i, data, v):
        if data >= ULTIMO_DIA + timedelta(hours=11, minutes=40):
            v["pressao"] = 195

    resultado = motor.analisar(pet, ate(gerar_leituras(alteracao=hipertensao), 12))
    assert resultado.nivel_risco == "ALTO"
    assert any(d.tipo == "critico" and d.sinal == "pressao" for d in resultado.desvios)


def test_cold_start_usa_faixas_populacionais(motor, pet):
    leituras = gerar_leituras(dias=0.5)
    resultado = motor.analisar(pet, leituras)
    assert resultado.modo == "referencia_populacional"
    assert resultado.nivel_risco is None


def test_cold_start_detecta_fc_fora_da_faixa_da_especie(motor, pet):
    def fc_alta(i, data, v):
        v["fc"] = 165

    resultado = motor.analisar(pet, gerar_leituras(dias=0.5, alteracao=fc_alta))
    assert resultado.modo == "referencia_populacional"
    assert resultado.nivel_risco in ("MEDIO", "ALTO")


def test_funciona_sem_temperatura_como_a_api_atual(motor, pet):
    resultado = motor.analisar(pet, gerar_leituras(com_temperatura=False))
    assert resultado.nivel_risco is None


def test_ablacao_sem_baseline_individual(pet):
    motor = MotorAnomalias(ConfigMotor(usar_baseline_individual=False))
    assert motor.analisar(pet, gerar_leituras()).modo == "referencia_populacional"


def test_modelo_isolation_forest_e_reaproveitado_no_cache(motor, pet):
    leituras = gerar_leituras()
    motor.analisar(pet, leituras[:-1])
    _, detector = motor._cache[pet.id]
    motor.analisar(pet, leituras)
    assert motor._cache[pet.id][1] is detector


def test_sem_leituras_validas_gera_erro(motor, pet):
    from app.schemas import Leitura

    with pytest.raises(ValueError):
        motor.analisar(pet, [Leitura(data_leitura=datetime(2026, 10, 1), frequencia_cardiaca=999)])


def _desvio(sinal: str, tipo: str, z: float | None = 4.0, valor: float = 100.0) -> Desvio:
    return Desvio(
        sinal=sinal, direcao="acima", tipo=tipo, valor_atual=valor, referencia=80.0,
        desvio_percentual=25.0, z_robusto=z, contexto="repouso", persistencia_horas=6.0,
    )


@pytest.mark.parametrize(
    ("desvios", "esperado"),
    [
        ([], None),
        ([_desvio("frequencia_cardiaca", "pontual")], "BAIXO"),
        ([_desvio("frequencia_cardiaca", "persistente")], "MEDIO"),
        ([_desvio("frequencia_cardiaca", "persistente"), _desvio("pressao", "persistente")], "MEDIO"),
        ([_desvio("frequencia_cardiaca", "persistente", z=6), _desvio("pressao", "persistente")], "ALTO"),
        ([_desvio("temperatura", "persistente", valor=39.8), _desvio("atividade", "tendencia")], "ALTO"),
        ([_desvio("pressao", "critico", z=None)], "ALTO"),
    ],
)
def test_matriz_de_severidade(motor, desvios, esperado):
    assert motor._severidade(desvios, []) == esperado
