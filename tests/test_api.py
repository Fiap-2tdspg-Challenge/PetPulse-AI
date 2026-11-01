from dataclasses import replace
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

import app.main as main
from tests.conftest import INICIO, gerar_leituras


@pytest.fixture
def cliente():
    return TestClient(main.app)


def corpo(leituras, **extra):
    return {
        "pet": {"id": 1, "nome": "Thor", "especie": "cao", "porte": "grande", "idadeAnos": 6, "sexo": "M"},
        "leituras": [l.model_dump(mode="json", by_alias=True) for l in leituras],
        "historicoClinico": [{"tipo": "VACINA", "descricao": "V10", "data": "2026-03-12"}],
        **extra,
    }


def test_health(cliente):
    resposta = cliente.get("/health")
    assert resposta.status_code == 200
    assert resposta.json()["status"] == "UP"


def test_analise_de_pet_saudavel(cliente):
    resposta = cliente.post("/v1/analises", json=corpo(gerar_leituras()))
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["petId"] == 1
    assert dados["gerarAlerta"] is False
    assert dados["modo"] == "baseline_individual"
    assert dados["alerta"] is None


def test_analise_com_anomalia_devolve_alerta_em_camel_case(cliente):
    inicio = INICIO + timedelta(days=4, hours=8)

    def taquicardia(i, data, v):
        if data >= inicio:
            v["atividade"] = 8
            v["fc"] = 120

    leituras = [l for l in gerar_leituras(alteracao=taquicardia) if l.data_leitura <= inicio + timedelta(hours=5)]
    dados = cliente.post("/v1/analises", json=corpo(leituras, gerarTexto=False)).json()
    assert dados["gerarAlerta"] is True
    assert dados["nivelRisco"] in ("MEDIO", "ALTO")
    assert dados["tipoAlerta"] == "FREQUENCIA_CARDIACA"
    assert dados["alerta"]["geradoPor"] == "template"
    assert "Thor" in dados["alerta"]["mensagem"]
    assert dados["desvios"][0]["desvioPercentual"] > 20


def test_analises_ficam_gravadas_no_historico(cliente):
    cliente.post("/v1/analises", json=corpo(gerar_leituras()))
    historico = cliente.get("/v1/analises", params={"petId": 1, "limite": 1}).json()
    assert len(historico) == 1
    registro = historico[0]
    assert registro["petId"] == 1
    assert registro["modo"] == "baseline_individual"
    assert registro["gerarAlerta"] is False
    assert isinstance(registro["justificativa"], list)
    assert cliente.get("/v1/analises", params={"petId": 999}).json() == []


def test_payload_invalido_retorna_422(cliente):
    assert cliente.post("/v1/analises", json=corpo([])).status_code == 422


def test_chave_de_servico_obrigatoria_quando_configurada(cliente, monkeypatch):
    monkeypatch.setattr(main, "configuracao", replace(main.configuracao, chave_servico="segredo"))
    leituras = gerar_leituras(dias=1)
    assert cliente.post("/v1/analises", json=corpo(leituras)).status_code == 401
    ok = cliente.post("/v1/analises", json=corpo(leituras), headers={"X-API-Key": "segredo"})
    assert ok.status_code == 200
