"""Gera o dataset sintético rotulado usado no treino, nos testes e na avaliação.

Não existe base pública de telemetria de coleiras com anomalias rotuladas, então
simulamos 14 dias de leituras (uma a cada 10 min) para 4 pets com perfis fisiológicos
diferentes e injetamos episódios conhecidos — alguns que DEVEM gerar alerta e outros
que NÃO devem (reação vacinal esperada, artefato de sensor).

Uso:  python dados/gerar_dataset.py                                    (desenvolvimento, semente 42)
      python dados/gerar_dataset.py --semente 2026 --saida dados/teste  (teste, dados nunca vistos)
Saída: leituras.csv, pets.json e episodios.json na pasta escolhida
"""

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PASTA = Path(__file__).parent
INICIO = datetime(2026, 10, 1)
DIAS = 14
PASSO_MIN = 10
SEMENTE = 42

PETS = [
    {
        "id": 1, "nome": "Thor", "especie": "cao", "raca": "Labrador", "porte": "grande",
        "idadeAnos": 6, "pesoKg": 32.0, "sexo": "M",
        "fisiologia": {"fc_repouso": 74, "fc_sono": 60, "fc_por_atividade": 0.9, "temp": 38.4,
                       "pressao": 132, "passeios": [7.0, 18.5], "prob_corrida": 0.5},
        "historicoClinico": [
            {"tipo": "VACINA", "descricao": "V10 (reforço anual)", "data": "2026-03-12"},
            {"tipo": "CONSULTA", "descricao": "Check-up anual sem alterações", "data": "2026-10-03"},
        ],
    },
    {
        "id": 2, "nome": "Mel", "especie": "cao", "raca": "Chihuahua", "porte": "pequeno",
        "idadeAnos": 11, "pesoKg": 2.8, "sexo": "F",
        "fisiologia": {"fc_repouso": 118, "fc_sono": 104, "fc_por_atividade": 0.7, "temp": 38.7,
                       "pressao": 145, "passeios": [8.0, 17.5], "prob_corrida": 0.1},
        "historicoClinico": [
            {"tipo": "MEDICAMENTO", "descricao": "Condroitina (suplemento articular)",
             "data": "2026-10-02", "dataRetorno": "2026-11-02"},
            {"tipo": "VACINA", "descricao": "Antirrábica", "data": "2026-10-09"},
        ],
    },
    {
        "id": 3, "nome": "Nina", "especie": "gato", "raca": "SRD", "porte": "pequeno",
        "idadeAnos": 4, "pesoKg": 4.2, "sexo": "F",
        "fisiologia": {"fc_repouso": 165, "fc_sono": 150, "fc_por_atividade": 0.6, "temp": 38.6,
                       "pressao": 135, "passeios": [], "prob_corrida": 0.0},
        "historicoClinico": [
            {"tipo": "EXAME", "descricao": "Hemograma de rotina normal", "data": "2026-10-04"},
        ],
    },
    {
        "id": 4, "nome": "Bob", "especie": "cao", "raca": "Border Collie", "porte": "medio",
        "idadeAnos": 3, "pesoKg": 18.0, "sexo": "M",
        "fisiologia": {"fc_repouso": 58, "fc_sono": 47, "fc_por_atividade": 1.0, "temp": 38.5,
                       "pressao": 128, "passeios": [6.5, 12.5, 19.0], "prob_corrida": 0.9},
        "historicoClinico": [],
    },
]

# deveAlertar=False: situações que um bom sistema NÃO deve transformar em alerta.
EPISODIOS = [
    {"id": "E1", "sinais": ["frequencia_cardiaca:acima"], "petId": 1, "inicio": "2026-10-10T13:00", "horas": 8, "deveAlertar": True,
     "nivelEsperado": "MEDIO",
     "descricao": "Taquicardia em repouso: FC 30% acima do padrão do Thor (~107 bpm, abaixo do limite fixo de 180)"},
    {"id": "E2", "sinais": ["frequencia_cardiaca:acima"], "petId": 1, "inicio": "2026-10-12T10:20", "horas": 0.1, "deveAlertar": False,
     "nivelEsperado": None, "descricao": "Artefato de sensor: uma leitura isolada de 205 bpm"},
    {"id": "E3", "sinais": ["temperatura:acima", "atividade:abaixo"], "petId": 2, "inicio": "2026-10-09T10:00", "horas": 30, "deveAlertar": False,
     "nivelEsperado": "BAIXO", "descricao": "Reação vacinal esperada: +0,5 °C e -35% de atividade"},
    {"id": "E4", "sinais": ["temperatura:acima", "atividade:abaixo", "frequencia_cardiaca:acima"], "petId": 2, "inicio": "2026-10-13T02:00", "horas": 30, "deveAlertar": True,
     "nivelEsperado": "ALTO", "descricao": "Febre em repouso com prostração (39,9 °C, -60% de atividade)"},
    {"id": "E5", "sinais": ["pressao:acima"], "petId": 3, "inicio": "2026-10-09T08:00", "horas": 60, "deveAlertar": True,
     "nivelEsperado": "MEDIO", "descricao": "Hipertensão progressiva: pressão sobe até +45 mmHg"},
    {"id": "E6", "sinais": ["atividade:abaixo"], "petId": 3, "inicio": "2026-10-13T00:00", "horas": 36, "deveAlertar": True,
     "nivelEsperado": "MEDIO", "descricao": "Prostração: queda de 55% na movimentação"},
    {"id": "E7", "sinais": ["frequencia_cardiaca:acima"], "petId": 4, "inicio": "2026-10-11T15:00", "horas": 7, "deveAlertar": True,
     "nivelEsperado": "MEDIO",
     "descricao": "Taquicardia em repouso: FC 38% acima do padrão do Bob (~92 bpm, dentro da faixa populacional)"},
]


def ruido_ar1(rng: np.random.Generator, n: int, sigma: float, phi: float = 0.7) -> np.ndarray:
    """Ruído autocorrelacionado: sinais vitais não pulam aleatoriamente a cada leitura."""
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal(0, sigma)
    return e


def atividade_cao(rng, horas, dia_semana, fis) -> np.ndarray:
    noite = (horas >= 22) | (horas < 6)
    a = np.where(noite, rng.uniform(0, 8, len(horas)), rng.uniform(4, 24, len(horas)))
    fim_de_semana = dia_semana >= 5
    for base in fis["passeios"]:
        inicio = base + rng.normal(0, 0.3)
        duracao = rng.uniform(0.5, 1.0) * (1.6 if fim_de_semana else 1.0)
        passeio = (horas >= inicio) & (horas < inicio + duracao)
        a[passeio] = rng.uniform(35, 58, passeio.sum())
        if rng.random() < fis["prob_corrida"]:
            c0 = inicio + rng.uniform(0, max(duracao - 0.3, 0.05))
            corrida = (horas >= c0) & (horas < c0 + rng.uniform(0.15, 0.35))
            a[corrida] = rng.uniform(65, 92, corrida.sum())
    for _ in range(rng.integers(1, 4)):
        t = rng.uniform(9, 21)
        brincadeira = (horas >= t) & (horas < t + 1 / 6)
        a[brincadeira] = rng.uniform(32, 70, brincadeira.sum())
    return a


def atividade_gato(rng, horas, dia_semana, fis) -> np.ndarray:
    a = rng.uniform(0, 18, len(horas))
    for janela in ((4.0, 6.5), (19.5, 23.0)):
        for _ in range(rng.integers(2, 4)):
            t = rng.uniform(*janela)
            surto = (horas >= t) & (horas < t + rng.uniform(0.15, 0.35))
            a[surto] = rng.uniform(35, 80, surto.sum())
    return a


def gerar_pet(rng: np.random.Generator, pet: dict) -> dict[str, np.ndarray]:
    fis = pet["fisiologia"]
    n_dia = 24 * 60 // PASSO_MIN
    horas_dia = np.arange(n_dia) * PASSO_MIN / 60
    gerador = atividade_gato if pet["especie"] == "gato" else atividade_cao

    atividade = np.concatenate(
        [gerador(rng, horas_dia, (INICIO + timedelta(days=d)).weekday(), fis) for d in range(DIAS)]
    )
    n = len(atividade)
    hora = np.tile(horas_dia, DIAS)
    dormindo = ((hora >= 22) | (hora < 6)) & (atividade < 10)

    fc_base = np.where(dormindo, fis["fc_sono"], fis["fc_repouso"])
    fc = fc_base + fis["fc_por_atividade"] * atividade + ruido_ar1(rng, n, 3.0)
    temperatura = (
        fis["temp"] + 0.004 * atividade + 0.12 * np.sin(2 * np.pi * (hora - 16) / 24)
        + ruido_ar1(rng, n, 0.04)
    )
    pressao = fis["pressao"] + 0.25 * atividade + ruido_ar1(rng, n, 3.0)
    return {"fc": fc, "atividade": atividade, "temperatura": temperatura, "pressao": pressao}


def injetar_episodios(rng, pet_id: int, s: dict[str, np.ndarray], datas: list[datetime], fis: dict):
    rotulo = np.array([""] * len(datas), dtype=object)
    for ep in (e for e in EPISODIOS if e["petId"] == pet_id):
        inicio = datetime.fromisoformat(ep["inicio"])
        fim = inicio + timedelta(hours=ep["horas"])
        m = np.array([inicio <= d < fim for d in datas])
        k = int(m.sum())
        rotulo[m] = ep["id"]
        progresso = np.linspace(0, 1, k)

        if ep["id"] in ("E1", "E7"):  # taquicardia com o pet parado
            s["atividade"][m] = rng.uniform(3, 15, k)
            alvo = 1.30 if ep["id"] == "E1" else 1.38
            fc_habitual = fis["fc_repouso"] + fis["fc_por_atividade"] * s["atividade"][m]
            s["fc"][m] = fc_habitual * alvo + rng.normal(0, 3, k)
        elif ep["id"] == "E2":  # uma leitura isolada absurda, mas fisicamente possível
            s["fc"][m] = 205
        elif ep["id"] == "E3":  # reação vacinal leve
            s["temperatura"][m] += 0.5 * np.sin(np.pi * progresso) + 0.05
            s["atividade"][m] *= 0.65
        elif ep["id"] == "E4":  # febre + prostração
            s["atividade"][m] = np.minimum(s["atividade"][m] * 0.4, 12)
            s["temperatura"][m] = fis["temp"] + np.minimum(progresso * 4, 1) * 1.2 + rng.normal(0, 0.05, k)
            s["fc"][m] = fis["fc_repouso"] * 1.15 + rng.normal(0, 3, k)
        elif ep["id"] == "E5":  # hipertensão que piora aos poucos
            s["pressao"][m] += 45 * np.minimum(progresso * 1.5, 1)
        elif ep["id"] == "E6":  # prostração
            s["atividade"][m] *= 0.45
    return rotulo


def gerar(semente: int = SEMENTE, pasta: Path = PASTA) -> None:
    pasta.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(semente)
    datas = [INICIO + timedelta(minutes=PASSO_MIN * i) for i in range(DIAS * 24 * 60 // PASSO_MIN)]
    linhas = []
    for pet in PETS:
        s = gerar_pet(rng, pet)
        rotulo = injetar_episodios(rng, pet["id"], s, datas, pet["fisiologia"])

        # Imperfeições reais: falhas de leitura e valores impossíveis vindos do sensor.
        n = len(datas)
        sem_fc = rng.random(n) < 0.01
        sem_temp = rng.random(n) < 0.005
        lixo = rng.choice(n, size=4, replace=False)

        for i, data in enumerate(datas):
            fc = None if sem_fc[i] else round(float(s["fc"][i]))
            if i in lixo:
                fc = int(rng.choice([0, 999]))
            linhas.append({
                "petId": pet["id"],
                "dataLeitura": data.isoformat(timespec="minutes"),
                "frequenciaCardiaca": fc,
                "nivelAtividade": round(float(np.clip(s["atividade"][i], 0, 100)), 1),
                "pressao": round(float(s["pressao"][i])),
                "temperatura": None if sem_temp[i] else round(float(s["temperatura"][i]), 2),
                "episodio": rotulo[i],
            })

    with open(pasta / "leituras.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)

    pets_publicos = [{k: v for k, v in p.items() if k != "fisiologia"} for p in PETS]
    (pasta / "pets.json").write_text(json.dumps(pets_publicos, ensure_ascii=False, indent=2), encoding="utf-8")
    (pasta / "episodios.json").write_text(json.dumps(EPISODIOS, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(linhas)} leituras de {len(PETS)} pets e {len(EPISODIOS)} episódios rotulados gerados em {pasta.name}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera o dataset sintético rotulado.")
    parser.add_argument("--semente", type=int, default=SEMENTE, help="semente aleatória (42 = desenvolvimento)")
    parser.add_argument("--saida", type=Path, default=PASTA, help="pasta de saída")
    args = parser.parse_args()
    gerar(args.semente, args.saida)


if __name__ == "__main__":
    main()
