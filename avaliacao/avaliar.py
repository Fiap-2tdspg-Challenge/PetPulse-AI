"""Avaliação do motor contra as regras fixas que o PetPulse usa hoje.

Simula a coleira enviando uma leitura a cada 10 min durante 7 dias para 4 pets e, a
cada leitura, pergunta a cada método se o tutor deve ser alertado — exatamente como a
API Java fará em produção. Os alertas passam pela mesma deduplicação (um alerta por pet
a cada 6 h, salvo escalada de nível), e são comparados com os episódios rotulados.

Além das métricas por episódio (o que o tutor percebe), calcula as métricas por leitura
com o scikit-learn, como nos labs (precision_score, recall_score, f1_score, matriz de confusão).

Uso:  python -m avaliacao.avaliar                                   (dataset de desenvolvimento)
      python -m avaliacao.avaliar --dados dados/teste --saida avaliacao/teste   (dataset de teste)
Saída: <saida>/resultados.md e <saida>/resultados.json
"""

import argparse
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.ml.motor import ConfigMotor, MotorAnomalias
from app.ml.preparacao import preparar
from avaliacao.dataset import PASTA_DADOS, carregar_episodios, carregar_leituras, carregar_pets

PASTA = Path(__file__).parent
INICIO_AVALIACAO = datetime(2026, 10, 8)
COOLDOWN_H = 6
TOLERANCIA_FIM_H = 2
NIVEIS = {None: 0, "BAIXO": 1, "MEDIO": 2, "ALTO": 3}


# --------------------------------------------------------------- métodos de comparação

def _pior(disparos: list[tuple[str, str]]) -> tuple[str | None, set[str]]:
    """De uma lista (nível, sinal) devolve o nível mais grave e os sinais que o causaram."""
    if not disparos:
        return None, set()
    nivel = max((n for n, _ in disparos), key=NIVEIS.get)
    return nivel, {s for n, s in disparos if n == nivel}


def regras_api_java(fc, atividade, pressao, temperatura):
    """IoTReadingAnalysisService atual: FC <60 ou >180, pressão >16 cmHg (160 mmHg),
    atividade <2 na escala 0–10 da API (20% na escala da coleira)."""
    disparos = []
    if fc is not None and fc < 60:
        disparos.append(("ALTO", "frequencia_cardiaca:abaixo"))
    if fc is not None and fc > 180:
        disparos.append(("ALTO", "frequencia_cardiaca:acima"))
    if pressao is not None and pressao > 160:
        disparos.append(("ALTO", "pressao:acima"))
    if atividade is not None and atividade < 20:
        disparos.append(("MEDIO", "atividade:abaixo"))
    return _pior(disparos)


def _faixa(disparos, sinal, valor, minimo, maximo, nivel):
    if valor < minimo:
        disparos.append((nivel, f"{sinal}:abaixo"))
    elif valor > maximo:
        disparos.append((nivel, f"{sinal}:acima"))


def faixas_firmware(fc, atividade, pressao, temperatura):
    """Regras clínicas populacionais do firmware da coleira (PetPulse-Iot.ino)."""
    disparos = []
    if fc is not None:
        if fc > 180:
            disparos.append(("ALTO", "frequencia_cardiaca:acima"))
        else:
            _faixa(disparos, "frequencia_cardiaca", fc, 60, 140, "MEDIO")
    if pressao is not None:
        if pressao > 200:
            disparos.append(("ALTO", "pressao:acima"))
        else:
            _faixa(disparos, "pressao", pressao, 100, 160, "MEDIO")
    if temperatura is not None:
        febre_em_repouso = atividade is not None and temperatura >= 39.5 and atividade <= 30
        if temperatura >= 40.5 or febre_em_repouso:
            disparos.append(("ALTO", "temperatura:acima"))
        elif temperatura <= 36.0:
            disparos.append(("ALTO", "temperatura:abaixo"))
        else:
            _faixa(disparos, "temperatura", temperatura, 37.5, 39.2, "MEDIO")
    return _pior(disparos)


@dataclass
class Avaliacao:
    nome: str
    alertas: list[tuple[int, datetime, str, set[str]]] = field(default_factory=list)
    leituras_sinalizadas: int = 0
    niveis_por_episodio: dict[str, int] = field(default_factory=dict)
    # Por leitura: 1 = dentro de um episódio que deve alertar / método disse MEDIO ou ALTO
    y_real: list[int] = field(default_factory=list)
    y_previsto: list[int] = field(default_factory=list)


def deduplicar(
    pet_id: int, data: datetime, nivel: str | None, sinais: set[str], estado: dict, avaliacao: Avaliacao
) -> None:
    """Mesma política da API Java: um alerta aberto por pet a cada 6 h, salvo escalada."""
    if nivel not in ("MEDIO", "ALTO"):
        return
    avaliacao.leituras_sinalizadas += 1
    ultimo = estado.get(pet_id)
    if ultimo and data - ultimo[0] < timedelta(hours=COOLDOWN_H) and NIVEIS[nivel] <= NIVEIS[ultimo[1]]:
        return
    estado[pet_id] = (data, nivel)
    avaliacao.alertas.append((pet_id, data, nivel, sinais))


def episodio_em(pet_id: int, data: datetime, sinais: set[str], episodios: list[dict]) -> dict | None:
    """Episódio a que o alerta corresponde: mesma janela de tempo E mesmo sinal vital na
    mesma direção ("frequencia_cardiaca:acima"). Um alerta de FC baixa durante o sono, no
    meio de um episódio de taquicardia, é coincidência, não detecção."""
    for ep in episodios:
        if (
            ep["petId"] == pet_id
            and ep["inicio"] <= data <= ep["fim"] + timedelta(hours=TOLERANCIA_FIM_H)
            and sinais & set(ep["sinais"])
        ):
            return ep
    return None


def em_episodio_positivo(pet_id: int, data: datetime, episodios: list[dict]) -> int:
    return int(any(
        ep["deveAlertar"] and ep["petId"] == pet_id and ep["inicio"] <= data <= ep["fim"] for ep in episodios
    ))


def simular(pasta_dados: Path = PASTA_DADOS) -> tuple[list[Avaliacao], list[dict]]:
    pets = carregar_pets(pasta_dados)
    leituras = carregar_leituras(pasta_dados)
    episodios = carregar_episodios(pasta_dados)

    motores = {
        "Baseline individual (sem Isolation Forest)": MotorAnomalias(ConfigMotor(usar_isolation_forest=False)),
        "Motor PetPulse-AI (baseline + Isolation Forest)": MotorAnomalias(),
    }
    regras = {"Regras atuais da API Java": regras_api_java, "Faixas fixas do firmware": faixas_firmware}
    avaliacoes = {nome: Avaliacao(nome) for nome in [*regras, *motores]}
    estados = {nome: {} for nome in avaliacoes}

    for pet_id, (perfil, historico) in pets.items():
        serie = preparar([l for l, _ in leituras[pet_id]])
        indices = [i for i, d in enumerate(serie.datas) if d >= INICIO_AVALIACAO]
        print(f"  {perfil.nome}: {len(indices)} leituras avaliadas")
        for i in indices:
            data = serie.datas[i]
            v = {nome: serie.valores[nome][i] for nome in serie.valores}
            v = {k: None if np.isnan(x) else float(x) for k, x in v.items()}
            saidas = {
                nome: regra(v["frequencia_cardiaca"], v["atividade"], v["pressao"], v["temperatura"])
                for nome, regra in regras.items()
            }
            janela = serie.recorte(np.arange(len(serie)) <= i)
            for nome, motor in motores.items():
                r = motor.analisar_serie(perfil, janela, historico)
                sinais = {f"{d.sinal}:{d.direcao}" for d in r.desvios} if r.nivel_risco else set()
                saidas[nome] = (r.nivel_risco, sinais)

            real = em_episodio_positivo(pet_id, data, episodios)
            for nome, (nivel, sinais) in saidas.items():
                avaliacoes[nome].y_real.append(real)
                avaliacoes[nome].y_previsto.append(int(nivel in ("MEDIO", "ALTO")))
                deduplicar(pet_id, data, nivel, sinais, estados[nome], avaliacoes[nome])
                ep = episodio_em(pet_id, data, sinais, episodios)
                if ep and data <= ep["fim"]:
                    atual = avaliacoes[nome].niveis_por_episodio.get(ep["id"], 0)
                    avaliacoes[nome].niveis_por_episodio[ep["id"]] = max(atual, NIVEIS[nivel])

    return list(avaliacoes.values()), episodios


def metricas(av: Avaliacao, episodios: list[dict], pets_dias: float) -> dict:
    positivos = [e for e in episodios if e["deveAlertar"]]
    verdadeiros, latencias, detectados = 0, {}, set()
    falsos_em_episodio_negativo = set()
    for pet_id, data, _, sinais in av.alertas:
        ep = episodio_em(pet_id, data, sinais, episodios)
        if ep and ep["deveAlertar"]:
            verdadeiros += 1
            if ep["id"] not in detectados:
                detectados.add(ep["id"])
                latencias[ep["id"]] = (data - ep["inicio"]).total_seconds() / 3600
        elif ep:
            falsos_em_episodio_negativo.add(ep["id"])

    emitidos = len(av.alertas)
    precisao = verdadeiros / emitidos if emitidos else 0.0
    recall = len(detectados) / len(positivos)
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0

    y_real, y_prev = av.y_real, av.y_previsto
    por_leitura = {
        "acuracia": round(accuracy_score(y_real, y_prev), 3),
        "precisao": round(precision_score(y_real, y_prev, zero_division=0), 3),
        "recall": round(recall_score(y_real, y_prev, zero_division=0), 3),
        "f1": round(f1_score(y_real, y_prev, zero_division=0), 3),
        "matrizConfusao": confusion_matrix(y_real, y_prev, labels=[0, 1]).tolist(),
        "relatorio": classification_report(
            y_real, y_prev, labels=[0, 1], target_names=["normal", "anomalia"], zero_division=0
        ),
    }
    return {
        "metodo": av.nome,
        "leiturasSinalizadas": av.leituras_sinalizadas,
        "alertasEmitidos": emitidos,
        "alertasVerdadeiros": verdadeiros,
        "falsosAlarmesPorPetDia": round((emitidos - verdadeiros) / pets_dias, 2),
        "precisao": round(precisao, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "episodiosDetectados": sorted(detectados),
        "latenciaMedianaH": round(statistics.median(latencias.values()), 1) if latencias else None,
        "latenciaPorEpisodioH": {k: round(v, 1) for k, v in sorted(latencias.items())},
        "alertouEmEpisodioQueNaoDevia": sorted(falsos_em_episodio_negativo),
        "nivelMaximoPorEpisodio": {
            k: [n for n, c in NIVEIS.items() if c == v][0] for k, v in sorted(av.niveis_por_episodio.items())
        },
        "porLeitura": por_leitura,
    }


def escrever_relatorio(
    resultados: list[dict], episodios: list[dict], pets_dias: float, pasta_saida: Path, titulo: str
) -> None:
    def pct(x):
        return f"{x * 100:.0f}%"

    linhas = [
        f"# {titulo}",
        "",
        f"Simulação de {pets_dias:.0f} pet-dias (4 pets × 7 dias, uma leitura a cada 10 min). "
        f"Todos os métodos passam pela mesma deduplicação de alertas ({COOLDOWN_H} h por pet, salvo escalada). "
        "Um alerta só conta como verdadeiro se ocorrer durante o episódio (ou até 2 h depois) **e** for "
        "disparado pelo mesmo sinal vital, na mesma direção (ex.: FC acima do normal). "
        "Gerado por `python -m avaliacao.avaliar`.",
        "",
        "| Método | Alertas enviados | Precisão | Recall (episódios) | F1 | Falsos alarmes / pet-dia | Latência mediana | Leituras sinalizadas* |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in resultados:
        latencia = f"{r['latenciaMedianaH']} h" if r["latenciaMedianaH"] is not None else "—"
        linhas.append(
            f"| {r['metodo']} | {r['alertasEmitidos']} | {pct(r['precisao'])} | "
            f"{len(r['episodiosDetectados'])}/5 ({pct(r['recall'])}) | {r['f1']:.2f} | "
            f"{r['falsosAlarmesPorPetDia']} | {latencia} | {r['leiturasSinalizadas']} |"
        )
    linhas += [
        "",
        "\\* Leituras sinalizadas = leituras classificadas como MÉDIO ou ALTO antes da deduplicação. "
        "Para as regras atuais da API, é o número de alertas que o `IoTReadingAnalysisService` grava hoje, "
        "porque ele cria um alerta por leitura fora da faixa.",
        "",
        "## Episódio a episódio",
        "",
        "| Episódio | Pet | Deve alertar? | Esperado | " + " | ".join(r["metodo"] for r in resultados) + " |",
        "|---|---|---|---|" + "---|" * len(resultados),
    ]
    nomes = {1: "Thor", 2: "Mel", 3: "Nina", 4: "Bob"}
    for ep in episodios:
        celulas = []
        for r in resultados:
            nivel = r["nivelMaximoPorEpisodio"].get(ep["id"]) or "—"
            if ep["id"] in r["latenciaPorEpisodioH"]:
                celulas.append(f"{nivel} (alerta em {r['latenciaPorEpisodioH'][ep['id']]} h)")
            elif ep["id"] in r["alertouEmEpisodioQueNaoDevia"]:
                celulas.append(f"{nivel} ⚠ alertou")
            else:
                celulas.append(f"{nivel}")
        linhas.append(
            f"| {ep['id']}: {ep['descricao']} | {nomes[ep['petId']]} | {'sim' if ep['deveAlertar'] else 'não'} | "
            f"{ep['nivelEsperado'] or '—'} | " + " | ".join(celulas) + " |"
        )

    linhas += [
        "",
        "## Métricas por leitura (scikit-learn)",
        "",
        "Cada leitura é uma amostra: classe real 1 se está dentro de um episódio que deve alertar, "
        "e previsão 1 se o método a classificou como MÉDIO ou ALTO. São as mesmas funções dos labs "
        "(`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`).",
        "",
        "| Método | Acurácia | Precisão | Recall | F1 | Matriz de confusão [[VN, FP], [FN, VP]] |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in resultados:
        m = r["porLeitura"]
        linhas.append(
            f"| {r['metodo']} | {pct(m['acuracia'])} | {pct(m['precisao'])} | {pct(m['recall'])} | "
            f"{m['f1']:.2f} | `{m['matrizConfusao']}` |"
        )
    linhas += [
        "",
        "O recall por leitura do motor é menor que o por episódio porque ele espera algumas horas de "
        "persistência antes de alertar: as primeiras leituras de cada episódio contam como \"não detectadas\". "
        "Para o tutor, o que importa é o episódio ser detectado (tabela do início).",
        "",
        "`classification_report` do motor:",
        "",
        "```text",
        resultados[-1]["porLeitura"]["relatorio"].rstrip(),
        "```",
    ]
    (pasta_saida / "resultados.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def executar(pasta_dados: Path = PASTA_DADOS, pasta_saida: Path = PASTA, titulo: str = "Resultados da avaliação"):
    pasta_saida.mkdir(parents=True, exist_ok=True)
    print(f"Simulando leituras de {pasta_dados}...")
    avaliacoes, episodios = simular(pasta_dados)
    pets_dias = 4 * 7
    resultados = [metricas(av, episodios, pets_dias) for av in avaliacoes]
    (pasta_saida / "resultados.json").write_text(
        json.dumps({"resultados": resultados, "alertas": {
            av.nome: [(p, d.isoformat(), n, sorted(s)) for p, d, n, s in av.alertas] for av in avaliacoes
        }}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    escrever_relatorio(resultados, episodios, pets_dias, pasta_saida, titulo)
    for r in resultados:
        m = r["porLeitura"]
        print(f"{r['metodo']:50} alertas={r['alertasEmitidos']:4} precisão={r['precisao']:.2f} "
              f"recall={r['recall']:.2f} f1={r['f1']:.2f} latência={r['latenciaMedianaH']} | "
              f"por leitura: P={m['precisao']:.2f} R={m['recall']:.2f} F1={m['f1']:.2f}")
    return resultados


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o motor contra as regras fixas.")
    parser.add_argument("--dados", type=Path, default=PASTA_DADOS)
    parser.add_argument("--saida", type=Path, default=PASTA)
    parser.add_argument("--titulo", default="Resultados da avaliação")
    args = parser.parse_args()
    executar(args.dados, args.saida, args.titulo)


if __name__ == "__main__":
    main()
