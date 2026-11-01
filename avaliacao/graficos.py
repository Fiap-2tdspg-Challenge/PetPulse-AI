"""Gera os gráficos do README a partir de avaliacao/resultados.json.

Uso:  python -m avaliacao.graficos   (rodar depois de python -m avaliacao.avaliar)
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from avaliacao.dataset import carregar_episodios, carregar_leituras  # noqa: E402

PASTA = Path(__file__).parent
SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRADE = "#e4e3df"
DESTAQUE = "#2a78d6"     # categórica 1: PetPulse-AI
COMPARACAO = "#eb6834"   # categórica 2: regras atuais
NEUTRO = "#a3a29c"
MOTOR = "Motor PetPulse-AI (baseline + Isolation Forest)"
API = "Regras atuais da API Java"
ROTULOS = {
    API: "Regras atuais da API Java",
    "Faixas fixas do firmware": "Faixas fixas do firmware",
    "Baseline individual (sem Isolation Forest)": "Baseline individual (sem IF)",
    MOTOR: "PetPulse-AI (baseline + IF)",
}


def _eixo_limpo(ax) -> None:
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(GRADE)
    ax.tick_params(colors=TEXTO_SECUNDARIO, length=0)
    ax.grid(axis="x", color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def comparativo(resultados: list[dict]) -> None:
    metricas = [
        ("precisao", "Precisão dos alertas", lambda v: f"{v * 100:.0f}%", 1.0),
        ("falsosAlarmesPorPetDia", "Alarmes falsos por pet por dia", lambda v: f"{v:.2f}", None),
    ]
    fig, eixos = plt.subplots(1, 2, figsize=(11, 3.4), facecolor=SUPERFICIE)
    nomes = [ROTULOS[r["metodo"]] for r in resultados][::-1]
    for ax, (chave, titulo, fmt, limite) in zip(eixos, metricas):
        valores = [r[chave] for r in resultados][::-1]
        cores = [DESTAQUE if r["metodo"] == MOTOR else NEUTRO for r in resultados][::-1]
        barras = ax.barh(nomes, valores, color=cores, height=0.55)
        maximo = limite or max(valores) * 1.15
        for barra, valor in zip(barras, valores):
            ax.text(barra.get_width() + maximo * 0.015, barra.get_y() + barra.get_height() / 2,
                    fmt(valor), va="center", fontsize=10, color=TEXTO)
        ax.set_xlim(0, maximo * 1.12)
        ax.set_title(titulo, loc="left", fontsize=12, color=TEXTO, fontweight="bold")
        _eixo_limpo(ax)
        ax.tick_params(axis="y", labelsize=10, labelcolor=TEXTO)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _, fmt=fmt: fmt(v)))
    eixos[1].set_yticklabels([])
    fig.tight_layout()
    fig.savefig(PASTA / "comparativo.png", dpi=150, facecolor=SUPERFICIE)
    plt.close(fig)


def linha_do_tempo_thor(alertas: dict) -> None:
    inicio, fim = datetime(2026, 10, 10, 6), datetime(2026, 10, 11, 6)
    leituras = [l for l, _ in carregar_leituras()[1] if inicio <= l.data_leitura <= fim]
    datas = [l.data_leitura for l in leituras]
    fc = [l.frequencia_cardiaca for l in leituras]
    e1 = next(e for e in carregar_episodios() if e["id"] == "E1")

    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=SUPERFICIE)
    ax.axvspan(e1["inicio"], e1["fim"], color="#efeeea", label="Episódio E1: taquicardia em repouso")
    ax.plot(datas, fc, color=TEXTO_SECUNDARIO, linewidth=1.2, label="FC do Thor (bpm)")

    for metodo, cor, marcador, y in ((API, COMPARACAO, "v", 195), (MOTOR, DESTAQUE, "o", 182)):
        momentos = [
            datetime.fromisoformat(d) for pet, d, *_ in alertas[metodo]
            if pet == 1 and inicio <= datetime.fromisoformat(d) <= fim
        ]
        ax.scatter(momentos, [y] * len(momentos), color=cor, marker=marcador, s=70, zorder=3,
                   edgecolor=SUPERFICIE, linewidth=1.5, label=f"Alerta enviado: {ROTULOS[metodo]}")

    ax.set_ylim(40, 205)
    ax.set_ylabel("bpm", color=TEXTO_SECUNDARIO)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
    ax.set_title("Thor, 10–11/10: o que cada método envia ao tutor", loc="left", fontsize=12,
                 color=TEXTO, fontweight="bold")
    _eixo_limpo(ax)
    ax.grid(axis="y", color=GRADE, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, frameon=False, fontsize=9,
              labelcolor=TEXTO)
    fig.tight_layout()
    fig.savefig(PASTA / "linha_do_tempo_thor.png", dpi=150, facecolor=SUPERFICIE)
    plt.close(fig)


def main() -> None:
    dados = json.loads((PASTA / "resultados.json").read_text(encoding="utf-8"))
    comparativo(dados["resultados"])
    linha_do_tempo_thor(dados["alertas"])
    print("Gráficos gerados em", PASTA)


if __name__ == "__main__":
    main()
