"""Simulador de coleira para a demonstração ponta a ponta (cenário real de uso).

Faz o papel da coleira/gateway enviando leituras para a API Java (POST /iot-readings):
1. Histórico normal dos últimos N dias (leituras retroativas: entram na baseline, sem alertas).
2. As últimas 6 horas com o cenário escolhido. As leituras da última hora disparam a
   análise do PetPulse-AI e, se houver anomalia, o SmartAlert aparece no app mobile.

Exemplo:
    python scripts/simular_coleira.py --api http://localhost:8080 --device-id 1 --cenario taquicardia

Credenciais: variáveis PETPULSE_EMAIL e PETPULSE_SENHA (padrão: tutor de demonstração do README da API).
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import httpx
import numpy as np

CENARIOS = ("normal", "taquicardia", "febre", "hipertensao", "prostracao")


def gerar(inicio: datetime, fim: datetime, passo_min: int, cenario: str | None, rng, perfil: dict) -> list[dict]:
    leituras = []
    ruido = 0.0
    t = inicio
    while t <= fim:
        noite = t.hour >= 22 or t.hour < 6
        passeio = t.hour in (7, 18) and t.minute < 40
        atividade = rng.uniform(40, 58) if passeio else rng.uniform(0, 8) if noite else rng.uniform(4, 24)
        ruido = 0.7 * ruido + rng.normal(0, 3)
        fc = perfil["fc"] - (12 if noite else 0) + 0.9 * atividade + ruido
        temperatura = perfil["temp"] + 0.004 * atividade + rng.normal(0, 0.04)
        pressao = perfil["pressao"] + 0.25 * atividade + rng.normal(0, 3)

        if cenario == "taquicardia":
            atividade = rng.uniform(3, 12)
            fc = (perfil["fc"] + 0.9 * atividade) * 1.32 + rng.normal(0, 3)
        elif cenario == "febre":
            atividade = rng.uniform(1, 8)
            temperatura = perfil["temp"] + 1.4 + rng.normal(0, 0.05)
            fc = perfil["fc"] * 1.15 + rng.normal(0, 3)
        elif cenario == "hipertensao":
            pressao = perfil["pressao"] + 40 + rng.normal(0, 3)
        elif cenario == "prostracao":
            atividade = rng.uniform(0, 4)

        leituras.append({
            "readingDate": t.replace(microsecond=0).isoformat(),
            "heartRate": int(round(fc)),
            "activityLevel": round(float(atividade), 1),
            "pressure": int(round(pressao)),
            "temperature": round(float(temperatura), 2),
        })
        t += timedelta(minutes=passo_min)
    return leituras


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default="http://localhost:8080")
    parser.add_argument("--device-id", type=int, required=True)
    parser.add_argument("--cenario", choices=CENARIOS, default="taquicardia")
    parser.add_argument("--dias", type=float, default=7, help="dias de histórico normal")
    parser.add_argument("--passo-min", type=int, default=15)
    parser.add_argument("--fc", type=float, default=80, help="FC de repouso do pet simulado")
    parser.add_argument("--fuso", choices=("local", "utc"), default=None,
                        help="relógio do servidor da API (padrão: utc para Render, local para localhost)")
    parser.add_argument("--sem-historico", action="store_true", help="envia só as últimas 6 horas")
    args = parser.parse_args()

    fuso = args.fuso or ("utc" if "onrender.com" in args.api else "local")
    agora = (datetime.now(timezone.utc).replace(tzinfo=None) if fuso == "utc" else datetime.now()) - timedelta(minutes=1)
    rng = np.random.default_rng(42)
    perfil = {"fc": args.fc, "temp": 38.5, "pressao": 132}

    email = os.getenv("PETPULSE_EMAIL", "security.tutor@petpulse.com")
    senha = os.getenv("PETPULSE_SENHA", "123456")

    with httpx.Client(base_url=args.api, timeout=60) as http:
        resposta = http.post("/login", json={"email": email, "password": senha})
        resposta.raise_for_status()
        http.headers["Authorization"] = f"Bearer {resposta.json()['token']}"

        inicio_cenario = agora - timedelta(hours=6)
        historico = [] if args.sem_historico else gerar(
            agora - timedelta(days=args.dias), inicio_cenario - timedelta(minutes=args.passo_min),
            args.passo_min, None, rng, perfil,
        )
        recentes = gerar(inicio_cenario, agora, 10, None if args.cenario == "normal" else args.cenario, rng, perfil)

        def enviar(leitura: dict) -> int:
            r = http.post("/iot-readings", json={"deviceId": args.device_id, **leitura})
            if r.status_code >= 400:
                print(f"  erro {r.status_code}: {r.text[:200]}", file=sys.stderr)
            return r.status_code

        if historico:
            print(f"Enviando {len(historico)} leituras retroativas ({args.dias:g} dias)...")
            with ThreadPoolExecutor(max_workers=6) as executor:
                codigos = list(executor.map(enviar, historico))
            print(f"  {sum(c < 300 for c in codigos)} aceitas")

        print(f"Enviando as últimas 6 h com o cenário '{args.cenario}' ({len(recentes)} leituras)...")
        for leitura in recentes:
            enviar(leitura)

        alertas = http.get("/smart-alerts", params={"size": 200}).json()["content"]
        da_ia = [a for a in alertas if a["origin"] == "MOTOR_IA" and a["status"] == "ABERTO"]
        print(f"\nAlertas abertos gerados pela IA: {len(da_ia)}")
        for a in da_ia[-3:]:
            print(f"  [{a['riskLevel']}] {a['petName']} · {a['alertTypeDescription']}")
            print(f"    {a['message']}")
            print(f"    → {a['recommendation']}")


if __name__ == "__main__":
    main()
