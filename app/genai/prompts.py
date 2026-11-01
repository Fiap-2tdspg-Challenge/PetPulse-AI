"""Engenharia de prompt do alerta contextual (persona, tarefa, contexto, formato,
restrições e um exemplo few-shot, como no Lab 1)."""

import json

SYSTEM_PROMPT = "\n".join([
    "Você é o redator de alertas de saúde do PetPulse, um sistema que monitora pets por uma coleira IoT.",
    "Tarefa: transformar o resultado técnico do motor de análise em um alerta curto e claro para o tutor.",
    "Regras obrigatórias:",
    "- Escreva em português do Brasil, para um tutor leigo, com tom calmo e direto.",
    "- Use SOMENTE os fatos do JSON recebido. Não invente valores, datas, sintomas, causas ou duração.",
    "- Todo número escrito no texto precisa existir nos fatos.",
    "- Não faça diagnóstico e não cite nomes de doenças: descreva o que mudou em relação ao padrão do próprio pet.",
    "- Nunca indique medicamentos, doses ou tratamentos caseiros.",
    "- O nível de risco já foi decidido pelo motor. Não o altere; a recomendação deve seguir a conduta_obrigatoria.",
    "- Se houver contexto clínico (vacina, medicação, exame), mencione-o de forma neutra quando for relevante.",
    "Formato: JSON com titulo (até 80 caracteres), mensagem (até 2 frases, cita o nome do pet,",
    "o sinal, a variação em relação ao padrão dele e há quanto tempo) e recomendacao (até 2 frases).",
])

EXEMPLO_ENTRADA = {
    "pet": {"nome": "Luna", "especie": "cão", "raca": "Beagle", "idade_anos": 5, "artigo": "da"},
    "nivel_risco": "MEDIO",
    "conduta_obrigatoria": "Recomendar observar o pet e agendar consulta de rotina com o veterinário.",
    "desvios": [{
        "sinal": "frequência cardíaca", "valor_atual": "112 bpm", "padrao_do_pet": "88 bpm",
        "variacao": "27%", "contexto": "em repouso", "duracao_horas": 5,
    }],
    "contexto_clinico": [],
}

EXEMPLO_SAIDA = {
    "titulo": "Frequência cardíaca da Luna acima do padrão",
    "mensagem": "A frequência cardíaca da Luna está 27% acima do padrão dela em repouso há cerca de "
                "5 horas (112 bpm, quando o habitual é 88 bpm).",
    "recomendacao": "Observe se ela está ofegante, inquieta ou sem apetite e agende uma consulta de "
                    "rotina com o veterinário.",
}


def montar_entrada(fatos: dict) -> str:
    return (
        "Exemplo de fatos:\n"
        f"{json.dumps(EXEMPLO_ENTRADA, ensure_ascii=False)}\n"
        "Exemplo de resposta:\n"
        f"{json.dumps(EXEMPLO_SAIDA, ensure_ascii=False)}\n\n"
        "Agora redija o alerta para estes fatos:\n"
        f"{json.dumps(fatos, ensure_ascii=False)}"
    )
