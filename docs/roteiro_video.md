# Roteiro do vídeo pitch (~5 min)

Apresentação feita por um integrante, em nome do grupo. O vídeo vai para o YouTube como **não listado**. Cada bloco indica o que aparece na tela.

| Tempo | Tela | Fala (resumo) |
|---|---|---|
| 0:00–0:30 | Logo do PetPulse + foto de um pet | **Problema.** Pets escondem dor e doença até o quadro avançar. A coleira gera dados 24 h, mas o tutor não sabe interpretar gráficos. E regras fixas como "FC acima de 180" não servem para todos: um Chihuahua idoso e um Border Collie atleta têm "normais" completamente diferentes. |
| 0:30–1:15 | Diagrama de arquitetura do README | **Solução e integração.** Coleira ESP32 → API Java (Oracle) → PetPulse-AI (FastAPI) → alerta gravado no Oracle → app mobile notifica. A IA que definimos na Sprint 3 tem três pilares: baseline individual, detecção de anomalias com Isolation Forest e alerta contextual escrito pelo Gemini. Se a IA cair, a API volta para as regras antigas, e o tutor nunca fica sem proteção. |
| 1:15–2:15 | VS Code (`app/ml/baseline.py` e `motor.py`) ou notebook (seções 6 a 9) | **Como a IA funciona.** A baseline aprende o normal *daquele* pet em cada contexto (repouso de dia, sono, caminhada, corrida) e corrige pela atividade do momento. O Isolation Forest, treinado só com o histórico do pet, aponta combinações incomuns. A matriz de severidade da Sprint 3 decide BAIXO, MÉDIO ou ALTO, e o histórico clínico entra no contexto: febre leve depois de vacina é rebaixada. O Gemini só escreve o texto; se inventar número ou citar remédio, o texto é descartado (guardrails). |
| 2:15–3:30 | Terminal com `scripts/simular_coleira.py --cenario taquicardia` → app no emulador | **Demonstração.** O simulador faz o papel da coleira: manda 7 dias de histórico normal e 6 horas de taquicardia em repouso. A API chama a IA, e o alerta aparece no sino do app com o selo "Análise da IA": *"A frequência cardíaca do Thor está 32% acima do padrão dele em repouso há 6 horas…"*. Depois, rodar com `--cenario febre --sem-historico` para mostrar o alerta ALTO virando notificação no celular. |
| 3:30–4:20 | `avaliacao/comparativo.png` e `linha_do_tempo_thor.png` | **Resultados.** Simulamos 28 pet-dias com 7 episódios marcados. As regras atuais teriam 13% de precisão, ~4 alarmes falsos por pet por dia e detectariam 3 dos 5 episódios. O PetPulse-AI tem 92% de precisão, 0,07 alarme falso por pet por dia e detecta os 5. Num segundo dataset, que não foi usado no desenvolvimento, mantém 85% de precisão e os 5 episódios (a mesma ideia do `train_test_split` dos labs). Destacar que as regras fixas não veem a taquicardia do Thor (107 bpm < 180) e disparam no erro de sensor de 205 bpm e na reação à vacina; a IA acerta os dois casos. |
| 4:20–5:00 | Tela final com os benefícios | **Valor.** Para o tutor: poucos alertas, em linguagem simples e feitos para o pet dele. Para a clínica: o alerta já chega com o contexto (quanto mudou, há quanto tempo, vacina recente), os pacientes chegam mais cedo e há menos emergências. Próximos passos: validar com dados reais de coleiras e com veterinários. |

**Sem Oracle ou API Java no ar?** A demonstração dos minutos 1:15–3:30 também pode ser feita só com a chave do Gemini, pelo notebook [`PetPulse_IA_Sprint4.ipynb`](../PetPulse_IA_Sprint4.ipynb). Os blocos usados são:
- seção 8: tabela dos episódios;
- seção 9: texto do Gemini e guardrails;
- seção 10: as 4 simulações pela API e o histórico salvo no banco.

O notebook já está salvo com as saídas, então dá para gravar rolando pelas células.

## Checklist antes de gravar

- [ ] API Java e PetPulse-AI no ar (abra `/actuator/health` e `/health` antes).
- [ ] `GEMINI_API_KEY` no `.env` do PetPulse-AI, para os textos virem do Gemini. Ele leva ~10 s por alerta. Sem a chave, o texto padrão é usado e a demonstração funciona igual.
- [ ] Pet de demonstração com dispositivo IoT cadastrado e o `deviceId` anotado.
- [ ] Simulador rodado uma vez antes, para conferir que o alerta aparece. Depois, apagar os alertas de teste com `DELETE /smart-alerts/{id}` no Swagger (a anti-duplicação bloqueia alertas do mesmo nível por 6 h).
- [ ] App mobile logado com o tutor dono do pet e permissão de notificação concedida.
- [ ] Gravação de tela em 1080p, com o terminal em fonte grande.
