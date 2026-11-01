# Resultados no dataset de teste (semente 2026, nunca usado no desenvolvimento)

Simulação de 28 pet-dias (4 pets × 7 dias, uma leitura a cada 10 min). Todos os métodos passam pela mesma deduplicação de alertas (6 h por pet, salvo escalada). Um alerta só conta como verdadeiro se ocorrer durante o episódio (ou até 2 h depois) **e** for disparado pelo mesmo sinal vital, na mesma direção (ex.: FC acima do normal). Gerado por `python -m avaliacao.avaliar`.

| Método | Alertas enviados | Precisão | Recall (episódios) | F1 | Falsos alarmes / pet-dia | Latência mediana | Leituras sinalizadas* |
|---|---:|---:|---:|---:|---:|---:|---:|
| Regras atuais da API Java | 142 | 12% | 3/5 (60%) | 0.20 | 4.46 | 4.8 h | 3503 |
| Faixas fixas do firmware | 92 | 11% | 2/5 (40%) | 0.17 | 2.93 | 15.6 h | 1711 |
| Baseline individual (sem Isolation Forest) | 26 | 85% | 5/5 (100%) | 0.92 | 0.14 | 2.8 h | 742 |
| Motor PetPulse-AI (baseline + Isolation Forest) | 26 | 85% | 5/5 (100%) | 0.92 | 0.14 | 2.8 h | 746 |

\* Leituras sinalizadas = leituras classificadas como MÉDIO ou ALTO antes da deduplicação. Para as regras atuais da API, é o número de alertas que o `IoTReadingAnalysisService` grava hoje, porque ele cria um alerta por leitura fora da faixa.

## Episódio a episódio

| Episódio | Pet | Deve alertar? | Esperado | Regras atuais da API Java | Faixas fixas do firmware | Baseline individual (sem Isolation Forest) | Motor PetPulse-AI (baseline + Isolation Forest) |
|---|---|---|---|---|---|---|---|
| E1: Taquicardia em repouso: FC 30% acima do padrão do Thor (~107 bpm, abaixo do limite fixo de 180) | Thor | sim | MEDIO | — | — | MEDIO (alerta em 2.8 h) | MEDIO (alerta em 2.8 h) |
| E2: Artefato de sensor: uma leitura isolada de 205 bpm | Thor | não | — | ALTO ⚠ alertou | ALTO ⚠ alertou | BAIXO | BAIXO |
| E3: Reação vacinal esperada: +0,5 °C e -35% de atividade | Mel | não | BAIXO | MEDIO ⚠ alertou | MEDIO ⚠ alertou | BAIXO | BAIXO |
| E4: Febre em repouso com prostração (39,9 °C, -60% de atividade) | Mel | sim | ALTO | MEDIO (alerta em 2.0 h) | ALTO (alerta em 3.2 h) | ALTO (alerta em 2.8 h) | ALTO (alerta em 2.8 h) |
| E5: Hipertensão progressiva: pressão sobe até +45 mmHg | Nina | sim | MEDIO | ALTO (alerta em 13.0 h) | ALTO (alerta em 28.0 h) | ALTO (alerta em 13.8 h) | ALTO (alerta em 13.8 h) |
| E6: Prostração: queda de 55% na movimentação | Nina | sim | MEDIO | MEDIO (alerta em 4.8 h) | — | MEDIO (alerta em 16.3 h) | MEDIO (alerta em 16.3 h) |
| E7: Taquicardia em repouso: FC 38% acima do padrão do Bob (~92 bpm, dentro da faixa populacional) | Bob | sim | MEDIO | — | — | MEDIO (alerta em 2.8 h) | MEDIO (alerta em 2.8 h) |

## Métricas por leitura (scikit-learn)

Cada leitura é uma amostra: classe real 1 se está dentro de um episódio que deve alertar, e previsão 1 se o método a classificou como MÉDIO ou ALTO. São as mesmas funções dos labs (`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`).

| Método | Acurácia | Precisão | Recall | F1 | Matriz de confusão [[VN, FP], [FN, VP]] |
|---|---:|---:|---:|---:|---|
| Regras atuais da API Java | 34% | 24% | 100% | 0.39 | `[[529, 2652], [0, 851]]` |
| Faixas fixas do firmware | 73% | 43% | 86% | 0.57 | `[[2206, 975], [115, 736]]` |
| Baseline individual (sem Isolation Forest) | 91% | 82% | 72% | 0.77 | `[[3051, 130], [239, 612]]` |
| Motor PetPulse-AI (baseline + Isolation Forest) | 91% | 83% | 72% | 0.77 | `[[3051, 130], [235, 616]]` |

O recall por leitura do motor é menor que o por episódio porque ele espera algumas horas de persistência antes de alertar: as primeiras leituras de cada episódio contam como "não detectadas". Para o tutor, o que importa é o episódio ser detectado (tabela do início).

`classification_report` do motor:

```text
              precision    recall  f1-score   support

      normal       0.93      0.96      0.94      3181
    anomalia       0.83      0.72      0.77       851

    accuracy                           0.91      4032
   macro avg       0.88      0.84      0.86      4032
weighted avg       0.91      0.91      0.91      4032
```
