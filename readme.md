# Clusterização de Níveis de Obesidade

Trabalho prático da disciplina de Sistemas Inteligentes Avançados.

## Problema

Dado um dataset com hábitos alimentares e condição física de pacientes, o objetivo é agrupar os pacientes em clusters sem uso de rótulos (aprendizado não supervisionado), identificar o perfil de cada grupo e inferir a qual cluster um novo paciente pertence.

**Dataset:** [Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

## Arquivos

| Arquivo | Descrição |
|---|---|
| `obesity_training.py` | Pré-processa os dados, determina o K ótimo pelo método do cotovelo e treina o modelo |
| `obesity_centroids.py` | Descreve o perfil de cada cluster com base nos centroides |
| `obesity_inference.py` | Recebe os dados de um novo paciente e informa a qual cluster ele pertence |

## Como executar

```bash
chmod +x run.sh
./run.sh
```

Os três scripts devem ser executados na ordem do `run.sh`. O treinamento gera os arquivos `.pkl` usados pelos demais módulos.