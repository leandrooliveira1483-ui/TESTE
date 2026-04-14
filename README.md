# Pipeline Understat (soccerdata) para previsão de mercados de futebol

Este repositório está organizado em **3 etapas independentes**, pensado para rodar no **Google Colab** e facilitar debug quando algo quebra.

## Estrutura

- `colab_pipeline/01_extract_understat.py`  
  Extrai dados do Understat via `soccerdata` e salva em CSV/Parquet/Excel.
- `colab_pipeline/02_build_features.py`  
  Cria variáveis pré-jogo (forma, rolling média de gols/xG etc.) e salva dataset final para modelagem.
- `colab_pipeline/03_train_predict_markets.py`  
  Treina modelo Poisson para gols mandante/visitante e transforma em probabilidades de mercados de aposta.

## Como rodar no Google Colab

### 0) Instalação de pacotes (versão estável para evitar conflitos do Colab)

> Se você viu mensagens como conflito de `requests`, `rich`, `urllib3` e `jedi`, use **estes comandos** (na ordem) e reinicie o runtime ao final.

```python
# 1) Atualiza o pip
!python -m pip install -q --upgrade pip

# 2) Instala o pipeline
!python -m pip install -q \
  pandas==2.2.3 \
  numpy==2.0.2 \
  scikit-learn==1.5.2 \
  openpyxl==3.1.5 \
  pyarrow==18.1.0 \
  soccerdata==1.8.7

# 3) Reforça compatibilidade com o ecossistema do Colab
!python -m pip install -q --upgrade \
  requests==2.32.4 \
  "urllib3>=2,<3" \
  "rich>=12.4.4,<14" \
  "jedi>=0.16"
```

Depois execute:

```python
import os
os.kill(os.getpid(), 9)  # reinicia runtime para carregar dependências corretas
```

### 1) Extração (rodar antes de tudo)

> A etapa 01 baixa **jogos finalizados e jogos futuros** (sem placar), para permitir previsão real dos próximos jogos.

> Se houver timeout/TLS do Understat em alguma liga/temporada, a etapa 01 agora faz retries e segue com as demais.
> Falhas parciais são registradas em `data/raw/understat_download_failures.txt`.


```python
!python colab_pipeline/01_extract_understat.py
```

Saídas:
- `data/raw/understat_matches_raw.csv`
- `data/raw/understat_matches_raw.parquet`
- `data/raw/understat_matches_raw.xlsx`
- `data/raw/understat_summary_by_league_season.xlsx`
- `data/raw/understat_schema_report.json` (validação de schema e missing)

### 2) Engenharia de atributos

```python
!python colab_pipeline/02_build_features.py
```

Saídas:
- `data/processed/model_dataset.csv`
- `data/processed/model_dataset.parquet`
- `data/processed/model_dataset.xlsx`
- `data/processed/data_quality_report.json`

### 3) Treino + previsões dos mercados

```python
!python colab_pipeline/03_train_predict_markets.py --calibration-method isotonic
```

Saídas:
- `data/output/predictions_markets_future.csv` (previsão de jogos que ainda vão acontecer)
- `data/output/predictions_markets_future.xlsx`
- `data/output/predictions_markets_backtest.csv` (avaliação em jogos já encerrados)
- `data/output/predictions_markets_backtest.xlsx`
- `data/output/predictions_markets_window_all.csv` (janela completa simulada: jogos encerrados + futuros)
- `data/output/predictions_markets_window_all.xlsx`
- `data/output/predictions_markets.csv` (atalho compatível com arquivo de futuros)
- `data/output/model_metrics.json` (inclui Brier, LogLoss, calibração, drift e relatório de odds)


### 3.1) Simulação por data-limite de treino (para testar acurácia por rodadas)

Use esse modo para simular cenário real: treina até uma data e avalia/prediz apenas após ela.

```python
!python colab_pipeline/03_train_predict_markets.py --calibration-method isotonic \
  --train-end-date 2026-04-10 \
  --predict-start-date 2026-04-11 \
  --predict-end-date 2026-04-20
```

- `--train-end-date`: última data permitida para treino.
- `--predict-start-date` e `--predict-end-date`: janela de análise/predição após o treino.
- Jogos da janela com resultado final vão para `predictions_markets_backtest.*` (medir acurácia).
- Jogos da janela sem resultado vão para `predictions_markets_future.*` (rodadas futuras).

> Importante: jogos da janela que **já terminaram** (ex.: 10/04/2026 e 11/04/2026) ficam em `predictions_markets_backtest.*`.
> Para ver tudo junto (encerrados + futuros), use `predictions_markets_window_all.*`.


### 3.2) Backtest com odds históricas (EV/ROI/CLV)

> Observação: o `--odds-file` **não altera as previsões do modelo**. Ele só adiciona o backtest financeiro (EV/ROI/CLV) sobre as probabilidades já previstas.


Crie um CSV (ex.: `data/external/odds.csv`) com colunas mínimas:
- `date`, `league`, `home_team`, `away_team`
- `odds_home`, `odds_draw`, `odds_away`

Opcional para CLV:
- `open_odds_home`, `open_odds_draw`, `open_odds_away`

Execução:
```python
!python colab_pipeline/03_train_predict_markets.py \
  --train-end-date 2026-04-07 \
  --predict-start-date 2026-04-08 \
  --predict-end-date 2026-04-14 \
  --calibration-method isotonic \
  --odds-file data/external/odds.csv
```

## Dados que o pipeline usa

Base do Understat (por jogo):
- Data da partida, liga, temporada
- Mandante e visitante
- Gols mandante e visitante
- xG mandante e visitante
- PPDA mandante e visitante (quando disponível)
- Deep completions mandante e visitante (quando disponível)

Features pré-jogo criadas com janela móvel (default = 5 jogos anteriores):
- média de gols marcados e sofridos
- média de xG a favor e contra
- média de PPDA e Deep
- média de pontos de forma (3/1/0)
- diferenças mandante - visitante nessas métricas

## Mercados previstos

A etapa 03 gera probabilidades para:
- Vitória mandante (1)
- Empate (X)
- Vitória visitante (2)
- Empate anula aposta mandante (DNB casa)
- Empate anula aposta visitante (DNB fora)
- Dupla chance 1X
- Dupla chance X2
- Dupla chance 12
- Over 1.5 gols (jogo)
- Over 2.5 gols (jogo)
- Over 3.5 gols (jogo)
- Over 0.5 gols mandante
- Over 1.5 gols mandante
- Over 2.5 gols mandante
- Over 3.5 gols mandante
- Over 0.5 gols visitante
- Over 1.5 gols visitante
- Over 2.5 gols visitante
- Over 3.5 gols visitante

## Observações importantes

- O modelo usa regressão de Poisson para estimar gols esperados de cada lado e depois calcula probabilidades dos mercados via distribuição de placares.
- As probabilidades não incluem margem da casa; para valor esperado em apostas, compare com odds de mercado e ajuste por overround.
- Próximo passo recomendado: calibração probabilística + backtest com ROI por mercado.
