# Pipeline Understat (soccerdata) para previsão de mercados de futebol

Este repositório agora está organizado em **3 etapas independentes**, pensado para rodar no **Google Colab** e facilitar debug quando algo quebra.

## Estrutura

- `colab_pipeline/01_extract_understat.py`  
  Extrai dados do Understat via `soccerdata` e salva em CSV/Parquet/Excel.
- `colab_pipeline/02_build_features.py`  
  Cria variáveis pré-jogo (forma, rolling média de gols/xG etc.) e salva dataset final para modelagem.
- `colab_pipeline/03_train_predict_markets.py`  
  Treina modelo Poisson para gols mandante/visitante e transforma em probabilidades de mercados de aposta.

## Como rodar no Google Colab

### 0) Instalação de pacotes

```python
!pip install -q soccerdata pandas numpy scikit-learn openpyxl pyarrow
```

### 1) Extração (rodar antes de tudo)

```python
!python colab_pipeline/01_extract_understat.py
```

Saídas:
- `data/raw/understat_matches_raw.csv`
- `data/raw/understat_matches_raw.parquet`
- `data/raw/understat_matches_raw.xlsx`
- `data/raw/understat_summary_by_league_season.xlsx`

### 2) Engenharia de atributos

```python
!python colab_pipeline/02_build_features.py
```

Saídas:
- `data/processed/model_dataset.csv`
- `data/processed/model_dataset.parquet`
- `data/processed/model_dataset.xlsx`

### 3) Treino + previsões dos mercados

```python
!python colab_pipeline/03_train_predict_markets.py
```

Saídas:
- `data/output/predictions_markets.csv`
- `data/output/predictions_markets.xlsx`
- `data/output/model_metrics.json`

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
- Se quiser, o próximo passo pode ser adicionar calibração probabilística e backtest com ROI por mercado.
