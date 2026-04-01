"""
Etapa 03 - Treino e previsão de mercados de aposta.

Estratégia:
- Treina 2 modelos PoissonRegressor:
  1) gols mandante
  2) gols visitante
- Converte lambdas previstos em probabilidades de scoreline (0..max_goals)
- Calcula probabilidades dos mercados solicitados.

Saídas:
- data/output/predictions_markets.csv
- data/output/predictions_markets.xlsx
- data/output/model_metrics.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path("data")
PROCESSED_DIR = BASE_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / "model_dataset.parquet"


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT_FILE}. Rode primeiro a etapa 02_build_features.py"
        )
    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def build_feature_list(df: pd.DataFrame) -> Tuple[list[str], list[str], list[str]]:
    base_num = [c for c in df.columns if ("avg_" in c or c.startswith("diff_"))]
    cat_cols = [c for c in ["league", "season", "home_team", "away_team"] if c in df.columns]

    must_have = ["home_goals", "away_goals", "date"]
    missing = [c for c in must_have if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset sem colunas necessárias: {missing}")

    feature_cols = base_num + cat_cols
    if not feature_cols:
        raise ValueError("Nenhuma feature foi selecionada. Verifique etapa 02.")

    return feature_cols, base_num, cat_cols


def train_test_split_time(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def build_model(num_cols: list[str], cat_cols: list[str]) -> Pipeline:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("reg", PoissonRegressor(alpha=0.2, max_iter=500)),
        ]
    )
    return model


def fit_models(train_df: pd.DataFrame, num_cols: list[str], cat_cols: list[str], features: list[str]):
    x_train = train_df[features]

    m_home = build_model(num_cols, cat_cols)
    m_away = build_model(num_cols, cat_cols)

    m_home.fit(x_train, train_df["home_goals"])
    m_away.fit(x_train, train_df["away_goals"])
    return m_home, m_away


def poisson_probs(lam: np.ndarray, max_goals: int = 8) -> np.ndarray:
    goals = np.arange(0, max_goals + 1)
    fact = np.array([np.math.factorial(int(g)) for g in goals])
    probs = np.exp(-lam[:, None]) * (lam[:, None] ** goals[None, :]) / fact[None, :]

    tail = 1 - probs.sum(axis=1, keepdims=True)
    probs[:, -1:] = probs[:, -1:] + np.clip(tail, 0, None)
    return probs


def market_probabilities(home_lambda: np.ndarray, away_lambda: np.ndarray, max_goals: int = 8) -> Dict[str, np.ndarray]:
    p_home = poisson_probs(home_lambda, max_goals=max_goals)
    p_away = poisson_probs(away_lambda, max_goals=max_goals)

    score_matrix = p_home[:, :, None] * p_away[:, None, :]

    idx = np.arange(max_goals + 1)
    home_win = (idx[:, None] > idx[None, :]).astype(float)
    draw = (idx[:, None] == idx[None, :]).astype(float)
    away_win = (idx[:, None] < idx[None, :]).astype(float)

    total_goals = idx[:, None] + idx[None, :]

    def mat_prob(mask: np.ndarray) -> np.ndarray:
        return (score_matrix * mask[None, :, :]).sum(axis=(1, 2))

    p_home_win = mat_prob(home_win)
    p_draw = mat_prob(draw)
    p_away_win = mat_prob(away_win)

    p_1x = p_home_win + p_draw
    p_x2 = p_away_win + p_draw
    p_12 = p_home_win + p_away_win

    p_over_15 = mat_prob((total_goals > 1.5).astype(float))
    p_over_25 = mat_prob((total_goals > 2.5).astype(float))
    p_over_35 = mat_prob((total_goals > 3.5).astype(float))

    p_home_over_05 = 1 - p_home[:, 0]
    p_home_over_15 = 1 - (p_home[:, 0] + p_home[:, 1])
    p_home_over_25 = 1 - (p_home[:, 0] + p_home[:, 1] + p_home[:, 2])
    p_home_over_35 = 1 - (p_home[:, 0] + p_home[:, 1] + p_home[:, 2] + p_home[:, 3])

    p_away_over_05 = 1 - p_away[:, 0]
    p_away_over_15 = 1 - (p_away[:, 0] + p_away[:, 1])
    p_away_over_25 = 1 - (p_away[:, 0] + p_away[:, 1] + p_away[:, 2])
    p_away_over_35 = 1 - (p_away[:, 0] + p_away[:, 1] + p_away[:, 2] + p_away[:, 3])

    # DNB (empate anula): probabilidade condicional de vitória dado não empate.
    eps = 1e-12
    p_home_dnb = p_home_win / np.clip((1 - p_draw), eps, None)
    p_away_dnb = p_away_win / np.clip((1 - p_draw), eps, None)

    return {
        "p_home_win": p_home_win,
        "p_draw": p_draw,
        "p_away_win": p_away_win,
        "p_home_dnb": p_home_dnb,
        "p_away_dnb": p_away_dnb,
        "p_double_chance_1x": p_1x,
        "p_double_chance_x2": p_x2,
        "p_double_chance_12": p_12,
        "p_over_1_5": p_over_15,
        "p_over_2_5": p_over_25,
        "p_over_3_5": p_over_35,
        "p_home_over_0_5": p_home_over_05,
        "p_home_over_1_5": p_home_over_15,
        "p_home_over_2_5": p_home_over_25,
        "p_home_over_3_5": p_home_over_35,
        "p_away_over_0_5": p_away_over_05,
        "p_away_over_1_5": p_away_over_15,
        "p_away_over_2_5": p_away_over_25,
        "p_away_over_3_5": p_away_over_35,
    }


def run_pipeline() -> None:
    df = load_data()
    features, num_cols, cat_cols = build_feature_list(df)
    train_df, test_df = train_test_split_time(df, test_size=0.2)

    home_model, away_model = fit_models(train_df, num_cols, cat_cols, features)

    x_test = test_df[features]
    pred_home_lambda = home_model.predict(x_test)
    pred_away_lambda = away_model.predict(x_test)

    markets = market_probabilities(pred_home_lambda, pred_away_lambda, max_goals=8)

    out = test_df[["date", "league", "season", "home_team", "away_team", "home_goals", "away_goals"]].copy()
    out["pred_home_lambda"] = pred_home_lambda
    out["pred_away_lambda"] = pred_away_lambda

    for k, v in markets.items():
        out[k] = v

    out_path_csv = OUTPUT_DIR / "predictions_markets.csv"
    out_path_xlsx = OUTPUT_DIR / "predictions_markets.xlsx"
    out.to_csv(out_path_csv, index=False)
    out.to_excel(out_path_xlsx, index=False)

    metrics = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "mae_home_goals": float(mean_absolute_error(test_df["home_goals"], pred_home_lambda)),
        "mae_away_goals": float(mean_absolute_error(test_df["away_goals"], pred_away_lambda)),
        "features_used": features,
    }

    with open(OUTPUT_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("=== Treino e previsão concluídos ===")
    print(f"Predições: {out_path_csv}")
    print(f"Predições Excel: {out_path_xlsx}")
    print(f"Métricas: {OUTPUT_DIR / 'model_metrics.json'}")


if __name__ == "__main__":
    run_pipeline()
