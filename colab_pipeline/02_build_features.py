"""
Etapa 02 - Engenharia de atributos (features).

Lê os dados brutos extraídos na etapa 01, gera features pré-jogo e salva em:
- data/processed/model_dataset.csv
- data/processed/model_dataset.parquet
- data/processed/model_dataset.xlsx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RAW_DIR / "understat_matches_raw.parquet"


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {INPUT_FILE}. Rode primeiro o script 01_extract_understat.py"
        )
    df = pd.read_parquet(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["league", "season", "date", "home_team", "away_team"]).reset_index(drop=True)
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["total_goals"] = out["home_goals"] + out["away_goals"]
    out["target_home_win"] = (out["home_goals"] > out["away_goals"]).astype(int)
    out["target_draw"] = (out["home_goals"] == out["away_goals"]).astype(int)
    out["target_away_win"] = (out["home_goals"] < out["away_goals"]).astype(int)
    return out


def _team_long_table(df: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_idx": df.index,
            "date": df["date"],
            "league": df["league"],
            "season": df["season"],
            "team": df["home_team"],
            "opponent": df["away_team"],
            "is_home": 1,
            "goals_for": df["home_goals"],
            "goals_against": df["away_goals"],
            "xg_for": df["home_xg"],
            "xg_against": df["away_xg"],
            "ppda": df["home_ppda"],
            "deep": df["home_deep"],
        }
    )

    away = pd.DataFrame(
        {
            "match_idx": df.index,
            "date": df["date"],
            "league": df["league"],
            "season": df["season"],
            "team": df["away_team"],
            "opponent": df["home_team"],
            "is_home": 0,
            "goals_for": df["away_goals"],
            "goals_against": df["home_goals"],
            "xg_for": df["away_xg"],
            "xg_against": df["home_xg"],
            "ppda": df["away_ppda"],
            "deep": df["away_deep"],
        }
    )

    long_df = pd.concat([home, away], ignore_index=True)
    long_df = long_df.sort_values(["team", "league", "date", "match_idx", "is_home"]).reset_index(drop=True)
    return long_df


def add_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    out = df.copy()
    long_df = _team_long_table(out)

    group_cols = ["team", "league"]
    metrics = ["goals_for", "goals_against", "xg_for", "xg_against", "ppda", "deep"]

    for m in metrics:
        long_df[f"{m}_avg_{window}"] = (
            long_df.groupby(group_cols)[m]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            .astype(float)
        )

    long_df["form_points"] = np.select(
        [long_df["goals_for"] > long_df["goals_against"], long_df["goals_for"] == long_df["goals_against"]],
        [3, 1],
        default=0,
    )
    long_df[f"form_points_avg_{window}"] = (
        long_df.groupby(group_cols)["form_points"]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .astype(float)
    )

    home_feats = long_df[long_df["is_home"] == 1].copy()
    away_feats = long_df[long_df["is_home"] == 0].copy()

    feat_cols = [c for c in long_df.columns if c.endswith(f"avg_{window}")]
    home_rename = {c: f"home_{c}" for c in feat_cols}
    away_rename = {c: f"away_{c}" for c in feat_cols}

    home_feats = home_feats[["match_idx", *feat_cols]].rename(columns=home_rename)
    away_feats = away_feats[["match_idx", *feat_cols]].rename(columns=away_rename)

    out = out.reset_index(names="match_idx")
    out = out.merge(home_feats, on="match_idx", how="left")
    out = out.merge(away_feats, on="match_idx", how="left")

    base_home = ["goals_for", "goals_against", "xg_for", "xg_against", "form_points"]
    for m in base_home:
        h_col = f"home_{m}_avg_{window}"
        a_col = f"away_{m}_avg_{window}"
        if h_col in out.columns and a_col in out.columns:
            out[f"diff_{m}_avg_{window}"] = out[h_col] - out[a_col]

    out = out.drop(columns=["match_idx"])
    out = out.sort_values(["league", "season", "date", "home_team", "away_team"]).reset_index(drop=True)
    return out


def save_outputs(df: pd.DataFrame) -> None:
    csv_path = PROCESSED_DIR / "model_dataset.csv"
    parquet_path = PROCESSED_DIR / "model_dataset.parquet"
    xlsx_path = PROCESSED_DIR / "model_dataset.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print("=== Features concluídas ===")
    print(f"Dataset modelagem: {len(df):,} linhas")
    print(f"CSV: {csv_path}")
    print(f"Parquet: {parquet_path}")
    print(f"Excel: {xlsx_path}")


if __name__ == "__main__":
    data = load_data()
    data = add_targets(data)
    data = add_rolling_features(data, window=5)
    save_outputs(data)
