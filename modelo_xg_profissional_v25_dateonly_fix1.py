#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo profissional de previsão de xG da próxima rodada.

Principais decisões desta versão:
- Pipeline estritamente xG-first: não usa hg/ag como fallback nem como alvo auxiliar.
- Quebra se hxg/axg estiverem ausentes no histórico.
- Sem módulos de previsão de gols/mercados.
- Priors da liga calculados incrementalmente no tempo, sem vazamento.
- team_ha incremental por time, baseado em xG de jogos em casa, sem vazamento.
- Ratings estruturais log-lineares em xG, com verificação de convergência.
- Calibração temporal explícita para ponto e quantis (treino e calibração separados no tempo).
- Early stopping cronológico no XGBoost.
- Backtest walk-forward com MAE/RMSE/bias/corr/coverage80.
- Data quality rígido.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# Configuração
# ==============================
PASSADAS_FILE = "passadas.xlsx"
ATUAIS_FILE = "atuais.xlsx"
PROXIMA_FILE = "proxima.xlsx"
OUTPUT_FILE = "previsao_xg_profissional.xlsx"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

XG_CLIP_MIN = 0.01
XG_CLIP_MAX = 6.50

SEASON_DECAY = 0.88
CURRENT_SEASON_BONUS = 1.20
ATUAIS_SOURCE_BONUS = 1.04

RECENT_WINDOW = 5
RECENT_DECAY = 0.85
H2H_GAMES = 6
H2H_PRIOR_N = 3
TEAM_HA_PRIOR_N = 8
CALIBRATION_FRAC = 0.15
MIN_CALIBRATION = 100
N_BACKTEST_FOLDS = 3
MIN_BACKTEST_TRAIN = 500

ENSEMBLE_BLEND_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)
POINT_ALPHA_GRID = [0.0, 0.25, 0.50, 0.65, 0.80, 0.90, 1.0]
INTERVAL_BIN_QUANTILES = [0.25, 0.50, 0.75]
MAX_FEATURE_MISSING_FRAC = 0.55
LINEAR_CAL_MIN_IMPROV = 0.002
HOME_INTERVAL_TARGET = 0.76
AWAY_INTERVAL_TARGET = 0.80
INTERVAL_SCALE_GRID = [1.0, 1.1, 1.2, 1.3, 1.45]
HOME_BLEND_GRID = np.round(np.linspace(0.0, 1.0, 21), 2)
DATE_WINDOW_SHORT = 7
DATE_WINDOW_LONG = 14
DEFAULT_DATE_STEP_DAYS = 7
RATING_DATE_DAILY_DECAY = 0.9975

DEFAULT_HOME_XG = 1.55
DEFAULT_AWAY_XG = 1.25

SEP = "=" * 88


# ==============================
# Utilitários
# ==============================
def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s)) if not unicodedata.combining(c))


def norm_col(c: str) -> str:
    c = strip_accents(c).lower().strip()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", c)).strip("_")



def clean_team(x):
    if pd.isna(x):
        return np.nan
    return re.sub(r"\s+", " ", str(x).strip())



def canon_team(x: str) -> str:
    x = strip_accents(str(x)).lower().strip()
    x = re.sub(r"\s+", " ", x)
    x = re.sub(r"[^a-z0-9 ]+", "", x)
    return x



def safe_div(a, b, default=np.nan):
    if pd.isna(b) or float(b) == 0.0:
        return default
    return float(a) / float(b)



def weighted_mean(values, decay=RECENT_DECAY):
    if len(values) == 0:
        return np.nan
    n = len(values)
    w = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=float)
    arr = np.array(values, dtype=float)
    return float(np.average(arr, weights=w))


# ==============================
# Data quality e carregamento
# ==============================
def prep_df(df: pd.DataFrame, is_future: bool = False) -> pd.DataFrame:
    df = df.copy()
    df.columns = [norm_col(c) for c in df.columns]
    rename = {
        "mandante": "home",
        "visitante": "away",
        "year": "ano",
        "round": "rodada",
        "home_xg": "hxg",
        "away_xg": "axg",
        "xg_home": "hxg",
        "xg_away": "axg",
        "data": "date",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    required = ["home", "away", "date"] + ([] if is_future else ["hxg", "axg"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    for c in ["ano", "rodada", "hxg", "axg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["home"] = df["home"].map(clean_team)
    df["away"] = df["away"].map(clean_team)
    if "ano" not in df.columns:
        df["ano"] = df["date"].dt.year.astype(float)
    else:
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
        df["ano"] = df["ano"].fillna(df["date"].dt.year.astype(float))
    if "rodada" not in df.columns:
        df["rodada"] = np.nan
    else:
        df["rodada"] = pd.to_numeric(df["rodada"], errors="coerce")
    df["_row"] = np.arange(len(df))
    df["tick"] = (df["date"].astype("int64") // 10**9).astype(float)
    return df




def derive_rounds_from_dates(hist_df: pd.DataFrame, next_df: pd.DataFrame | None = None):
    """
    Deriva uma rodada interna pela ordem das datas dentro de cada ano.
    O input não precisa mais fornecer a coluna rodada.
    """
    frames = [hist_df.copy()]
    if next_df is not None:
        frames.append(next_df.copy())
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    if all_df["date"].isna().any():
        raise ValueError("Existem datas nulas; não é possível derivar rodada interna de forma confiável.")

    all_df = all_df.sort_values(["ano", "date", "_row"]).reset_index(drop=True)
    all_df["_rodada_derivada"] = all_df.groupby("ano")["date"].rank(method="dense").astype(int)

    hist_n = len(hist_df)
    hist_out = hist_df.copy()
    hist_out["rodada"] = all_df.loc[:hist_n - 1, "_rodada_derivada"].values

    if next_df is None:
        return hist_out

    next_out = next_df.copy()
    next_out["rodada"] = all_df.loc[hist_n:, "_rodada_derivada"].values
    return hist_out, next_out


def infer_missing_future_dates(hist_df: pd.DataFrame, next_df: pd.DataFrame) -> pd.DataFrame:
    next_df = next_df.copy()
    if "date" not in next_df.columns:
        next_df["date"] = pd.NaT
    missing_mask = next_df["date"].isna()
    if not missing_mask.any():
        return next_df

    hist_with_dates = hist_df.dropna(subset=["date"]).copy()
    if hist_with_dates.empty:
        return next_df

    for ano in sorted(next_df.loc[missing_mask, "ano"].dropna().astype(int).unique()):
        ref = (
            hist_with_dates.loc[hist_with_dates["ano"].astype(int) == int(ano), ["rodada", "date"]]
            .dropna()
            .groupby("rodada", as_index=False)["date"]
            .min()
            .sort_values("rodada")
        )
        if ref.empty:
            continue
        if len(ref) >= 2:
            round_gaps = ref["rodada"].diff().dropna().astype(float)
            date_gaps = ref["date"].diff().dropna().dt.days.astype(float)
            valid = round_gaps > 0
            if valid.any():
                step_days = float(np.median((date_gaps[valid] / round_gaps[valid]).clip(lower=1.0, upper=14.0)))
            else:
                step_days = float(DEFAULT_DATE_STEP_DAYS)
        else:
            step_days = float(DEFAULT_DATE_STEP_DAYS)

        base_round = int(ref["rodada"].iloc[0])
        base_date = pd.Timestamp(ref["date"].iloc[0])
        mask = missing_mask & (next_df["ano"].astype(float) == float(ano))
        est_dates = []
        for rodada in next_df.loc[mask, "rodada"].astype(float):
            delta_days = int(round((float(rodada) - base_round) * step_days))
            est_dates.append(base_date + pd.Timedelta(days=delta_days))
        next_df.loc[mask, "date"] = est_dates

    return next_df


def count_recent_matches(records, ref_date: pd.Timestamp, days: int) -> int:
    if ref_date is None or pd.isna(ref_date):
        return 0
    cutoff = ref_date - pd.Timedelta(days=days)
    return int(sum(1 for d in records if pd.notna(d) and d > cutoff and d < ref_date))


def rest_days(last_date, ref_date: pd.Timestamp, default_value: float = 7.0) -> float:
    if last_date is None or pd.isna(last_date) or ref_date is None or pd.isna(ref_date):
        return float(default_value)
    return float(max((pd.Timestamp(ref_date) - pd.Timestamp(last_date)).days, 0))


def default_calendar_state():
    return {
        "last_overall": pd.NaT,
        "last_home": pd.NaT,
        "last_away": pd.NaT,
        "overall_dates": deque(maxlen=60),
        "home_dates": deque(maxlen=30),
        "away_dates": deque(maxlen=30),
    }


def validate_history_df(df: pd.DataFrame, name: str) -> None:
    problems = []
    for c in ["ano", "home", "away", "hxg", "axg", "date"]:
        if c not in df.columns:
            problems.append(f"{name}: coluna ausente {c}")
            continue
        if df[c].isna().any():
            problems.append(f"{name}: {c} com {int(df[c].isna().sum())} nulos")

    if "rodada" in df.columns and df["rodada"].notna().any() and (df["rodada"] <= 0).any():
        problems.append(f"{name}: rodada <= 0 em {int((df['rodada'] <= 0).sum())} linhas")
    if (df["ano"] <= 0).any():
        problems.append(f"{name}: ano <= 0 em {int((df['ano'] <= 0).sum())} linhas")
    if (df["hxg"] < 0).any() or (df["axg"] < 0).any():
        problems.append(
            f"{name}: xG negativo (hxg={int((df['hxg'] < 0).sum())}, axg={int((df['axg'] < 0).sum())})"
        )
    if (df["home"] == df["away"]).any():
        problems.append(f"{name}: home == away em {int((df['home'] == df['away']).sum())} linhas")

    dup_cols = ["date", "home", "away"]
    dups = df.duplicated(subset=dup_cols, keep=False)
    if dups.any():
        problems.append(f"{name}: {int(dups.sum())} linhas duplicadas por {dup_cols}")

    alias_map = defaultdict(set)
    for t in pd.concat([df["home"], df["away"]]).dropna().astype(str):
        alias_map[canon_team(t)].add(t)
    alias_conflicts = {k: sorted(v) for k, v in alias_map.items() if len(v) > 1}
    if alias_conflicts:
        examples = "; ".join([f"{k}: {v}" for k, v in list(alias_conflicts.items())[:5]])
        problems.append(f"{name}: aliases prováveis detectados -> {examples}")

    # Verificação simples de consistência intra-temporada por time
    # Só faz sentido se a coluna rodada estiver realmente preenchida no input.
    if "rodada" in df.columns and df["rodada"].notna().any():
        for team, grp in df.groupby(["ano", "home"]):
            rodada_grp = grp["rodada"].dropna()
            if rodada_grp.duplicated().any():
                problems.append(f"{name}: time mandante {team} tem rodadas repetidas")
                break
        for team, grp in df.groupby(["ano", "away"]):
            rodada_grp = grp["rodada"].dropna()
            if rodada_grp.duplicated().any():
                problems.append(f"{name}: time visitante {team} tem rodadas repetidas")
                break

    if problems:
        raise ValueError("Falha de data quality:\n- " + "\n- ".join(problems))



def validate_future_df(df: pd.DataFrame, hist_teams: set[str], name: str) -> None:
    problems = []
    for c in ["ano", "home", "away", "date"]:
        if c not in df.columns:
            problems.append(f"{name}: coluna ausente {c}")
            continue
        if df[c].isna().any():
            problems.append(f"{name}: {c} com {int(df[c].isna().sum())} nulos")
    if df["date"].isna().any():
        problems.append(f"{name}: date com {int(df['date'].isna().sum())} nulos")
    if "rodada" in df.columns and df["rodada"].notna().any() and (df["rodada"] <= 0).any():
        problems.append(f"{name}: rodada <= 0 em {int((df['rodada'] <= 0).sum())} linhas")
    if (df["home"] == df["away"]).any():
        problems.append(f"{name}: home == away em {int((df['home'] == df['away']).sum())} linhas")

    future_teams = set(pd.concat([df["home"], df["away"]]).dropna().astype(str).unique())
    missing_teams = sorted(t for t in future_teams if t not in hist_teams)
    if missing_teams:
        problems.append(f"{name}: times sem histórico detectados: {missing_teams}")

    if problems:
        raise ValueError("Falha de data quality (próxima rodada):\n- " + "\n- ".join(problems))


# ==============================
# Estado acumulado xG-only
# ==============================
def make_block():
    return {"games": 0, "xgf": 0.0, "xga": 0.0}



def update_block(block, xgf: float, xga: float):
    block["games"] += 1
    block["xgf"] += float(xgf)
    block["xga"] += float(xga)



def default_state():
    return {"overall": make_block(), "home": make_block(), "away": make_block()}



def make_recent():
    return {k: deque(maxlen=RECENT_WINDOW) for k in ["xgf", "xga"]}



def default_recent():
    return {"overall": make_recent(), "home": make_recent(), "away": make_recent()}



def update_recent(rec, xgf: float, xga: float):
    rec["xgf"].append(float(xgf))
    rec["xga"].append(float(xga))



def build_rank_table_xg(season_tbl, teams, prev_summary):
    rows = []
    for team in teams:
        b = season_tbl.get(team, {}).get("overall", make_block())
        g = b["games"]
        prev = prev_summary.get(team, {})
        xgf_pg = safe_div(b["xgf"], g, DEFAULT_HOME_XG)
        xga_pg = safe_div(b["xga"], g, DEFAULT_AWAY_XG)
        xgd_pg = xgf_pg - xga_pg
        rows.append(
            {
                "team": team,
                "games": g,
                "xgf": b["xgf"],
                "xga": b["xga"],
                "xgd": b["xgf"] - b["xga"],
                "xgf_pg": xgf_pg,
                "xga_pg": xga_pg,
                "xgd_pg": xgd_pg,
                "prev_xgd_pg": prev.get("prev_xgd_pg", np.nan),
            }
        )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values(["xgd_pg", "xgf_pg", "games"], ascending=[False, False, False]).reset_index(drop=True)
    out["rank_xg"] = np.arange(1, len(out) + 1)
    out["rank_pct"] = out["rank_xg"] / max(len(out), 1)
    return out[["team", "rank_xg", "rank_pct", "games", "xgf", "xga", "xgd", "xgf_pg", "xga_pg", "xgd_pg"]]



def finalize_season(season_tbl, teams, prev_summary):
    ranking = build_rank_table_xg(season_tbl, teams, prev_summary)
    out = dict(prev_summary)
    for _, r in ranking.iterrows():
        out[r["team"]] = {
            "prev_rank_xg": int(r["rank_xg"]),
            "prev_rank_pct": float(r["rank_pct"]),
            "prev_xgf_pg": float(r["xgf_pg"]),
            "prev_xga_pg": float(r["xga_pg"]),
            "prev_xgd_pg": float(r["xgd_pg"]),
        }
    return out


# ==============================
# Ratings estruturais em xG
# ==============================
def default_home_adv(df: pd.DataFrame) -> float:
    home_mean = max(float(df["hxg"].mean()), 0.10)
    away_mean = max(float(df["axg"].mean()), 0.10)
    return float(np.log(home_mean / away_mean))



def compute_xg_ratings(df: pd.DataFrame, decay: float = 0.985) -> Tuple[Dict[str, float], Dict[str, float], float]:
    if len(df) < 20:
        return {}, {}, default_home_adv(df)

    sort_cols = ["date", "ano", "rodada", "_row"] if "date" in df.columns else ["ano", "rodada", "_row"]
    work = df.sort_values(sort_cols).reset_index(drop=True).copy()
    teams = sorted(pd.unique(work[["home", "away"]].values.ravel()))
    idx = {t: i for i, t in enumerate(teams)}
    hi = np.array([idx[t] for t in work["home"]], dtype=int)
    ai = np.array([idx[t] for t in work["away"]], dtype=int)
    y_h = np.log(np.clip(work["hxg"].values.astype(float), XG_CLIP_MIN, XG_CLIP_MAX))
    y_a = np.log(np.clip(work["axg"].values.astype(float), XG_CLIP_MIN, XG_CLIP_MAX))
    if "date" in work.columns and work["date"].notna().all():
        days_ago = (work["date"].max() - work["date"]).dt.days.astype(float).values
        w = np.power(RATING_DATE_DAILY_DECAY, days_ago)
    else:
        ticks = work["tick"].values.astype(float)
        max_tick = ticks.max()
        w = (decay ** (max_tick - ticks)).astype(float)
    n = len(teams)

    def loss(p):
        att = p[:n]
        dfe = p[n : 2 * n]
        ha = p[-1]
        pred_h = att[hi] - dfe[ai] + ha
        pred_a = att[ai] - dfe[hi]
        err = (y_h - pred_h) ** 2 + (y_a - pred_a) ** 2
        return float(np.dot(w, err))

    x0 = np.zeros(2 * n + 1, dtype=float)
    x0[-1] = default_home_adv(work)
    res = minimize(
        loss,
        x0,
        method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda p: float(np.sum(p[:n]))}],
        options={"maxiter": 800, "ftol": 1e-9},
    )
    if not res.success:
        raise RuntimeError(f"Falha na otimização dos ratings xG: {res.message}")

    p = res.x
    att = {teams[i]: float(p[i]) for i in range(n)}
    dfe = {teams[i]: float(p[n + i]) for i in range(n)}
    ha = float(p[-1])
    return att, dfe, ha



def build_rating_snapshots(hist_df: pd.DataFrame, next_df: pd.DataFrame) -> dict[tuple[pd.Timestamp, int, int], tuple[dict, dict, float]]:
    """
    Snapshot de ratings estruturais no início de cada data de jogo.
    Usa apenas jogos anteriores à data-alvo, eliminando vazamento intra-rodada quando datas reais existirem.
    """
    work_hist = hist_df.sort_values(["date", "ano", "rodada", "_row"]).reset_index(drop=True).copy()
    target_keys = (
        pd.concat(
            [
                work_hist[["date", "ano", "rodada"]],
                next_df[["date", "ano", "rodada"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values(["date", "ano", "rodada"])
        .itertuples(index=False, name=None)
    )

    snapshots: dict[tuple[pd.Timestamp, int, int], tuple[dict, dict, float]] = {}
    for date, ano, rodada in target_keys:
        past = work_hist[work_hist["date"] < pd.Timestamp(date)]
        if len(past) >= 20:
            snapshots[(pd.Timestamp(date), int(ano), int(rodada))] = compute_xg_ratings(past)
        else:
            snapshots[(pd.Timestamp(date), int(ano), int(rodada))] = ({}, {}, default_home_adv(work_hist))
    return snapshots


# ==============================
# Features incrementalmente sem vazamento
# ==============================
def current_league_priors(lg):
    g = max(int(lg["games"]), 0)
    if g == 0:
        return {
            "home_xgf_pg": DEFAULT_HOME_XG,
            "away_xgf_pg": DEFAULT_AWAY_XG,
            "home_xga_pg": DEFAULT_AWAY_XG,
            "away_xga_pg": DEFAULT_HOME_XG,
        }
    return {
        "home_xgf_pg": float(lg["home_xgf"] / g),
        "away_xgf_pg": float(lg["away_xgf"] / g),
        "home_xga_pg": float(lg["home_xga"] / g),
        "away_xga_pg": float(lg["away_xga"] / g),
    }



def shrunk_rate(total: float, games: int, league_mean: float, prior_n: int) -> float:
    w = games / max(games + prior_n, 1)
    empirical = safe_div(total, games, league_mean)
    return float(w * empirical + (1.0 - w) * league_mean)



def block_features(prefix: str, block: dict, xgf_lm: float, xga_lm: float, prior_n_xgf: int = 5, prior_n_xga: int = 8):
    g = block["games"]
    xgf_pg = shrunk_rate(block["xgf"], g, xgf_lm, prior_n_xgf)
    xga_pg = shrunk_rate(block["xga"], g, xga_lm, prior_n_xga)
    return {
        f"{prefix}_g": float(g),
        f"{prefix}_xgf_pg": xgf_pg,
        f"{prefix}_xga_pg": xga_pg,
        f"{prefix}_xgd_pg": xgf_pg - xga_pg,
    }



def recent_features(prefix: str, rec: dict, xgf_lm: float, xga_lm: float, prior_n: int = 4):
    n = len(rec["xgf"])
    xgf = weighted_mean(rec["xgf"])
    xga = weighted_mean(rec["xga"])
    xgf = shrunk_rate((xgf or 0.0) * n if pd.notna(xgf) else 0.0, n, xgf_lm, prior_n)
    xga = shrunk_rate((xga or 0.0) * n if pd.notna(xga) else 0.0, n, xga_lm, prior_n)
    return {
        f"{prefix}_n": float(n),
        f"{prefix}_xgf": xgf,
        f"{prefix}_xga": xga,
        f"{prefix}_xgd": xgf - xga,
    }




def date_context_features(prefix: str, cal_state: dict, ref_date: pd.Timestamp):
    last_overall = rest_days(cal_state.get("last_overall"), ref_date)
    last_venue = rest_days(cal_state.get(f"last_{prefix}"), ref_date)
    return {
        f"{prefix}_rest_overall_days": last_overall,
        f"{prefix}_rest_venue_days": last_venue,
        f"{prefix}_games_last7": float(count_recent_matches(cal_state.get("overall_dates", []), ref_date, DATE_WINDOW_SHORT)),
        f"{prefix}_games_last14": float(count_recent_matches(cal_state.get("overall_dates", []), ref_date, DATE_WINDOW_LONG)),
        f"{prefix}_venue_games_last7": float(count_recent_matches(cal_state.get(f"{prefix}_dates", []), ref_date, DATE_WINDOW_SHORT)),
        f"{prefix}_venue_games_last14": float(count_recent_matches(cal_state.get(f"{prefix}_dates", []), ref_date, DATE_WINDOW_LONG)),
        f"{prefix}_is_short_rest": float(last_overall <= 4.0),
        f"{prefix}_is_long_rest": float(last_overall >= 8.0),
    }


def get_team_home_advantage(home_team: str, team_home_state: dict, lg_priors: dict) -> float:
    st = team_home_state.get(home_team, {"games": 0, "xgf": 0.0})
    g = int(st["games"])
    home_xgf = shrunk_rate(st["xgf"], g, lg_priors["home_xgf_pg"], TEAM_HA_PRIOR_N)
    ratio = max(home_xgf, 0.05) / max(lg_priors["home_xgf_pg"], 0.05)
    return float(np.clip(np.log(ratio), -0.35, 0.35))



def get_h2h_features(home: str, away: str, h2h_hist: dict, lg_priors: dict):
    key = tuple(sorted([home, away]))
    entries = list(h2h_hist.get(key, []))
    if not entries:
        return {
            "h2h_n": 0.0,
            "h2h_home_xgf": lg_priors["home_xgf_pg"],
            "h2h_away_xgf": lg_priors["away_xgf_pg"],
            "h2h_diff_xgf": lg_priors["home_xgf_pg"] - lg_priors["away_xgf_pg"],
        }
    hx, ax = 0.0, 0.0
    for e in entries:
        if e["home"] == home:
            hx += e["hxg"]
            ax += e["axg"]
        else:
            hx += e["axg"]
            ax += e["hxg"]
    n = len(entries)
    hx_s = shrunk_rate(hx, n, lg_priors["home_xgf_pg"], H2H_PRIOR_N)
    ax_s = shrunk_rate(ax, n, lg_priors["away_xgf_pg"], H2H_PRIOR_N)
    return {
        "h2h_n": float(n),
        "h2h_home_xgf": hx_s,
        "h2h_away_xgf": ax_s,
        "h2h_diff_xgf": hx_s - ax_s,
    }



def make_match_features(
    home: str,
    away: str,
    ano: int,
    rodada: int,
    match_date: pd.Timestamp,
    season_start_date: pd.Timestamp,
    career,
    season,
    recent,
    season_tbl,
    teams_by_year,
    prev_summary,
    lg,
    h2h_hist,
    att_r,
    def_r,
    global_ha,
    team_home_state,
    calendar_state,
):
    lg_priors = current_league_priors(lg)
    season_day = 0.0 if pd.isna(match_date) or pd.isna(season_start_date) else float(max((pd.Timestamp(match_date) - pd.Timestamp(season_start_date)).days, 0))
    f = {"ano": float(ano), "rodada": float(rodada), "is_early": float(rodada <= 5), "season_day": season_day}

    # blocos shrinkados
    f.update(block_features("h_c_home", career[home]["home"], lg_priors["home_xgf_pg"], lg_priors["home_xga_pg"]))
    f.update(block_features("a_c_away", career[away]["away"], lg_priors["away_xgf_pg"], lg_priors["away_xga_pg"]))
    f.update(block_features("h_s_home", season[home]["home"], lg_priors["home_xgf_pg"], lg_priors["home_xga_pg"]))
    f.update(block_features("a_s_away", season[away]["away"], lg_priors["away_xgf_pg"], lg_priors["away_xga_pg"]))
    f.update(block_features("h_c_overall", career[home]["overall"], (lg_priors["home_xgf_pg"] + lg_priors["away_xgf_pg"]) / 2, (lg_priors["home_xga_pg"] + lg_priors["away_xga_pg"]) / 2))
    f.update(block_features("a_c_overall", career[away]["overall"], (lg_priors["home_xgf_pg"] + lg_priors["away_xgf_pg"]) / 2, (lg_priors["home_xga_pg"] + lg_priors["away_xga_pg"]) / 2))
    f.update(recent_features("h_r_home", recent[home]["home"], lg_priors["home_xgf_pg"], lg_priors["home_xga_pg"]))
    f.update(recent_features("a_r_away", recent[away]["away"], lg_priors["away_xgf_pg"], lg_priors["away_xga_pg"]))

    # momentum
    f["mom_h_xgf"] = f["h_r_home_xgf"] - f["h_s_home_xgf_pg"]
    f["mom_h_xga"] = f["h_r_home_xga"] - f["h_s_home_xga_pg"]
    f["mom_a_xgf"] = f["a_r_away_xgf"] - f["a_s_away_xgf_pg"]
    f["mom_a_xga"] = f["a_r_away_xga"] - f["a_s_away_xga_pg"]
    # calendário / recência real por data
    f.update(date_context_features("home", calendar_state[home], match_date))
    f.update(date_context_features("away", calendar_state[away], match_date))
    f["rest_diff_overall"] = f["home_rest_overall_days"] - f["away_rest_overall_days"]
    f["rest_diff_venue"] = f["home_rest_venue_days"] - f["away_rest_venue_days"]
    f["games_last7_diff"] = f["home_games_last7"] - f["away_games_last7"]
    f["games_last14_diff"] = f["home_games_last14"] - f["away_games_last14"]
    f["venue_games_last7_diff"] = f["home_venue_games_last7"] - f["away_venue_games_last7"]
    f["venue_games_last14_diff"] = f["home_venue_games_last14"] - f["away_venue_games_last14"]


    # ranking xG-only
    curr_rank = build_rank_table_xg(season_tbl, teams_by_year.get(ano, []), prev_summary)
    rmap = curr_rank.set_index("team").to_dict("index") if not curr_rank.empty else {}
    hr = rmap.get(home, {})
    ar = rmap.get(away, {})
    for key in ["rank_xg", "rank_pct", "games", "xgf_pg", "xga_pg", "xgd_pg"]:
        f[f"h_rank_{key}"] = hr.get(key, np.nan)
        f[f"a_rank_{key}"] = ar.get(key, np.nan)

    hp = prev_summary.get(home, {})
    ap = prev_summary.get(away, {})
    for key in ["prev_rank_xg", "prev_rank_pct", "prev_xgf_pg", "prev_xga_pg", "prev_xgd_pg"]:
        f[f"h_{key}"] = hp.get(key, np.nan)
        f[f"a_{key}"] = ap.get(key, np.nan)

    # liga incremental
    f["lg_home_xgf_pg"] = lg_priors["home_xgf_pg"]
    f["lg_away_xgf_pg"] = lg_priors["away_xgf_pg"]
    f["lg_home_xga_pg"] = lg_priors["home_xga_pg"]
    f["lg_away_xga_pg"] = lg_priors["away_xga_pg"]

    # ratings estruturais com team_ha incremental
    att_h = att_r.get(home, 0.0)
    def_h = def_r.get(home, 0.0)
    att_a = att_r.get(away, 0.0)
    def_a = def_r.get(away, 0.0)
    team_ha = get_team_home_advantage(home, team_home_state, lg_priors)
    lam_h = float(np.exp(att_h - def_a + global_ha + team_ha))
    lam_a = float(np.exp(att_a - def_h))
    f.update(
        {
            "att_h": att_h,
            "def_h": def_h,
            "att_a": att_a,
            "def_a": def_a,
            "global_home_adv": float(global_ha),
            "team_home_adv": float(team_ha),
            "xg_rating_home": lam_h,
            "xg_rating_away": lam_a,
            "xg_rating_diff": lam_h - lam_a,
        }
    )

    # H2H incremental
    f.update(get_h2h_features(home, away, h2h_hist, lg_priors))

    # interações / diffs
    f["diff_rank_xgd"] = f["h_rank_xgd_pg"] - f["a_rank_xgd_pg"] if pd.notna(f["h_rank_xgd_pg"]) and pd.notna(f["a_rank_xgd_pg"]) else np.nan
    f["diff_prev_xgd"] = f["h_prev_xgd_pg"] - f["a_prev_xgd_pg"] if pd.notna(f["h_prev_xgd_pg"]) and pd.notna(f["a_prev_xgd_pg"]) else np.nan
    f["diff_season_xgf"] = f["h_s_home_xgf_pg"] - f["a_s_away_xga_pg"]
    f["diff_recent_xgf"] = f["h_r_home_xgf"] - f["a_r_away_xga"]
    f["ratio_home_attack_vs_away_def"] = np.clip(f["h_s_home_xgf_pg"] / max(f["a_s_away_xga_pg"], 0.25), 0.0, 5.0)
    f["ratio_away_attack_vs_home_def"] = np.clip(f["a_s_away_xgf_pg"] / max(f["h_s_home_xga_pg"], 0.25), 0.0, 5.0)
    return f



def build_datasets(hist_df: pd.DataFrame, next_df: pd.DataFrame, rating_snapshots: dict, final_ratings: tuple[dict, dict, float]):
    hist_df = hist_df.sort_values(["date", "ano", "rodada", "_row"]).reset_index(drop=True)
    next_df = next_df.sort_values(["date", "ano", "rodada", "_row"]).reset_index(drop=True)

    teams_by_year = defaultdict(set)
    for df in [hist_df, next_df]:
        for _, r in df.iterrows():
            teams_by_year[int(r["ano"])].add(r["home"])
            teams_by_year[int(r["ano"])].add(r["away"])
    teams_by_year = {k: sorted(v) for k, v in teams_by_year.items()}

    career = defaultdict(default_state)
    season = defaultdict(default_state)
    recent = defaultdict(default_recent)
    season_tbl = defaultdict(default_state)
    prev_summary = {}
    lg = {"games": 0, "home_xgf": 0.0, "away_xgf": 0.0, "home_xga": 0.0, "away_xga": 0.0}
    h2h_hist = {}
    team_home_state = defaultdict(lambda: {"games": 0, "xgf": 0.0})
    calendar_state = defaultdict(default_calendar_state)
    season_start_dates = {int(k): pd.Timestamp(v) for k, v in pd.concat([hist_df[["ano", "date"]], next_df[["ano", "date"]]], ignore_index=True).dropna().groupby("ano")["date"].min().to_dict().items()}

    cur_year = None
    cur_teams = []
    rows = []

    def roll_year(ny: int):
        nonlocal cur_year, cur_teams, season, season_tbl, prev_summary
        if cur_year is None:
            cur_year = ny
            cur_teams = teams_by_year.get(ny, [])
            return
        if ny != cur_year:
            prev_summary = finalize_season(season_tbl, cur_teams, prev_summary)
            season = defaultdict(default_state)
            season_tbl = defaultdict(default_state)
            cur_year = ny
            cur_teams = teams_by_year.get(ny, [])

    for (match_date, ano, rodada), grp in hist_df.groupby(["date", "ano", "rodada"], sort=True):
        match_date = pd.Timestamp(match_date)
        ano = int(ano)
        rodada = int(rodada)
        roll_year(ano)
        att_r, def_r, g_ha = rating_snapshots.get((match_date, ano, rodada), ({}, {}, default_home_adv(hist_df)))

        staged_updates = []
        for _, row in grp.iterrows():
            home = row["home"]
            away = row["away"]
            feats = make_match_features(
                home=home,
                away=away,
                ano=ano,
                rodada=rodada,
                match_date=match_date,
                season_start_date=season_start_dates.get(ano, match_date),
                career=career,
                season=season,
                recent=recent,
                season_tbl=season_tbl,
                teams_by_year=teams_by_year,
                prev_summary=prev_summary,
                lg=lg,
                h2h_hist=h2h_hist,
                att_r=att_r,
                def_r=def_r,
                global_ha=g_ha,
                team_home_state=team_home_state,
                calendar_state=calendar_state,
            )
            feats.update(
                {
                    "home": home,
                    "away": away,
                    "hxg": float(row["hxg"]),
                    "axg": float(row["axg"]),
                    "dataset_source": str(row["dataset_source"]),
                }
            )
            rows.append(feats)
            staged_updates.append((home, away, float(row["hxg"]), float(row["axg"]), match_date))

        for home, away, hxg, axg, match_date in staged_updates:
            for st in [career, season, season_tbl]:
                update_block(st[home]["overall"], hxg, axg)
                update_block(st[home]["home"], hxg, axg)
                update_block(st[away]["overall"], axg, hxg)
                update_block(st[away]["away"], axg, hxg)
            update_recent(recent[home]["overall"], hxg, axg)
            update_recent(recent[home]["home"], hxg, axg)
            update_recent(recent[away]["overall"], axg, hxg)
            update_recent(recent[away]["away"], axg, hxg)
            lg["games"] += 1
            lg["home_xgf"] += hxg
            lg["away_xgf"] += axg
            lg["home_xga"] += axg
            lg["away_xga"] += hxg
            key = tuple(sorted([home, away]))
            if key not in h2h_hist:
                h2h_hist[key] = deque(maxlen=H2H_GAMES)
            h2h_hist[key].append({"home": home, "away": away, "hxg": hxg, "axg": axg})
            team_home_state[home]["games"] += 1
            team_home_state[home]["xgf"] += hxg
            calendar_state[home]["last_overall"] = match_date
            calendar_state[home]["last_home"] = match_date
            calendar_state[home]["overall_dates"].append(match_date)
            calendar_state[home]["home_dates"].append(match_date)
            calendar_state[away]["last_overall"] = match_date
            calendar_state[away]["last_away"] = match_date
            calendar_state[away]["overall_dates"].append(match_date)
            calendar_state[away]["away_dates"].append(match_date)

    hist_feats = pd.DataFrame(rows)

    final_att, final_def, final_ha = final_ratings
    next_rows = []
    fut_year = cur_year
    fut_prev_summary = prev_summary
    fut_season_tbl = season_tbl
    fut_teams = cur_teams

    for (match_date, ano, rodada), grp in next_df.groupby(["date", "ano", "rodada"], sort=True):
        match_date = pd.Timestamp(match_date)
        ano = int(ano)
        rodada = int(rodada)
        if fut_year is None:
            fut_year = ano
            fut_teams = teams_by_year.get(ano, [])
        elif ano != fut_year:
            fut_prev_summary = finalize_season(fut_season_tbl, fut_teams, fut_prev_summary)
            fut_season_tbl = defaultdict(default_state)
            fut_year = ano
            fut_teams = teams_by_year.get(ano, [])

        for _, row in grp.iterrows():
            feats = make_match_features(
                home=row["home"],
                away=row["away"],
                ano=ano,
                rodada=rodada,
                match_date=match_date,
                season_start_date=season_start_dates.get(ano, match_date),
                career=career,
                season=season,
                recent=recent,
                season_tbl=fut_season_tbl,
                teams_by_year=teams_by_year,
                prev_summary=fut_prev_summary,
                lg=lg,
                h2h_hist=h2h_hist,
                att_r=final_att,
                def_r=final_def,
                global_ha=final_ha,
                team_home_state=team_home_state,
                calendar_state=calendar_state,
            )
            feats.update({"home": row["home"], "away": row["away"]})
            next_rows.append(feats)

    next_feats = pd.DataFrame(next_rows)
    ranking = build_rank_table_xg(season_tbl, cur_teams, prev_summary) if cur_year is not None else pd.DataFrame()
    return hist_feats, next_feats, ranking


# ==============================
# Pesos amostrais
# ==============================
def sample_weights(df: pd.DataFrame) -> np.ndarray:
    years = df["ano"].astype(float).values
    rounds = df["rodada"].astype(float).values
    max_year = np.nanmax(years)
    season_w = np.power(SEASON_DECAY, max_year - years)
    season_w = np.where(years == max_year, season_w * CURRENT_SEASON_BONUS, season_w)
    max_round_year = df.groupby("ano")["rodada"].transform("max").values.astype(float)
    round_pos = rounds / np.maximum(max_round_year, 1.0)
    round_w = 0.88 + 0.24 * round_pos
    src = df.get("dataset_source", pd.Series(["hist"] * len(df))).astype(str).values
    src_w = np.where(src == "atuais", ATUAIS_SOURCE_BONUS, 1.0)
    return season_w * round_w * src_w


# ==============================
# Modelagem
# ==============================
def get_xgb_model(target_mode: str = "direct", early_stopping_rounds: int | None = 40):
    from xgboost import XGBRegressor

    common = dict(
        n_estimators=110,
        learning_rate=0.045,
        max_depth=3,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.20,
        reg_lambda=2.00,
        gamma=0.05,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )
    if early_stopping_rounds is not None:
        common["early_stopping_rounds"] = int(early_stopping_rounds)

    if target_mode == "direct":
        return XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.25, **common), "xgb_tweedie"
    if target_mode == "log1p":
        return XGBRegressor(objective="reg:squarederror", **common), "xgb_log1p"
    raise ValueError(f"target_mode inválido: {target_mode}")



def train_one_model(X: pd.DataFrame, y: pd.Series, w: np.ndarray, target_mode: str = "direct"):
    n = len(X)
    val_size = max(60, int(n * 0.12))
    y_fit_all = np.log1p(y) if target_mode == "log1p" else y
    if n - val_size < 120:
        model, name = get_xgb_model(target_mode=target_mode, early_stopping_rounds=None)
        model.fit(X, y_fit_all, sample_weight=w, verbose=False)
        return model, name, target_mode

    X_fit = X.iloc[:-val_size]
    X_val = X.iloc[-val_size:]
    y_fit = y_fit_all.iloc[:-val_size]
    y_val = y_fit_all.iloc[-val_size:]
    w_fit = w[:-val_size]
    w_val = w[-val_size:]

    model, name = get_xgb_model(target_mode=target_mode, early_stopping_rounds=25)
    model.fit(
        X_fit,
        y_fit,
        sample_weight=w_fit,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )
    return model, name, target_mode



def predict_trained_model(model, X: pd.DataFrame, target_mode: str) -> np.ndarray:
    pred = np.asarray(model.predict(X), dtype=float)
    if target_mode == "log1p":
        pred = np.expm1(pred)
    return np.clip(pred, XG_CLIP_MIN, XG_CLIP_MAX)



def select_feature_columns(train_df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    protected = {
        "xg_rating_home",
        "xg_rating_away",
        "xg_rating_diff",
        "global_home_adv",
        "team_home_adv",
        "h_s_home_xgf_pg",
        "h_s_home_xga_pg",
        "a_s_away_xgf_pg",
        "a_s_away_xga_pg",
        "h_r_home_xgf",
        "h_r_home_xga",
        "a_r_away_xgf",
        "a_r_away_xga",
        "diff_season_xgf",
        "diff_recent_xgf",
        "ratio_home_attack_vs_away_def",
        "ratio_away_attack_vs_home_def",
    }
    keep = []
    for c in feat_cols:
        s = train_df[c]
        miss = float(s.isna().mean())
        nunq = int(s.nunique(dropna=True))
        if c in protected or (miss <= MAX_FEATURE_MISSING_FRAC and nunq > 1):
            keep.append(c)
    return keep


class LinearCalibrator:
    def __init__(self, slope: float = 1.0, intercept: float = 0.0):
        self.slope = float(slope)
        self.intercept = float(intercept)

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        return self.intercept + self.slope * x


@dataclass
class SideBlendWeights:
    main: float = 1.0
    aux: float = 0.0
    anchor: float = 0.0


@dataclass
class SideIntervalTable:
    cuts: tuple[float, ...] = ()
    q10_by_bin: tuple[float, ...] = ()
    q90_by_bin: tuple[float, ...] = ()
    global_q10: float = 0.0
    global_q90: float = 0.0
    q10_scale: float = 1.0
    q90_scale: float = 1.0


@dataclass
class SideCalibrators:
    point: IsotonicRegression | None = None
    point_alpha: float = 1.0
    interval_table: SideIntervalTable | None = None
    q10_shift: float = 0.0
    q90_shift: float = 0.0


@dataclass
class PairModel:
    feat_cols: list[str]
    medians: pd.Series
    home_point_main: object
    home_point_aux: object
    away_point_main: object
    away_point_aux: object
    home_main_mode: str
    home_aux_mode: str
    away_main_mode: str
    away_aux_mode: str
    home_blend: SideBlendWeights
    away_blend: SideBlendWeights
    home_cal: SideCalibrators
    away_cal: SideCalibrators
    model_name: str

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.feat_cols].fillna(self.medians)

    def _predict_model(self, model, X: pd.DataFrame, target_mode: str) -> np.ndarray:
        return predict_trained_model(model, X, target_mode)

    def _blend_side(self, X: pd.DataFrame, anchor: np.ndarray, main_model, aux_model, main_mode: str, aux_mode: str, weights: SideBlendWeights) -> np.ndarray:
        main_pred = self._predict_model(main_model, X, main_mode)
        aux_pred = self._predict_model(aux_model, X, aux_mode)
        pred = weights.main * main_pred + weights.aux * aux_pred + weights.anchor * np.asarray(anchor, dtype=float)
        return np.clip(pred, XG_CLIP_MIN, XG_CLIP_MAX)

    def _apply_point_calibration(self, cal: SideCalibrators, raw_pred: np.ndarray) -> np.ndarray:
        raw_pred = np.asarray(raw_pred, dtype=float)
        if cal.point is None:
            return np.clip(raw_pred, XG_CLIP_MIN, XG_CLIP_MAX)
        iso_pred = np.asarray(cal.point.predict(raw_pred), dtype=float)
        pred = cal.point_alpha * raw_pred + (1.0 - cal.point_alpha) * iso_pred
        return np.clip(pred, XG_CLIP_MIN, XG_CLIP_MAX)

    def _interval_shifts(self, cal: SideCalibrators, point_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        point_pred = np.asarray(point_pred, dtype=float)
        table = cal.interval_table
        if table is None or len(table.q10_by_bin) == 0:
            return np.full(len(point_pred), cal.q10_shift), np.full(len(point_pred), cal.q90_shift)
        if len(table.cuts) == 0:
            bins = np.zeros(len(point_pred), dtype=int)
        else:
            bins = np.digitize(point_pred, np.asarray(table.cuts, dtype=float), right=False)
        q10 = np.asarray(table.q10_by_bin, dtype=float)[bins] * float(getattr(table, "q10_scale", 1.0))
        q90 = np.asarray(table.q90_by_bin, dtype=float)[bins] * float(getattr(table, "q90_scale", 1.0))
        return q10, q90

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._prepare(df)
        hp_raw = self._blend_side(
            X,
            df["xg_rating_home"].values,
            self.home_point_main,
            self.home_point_aux,
            self.home_main_mode,
            self.home_aux_mode,
            self.home_blend,
        )
        ap_raw = self._blend_side(
            X,
            df["xg_rating_away"].values,
            self.away_point_main,
            self.away_point_aux,
            self.away_main_mode,
            self.away_aux_mode,
            self.away_blend,
        )
        hp = self._apply_point_calibration(self.home_cal, hp_raw)
        ap = self._apply_point_calibration(self.away_cal, ap_raw)

        h_q10_shift, h_q90_shift = self._interval_shifts(self.home_cal, hp)
        a_q10_shift, a_q90_shift = self._interval_shifts(self.away_cal, ap)
        h10 = np.clip(hp + h_q10_shift, XG_CLIP_MIN, XG_CLIP_MAX)
        a10 = np.clip(ap + a_q10_shift, XG_CLIP_MIN, XG_CLIP_MAX)
        h90 = np.clip(hp + h_q90_shift, XG_CLIP_MIN, XG_CLIP_MAX)
        a90 = np.clip(ap + a_q90_shift, XG_CLIP_MIN, XG_CLIP_MAX)

        h10 = np.minimum(h10, hp)
        a10 = np.minimum(a10, ap)
        h90 = np.maximum(h90, hp)
        a90 = np.maximum(a90, ap)
        return pd.DataFrame(
            {
                "xg_home_raw": hp_raw,
                "xg_away_raw": ap_raw,
                "xg_home": hp,
                "xg_away": ap,
                "xg_home_q10": h10,
                "xg_home_q90": h90,
                "xg_away_q10": a10,
                "xg_away_q90": a90,
            }
        )



def choose_blend_weights(main_pred: np.ndarray, aux_pred: np.ndarray, anchor_pred: np.ndarray, target: np.ndarray) -> SideBlendWeights:
    main_pred = np.asarray(main_pred, dtype=float)
    aux_pred = np.asarray(aux_pred, dtype=float)
    anchor_pred = np.asarray(anchor_pred, dtype=float)
    target = np.asarray(target, dtype=float)
    best = SideBlendWeights(main=1.0, aux=0.0, anchor=0.0)
    best_mae = mean_absolute_error(target, main_pred)
    for w_main in ENSEMBLE_BLEND_GRID:
        for w_aux in ENSEMBLE_BLEND_GRID:
            w_anchor = round(1.0 - float(w_main) - float(w_aux), 10)
            if w_anchor < 0 or w_anchor > 1:
                continue
            pred = w_main * main_pred + w_aux * aux_pred + w_anchor * anchor_pred
            mae = mean_absolute_error(target, np.clip(pred, XG_CLIP_MIN, XG_CLIP_MAX))
            if mae + 1e-9 < best_mae:
                best_mae = mae
                best = SideBlendWeights(main=float(w_main), aux=float(w_aux), anchor=float(w_anchor))
    return best



def fit_blended_isotonic(preds: np.ndarray, target: np.ndarray):
    preds = np.asarray(preds, dtype=float)
    target = np.asarray(target, dtype=float)
    raw = np.clip(preds, XG_CLIP_MIN, XG_CLIP_MAX)
    base_mae = mean_absolute_error(target, raw)
    if len(preds) < 20 or np.std(preds) < 1e-8:
        return None, 1.0, raw
    x_mean = float(np.mean(preds))
    y_mean = float(np.mean(target))
    var_x = float(np.var(preds))
    slope = float(np.cov(preds, target, ddof=0)[0, 1] / var_x) if var_x > 1e-12 else 1.0
    slope = float(np.clip(slope, 0.90, 1.15))
    intercept = float(np.clip(y_mean - slope * x_mean, -0.18, 0.18))
    lin = LinearCalibrator(slope=slope, intercept=intercept)
    adj = np.clip(lin.predict(preds), XG_CLIP_MIN, XG_CLIP_MAX)
    if mean_absolute_error(target, adj) + LINEAR_CAL_MIN_IMPROV < base_mae:
        return lin, 1.0, adj
    return None, 1.0, raw


def fit_home_point_calibration(preds: np.ndarray, target: np.ndarray):
    """
    Calibração do ponto central do mandante:
    - nunca comprime a escala (slope >= 1.0)
    - só entra se trouxer ganho real de MAE no bloco temporal de calibração
    """
    preds = np.asarray(preds, dtype=float)
    target = np.asarray(target, dtype=float)
    raw = np.clip(preds, XG_CLIP_MIN, XG_CLIP_MAX)
    base_mae = mean_absolute_error(target, raw)
    if len(preds) < 20 or np.std(preds) < 1e-8:
        return None, 1.0, raw

    x_mean = float(np.mean(preds))
    y_mean = float(np.mean(target))
    var_x = float(np.var(preds))
    slope = float(np.cov(preds, target, ddof=0)[0, 1] / var_x) if var_x > 1e-12 else 1.0
    slope = float(np.clip(slope, 1.00, 1.25))
    intercept = float(np.clip(y_mean - slope * x_mean, -0.22, 0.22))
    lin = LinearCalibrator(slope=slope, intercept=intercept)
    adj = np.clip(lin.predict(preds), XG_CLIP_MIN, XG_CLIP_MAX)
    if mean_absolute_error(target, adj) + 0.001 < base_mae:
        return lin, 1.0, adj
    return None, 1.0, raw




def fit_interval_table(point_pred: np.ndarray, target: np.ndarray, target_coverage: float = 0.80) -> SideIntervalTable:
    point_pred = np.asarray(point_pred, dtype=float)
    target = np.asarray(target, dtype=float)
    resid = target - point_pred
    g_q10 = float(np.quantile(resid, 0.10))
    g_q90 = float(np.quantile(resid, 0.90))
    if len(point_pred) < 60:
        base = SideIntervalTable(cuts=(), q10_by_bin=(g_q10,), q90_by_bin=(g_q90,), global_q10=g_q10, global_q90=g_q90)
    else:
        raw_cuts = np.quantile(point_pred, INTERVAL_BIN_QUANTILES)
        cuts = []
        for v in raw_cuts:
            v = float(v)
            if not cuts or v > cuts[-1] + 1e-8:
                cuts.append(v)
        bins = np.digitize(point_pred, np.asarray(cuts, dtype=float), right=False)
        n_bins = len(cuts) + 1
        min_bin = max(15, int(len(point_pred) * 0.10))
        q10_by_bin = []
        q90_by_bin = []
        for b in range(n_bins):
            rr = resid[bins == b]
            if len(rr) < min_bin:
                q10_by_bin.append(g_q10)
                q90_by_bin.append(g_q90)
            else:
                q10_by_bin.append(float(np.quantile(rr, 0.10)))
                q90_by_bin.append(float(np.quantile(rr, 0.90)))
        base = SideIntervalTable(cuts=tuple(cuts), q10_by_bin=tuple(q10_by_bin), q90_by_bin=tuple(q90_by_bin), global_q10=g_q10, global_q90=g_q90)

    bins = np.zeros(len(point_pred), dtype=int) if len(base.cuts) == 0 else np.digitize(point_pred, np.asarray(base.cuts, dtype=float), right=False)
    q10_arr = np.asarray(base.q10_by_bin, dtype=float)
    q90_arr = np.asarray(base.q90_by_bin, dtype=float)
    best = base
    best_loss = float('inf')
    for s10 in INTERVAL_SCALE_GRID:
        lo = point_pred + q10_arr[bins] * float(s10)
        for s90 in INTERVAL_SCALE_GRID:
            hi = point_pred + q90_arr[bins] * float(s90)
            cov = float(np.mean((target >= lo) & (target <= hi)))
            width = float(np.mean(hi - lo))
            loss = abs(cov - target_coverage) + 0.02 * width
            if loss < best_loss - 1e-12:
                best_loss = loss
                best = SideIntervalTable(cuts=base.cuts, q10_by_bin=base.q10_by_bin, q90_by_bin=base.q90_by_bin, global_q10=base.global_q10, global_q90=base.global_q90, q10_scale=float(s10), q90_scale=float(s90))
    return best



def fit_pair_model(train_df: pd.DataFrame, feat_cols: list[str], calibration_frac: float = CALIBRATION_FRAC) -> PairModel:
    train_df = train_df.reset_index(drop=True)
    cal_n = max(MIN_CALIBRATION, int(len(train_df) * calibration_frac))
    if len(train_df) - cal_n < 200:
        cal_n = max(80, min(int(len(train_df) * 0.20), len(train_df) - 150))
    if len(train_df) - cal_n < 120:
        raise ValueError("Amostra insuficiente para separar treino e calibração temporal.")

    core = train_df.iloc[:-cal_n].copy()
    calib = train_df.iloc[-cal_n:].copy().reset_index(drop=True)
    used_feat_cols = select_feature_columns(core, feat_cols)
    med = core[used_feat_cols].median()
    X_core = core[used_feat_cols].fillna(med)
    w_core = sample_weights(core)

    y_h_core = core["hxg"].clip(XG_CLIP_MIN, XG_CLIP_MAX)
    y_a_core = core["axg"].clip(XG_CLIP_MIN, XG_CLIP_MAX)

    # Home: ensemble mais rico para reduzir compressão da escala.
    home_direct, home_direct_name, home_direct_mode = train_one_model(X_core, y_h_core, w_core, target_mode="direct")
    home_log, home_log_name, home_log_mode = train_one_model(X_core, y_h_core, w_core, target_mode="log1p")

    # Away: mantém o bloco mais simples e estável.
    away_main, away_main_name, away_main_mode = train_one_model(X_core, y_a_core, w_core, target_mode="direct")

    split_blend = max(35, int(len(calib) * 0.45))
    split_blend = min(split_blend, len(calib) - 35)
    if split_blend <= 0:
        split_blend = len(calib) // 2
    blend_df = calib.iloc[:split_blend].copy()
    final_cal = calib.iloc[split_blend:].copy()
    if len(final_cal) < 30:
        blend_df = calib.iloc[: len(calib) // 2].copy()
        final_cal = calib.iloc[len(calib) // 2 :].copy()

    X_blend = blend_df[used_feat_cols].fillna(med)
    X_fcal = final_cal[used_feat_cols].fillna(med)

    y_h_blend = blend_df["hxg"].clip(XG_CLIP_MIN, XG_CLIP_MAX).values
    y_a_blend = blend_df["axg"].clip(XG_CLIP_MIN, XG_CLIP_MAX).values

    home_direct_blend = predict_trained_model(home_direct, X_blend, home_direct_mode)
    home_log_blend = predict_trained_model(home_log, X_blend, home_log_mode)
    home_anchor_blend = blend_df["xg_rating_home"].clip(XG_CLIP_MIN, XG_CLIP_MAX).values

    away_main_blend = predict_trained_model(away_main, X_blend, away_main_mode)
    away_anchor_blend = blend_df["xg_rating_away"].clip(XG_CLIP_MIN, XG_CLIP_MAX).values

    # Home 3-way search:
    # loss = MAE + pequena penalidade por subdispersão e por viés.
    home_blend = SideBlendWeights(main=1.0, aux=0.0, anchor=0.0)
    best_home_loss = float("inf")
    target_home_std = float(np.std(y_h_blend))
    for w_main in HOME_BLEND_GRID:
        for w_aux in HOME_BLEND_GRID:
            w_anchor = round(1.0 - float(w_main) - float(w_aux), 10)
            if w_anchor < 0 or w_anchor > 1:
                continue
            cand_h = np.clip(
                w_main * home_direct_blend + w_aux * home_log_blend + w_anchor * home_anchor_blend,
                XG_CLIP_MIN,
                XG_CLIP_MAX,
            )
            mae_h = mean_absolute_error(y_h_blend, cand_h)
            underdisp = max(0.0, target_home_std - float(np.std(cand_h)))
            bias_pen = abs(float(np.mean(cand_h - y_h_blend)))
            loss_h = mae_h + 0.02 * underdisp + 0.01 * bias_pen
            if loss_h + 1e-12 < best_home_loss:
                best_home_loss = loss_h
                home_blend = SideBlendWeights(main=float(w_main), aux=float(w_aux), anchor=float(w_anchor))

    # Away 2-way: modelo direto + rating estrutural.
    away_blend = SideBlendWeights(main=1.0, aux=0.0, anchor=0.0)
    best_away = mean_absolute_error(y_a_blend, away_main_blend)
    for w_main in HOME_BLEND_GRID:
        w_anchor = round(1.0 - float(w_main), 10)
        if w_anchor < 0 or w_anchor > 1:
            continue
        cand_a = np.clip(
            w_main * away_main_blend + w_anchor * away_anchor_blend,
            XG_CLIP_MIN,
            XG_CLIP_MAX,
        )
        mae_a = mean_absolute_error(y_a_blend, cand_a)
        if mae_a + 1e-12 < best_away:
            best_away = mae_a
            away_blend = SideBlendWeights(main=float(w_main), aux=0.0, anchor=float(w_anchor))

    home_raw_fcal = np.clip(
        home_blend.main * predict_trained_model(home_direct, X_fcal, home_direct_mode)
        + home_blend.aux * predict_trained_model(home_log, X_fcal, home_log_mode)
        + home_blend.anchor * final_cal["xg_rating_home"].clip(XG_CLIP_MIN, XG_CLIP_MAX).values,
        XG_CLIP_MIN,
        XG_CLIP_MAX,
    )
    away_raw_fcal = np.clip(
        away_blend.main * predict_trained_model(away_main, X_fcal, away_main_mode)
        + away_blend.anchor * final_cal["xg_rating_away"].clip(XG_CLIP_MIN, XG_CLIP_MAX).values,
        XG_CLIP_MIN,
        XG_CLIP_MAX,
    )

    y_h_fcal = final_cal["hxg"].clip(XG_CLIP_MIN, XG_CLIP_MAX).values
    y_a_fcal = final_cal["axg"].clip(XG_CLIP_MIN, XG_CLIP_MAX).values

    # Mandante: calibração só não-compressiva.
    home_point_cal, home_point_alpha, hp_cal_corr = fit_home_point_calibration(home_raw_fcal, y_h_fcal)

    # Visitante: mantém a calibração linear padrão.
    away_point_cal, away_point_alpha, ap_cal_corr = fit_blended_isotonic(away_raw_fcal, y_a_fcal)

    home_interval = fit_interval_table(hp_cal_corr, y_h_fcal, target_coverage=HOME_INTERVAL_TARGET)
    away_interval = fit_interval_table(ap_cal_corr, y_a_fcal, target_coverage=AWAY_INTERVAL_TARGET)

    home_cal = SideCalibrators(
        point=home_point_cal,
        point_alpha=home_point_alpha,
        interval_table=home_interval,
        q10_shift=home_interval.global_q10,
        q90_shift=home_interval.global_q90,
    )
    away_cal = SideCalibrators(
        point=away_point_cal,
        point_alpha=away_point_alpha,
        interval_table=away_interval,
        q10_shift=away_interval.global_q10,
        q90_shift=away_interval.global_q90,
    )

    model_name = "v25_dateonly_home[xgb_tweedie+xgb_log1p+rating]_away[xgb_tweedie+rating]_date_primary(rest+calendar)"
    return PairModel(
        feat_cols=used_feat_cols,
        medians=med,
        home_point_main=home_direct,
        home_point_aux=home_log,
        away_point_main=away_main,
        away_point_aux=away_main,
        home_main_mode=home_direct_mode,
        home_aux_mode=home_log_mode,
        away_main_mode=away_main_mode,
        away_aux_mode=away_main_mode,
        home_blend=home_blend,
        away_blend=away_blend,
        home_cal=home_cal,
        away_cal=away_cal,
        model_name=model_name,
    )



# ==============================
# Backtest
# ==============================
def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))



def run_backtest(hist_f: pd.DataFrame, feat_cols: list[str], n_folds: int = N_BACKTEST_FOLDS, min_train: int = MIN_BACKTEST_TRAIN):
    n = len(hist_f)
    round_keys = hist_f[["ano", "rodada"]].drop_duplicates().reset_index(drop=True)
    round_starts = []
    for _, rr in round_keys.iterrows():
        idx = hist_f.index[(hist_f["ano"] == rr["ano"]) & (hist_f["rodada"] == rr["rodada"])].min()
        round_starts.append(int(idx))
    candidate_starts = [s for s in round_starts if s >= min_train]
    if not candidate_starts:
        return pd.DataFrame()

    starts = []
    for i in range(n_folds):
        pos = int(round(i * (len(candidate_starts) - 1) / max(n_folds - 1, 1)))
        starts.append(candidate_starts[pos])
    starts = sorted(set(starts))

    rows = []
    for fold, test_start in enumerate(starts, start=1):
        default_step = max(50, (n - min_train) // max(n_folds, 1))
        provisional_end = min(n, test_start + default_step)
        later = [s for s in round_starts if s >= provisional_end]
        test_end = later[0] if later else n
        if test_end <= test_start:
            continue

        tr = hist_f.iloc[:test_start].copy()
        te = hist_f.iloc[test_start:test_end].copy()
        model = fit_pair_model(tr, feat_cols)
        pred = model.predict(te)

        real_h = te["hxg"].values
        real_a = te["axg"].values
        pred_h = pred["xg_home"].values
        pred_a = pred["xg_away"].values

        mae_h = mean_absolute_error(real_h, pred_h)
        mae_a = mean_absolute_error(real_a, pred_a)
        rmse_h = math.sqrt(mean_squared_error(real_h, pred_h))
        rmse_a = math.sqrt(mean_squared_error(real_a, pred_a))
        bias_h = float(np.mean(pred_h - real_h))
        bias_a = float(np.mean(pred_a - real_a))
        corr_h = float(np.corrcoef(real_h, pred_h)[0, 1]) if len(te) > 3 else np.nan
        corr_a = float(np.corrcoef(real_a, pred_a)[0, 1]) if len(te) > 3 else np.nan
        cov_h = interval_coverage(real_h, pred["xg_home_q10"].values, pred["xg_home_q90"].values)
        cov_a = interval_coverage(real_a, pred["xg_away_q10"].values, pred["xg_away_q90"].values)
        hit50_h = float(np.mean(np.abs(pred_h - real_h) <= 0.50))
        hit50_a = float(np.mean(np.abs(pred_a - real_a) <= 0.50))
        hit75_h = float(np.mean(np.abs(pred_h - real_h) <= 0.75))
        hit75_a = float(np.mean(np.abs(pred_a - real_a) <= 0.75))

        rows.append(
            {
                "fold": fold,
                "n_train": len(tr),
                "n_test": len(te),
                "mae_home": round(mae_h, 4),
                "mae_away": round(mae_a, 4),
                "mae_mean": round((mae_h + mae_a) / 2, 4),
                "rmse_home": round(rmse_h, 4),
                "rmse_away": round(rmse_a, 4),
                "bias_home": round(bias_h, 4),
                "bias_away": round(bias_a, 4),
                "corr_home": round(corr_h, 4),
                "corr_away": round(corr_a, 4),
                "coverage80_home": round(cov_h, 4),
                "coverage80_away": round(cov_a, 4),
                "hit_le_050_home": round(hit50_h, 4),
                "hit_le_050_away": round(hit50_a, 4),
                "hit_le_075_home": round(hit75_h, 4),
                "hit_le_075_away": round(hit75_a, 4),
            }
        )
        print(
            f"  fold {fold}: MAE=({mae_h:.3f},{mae_a:.3f}) | "
            f"Bias=({bias_h:+.3f},{bias_a:+.3f}) | Cov80=({cov_h:.3f},{cov_a:.3f}) | "
            f"Hit<=0.50=({hit50_h:.3f},{hit50_a:.3f})"
        )
    return pd.DataFrame(rows)



# ==============================
# Régua oficial de performance
# ==============================
def _between(x: float, lo: float, hi: float) -> bool:
    return pd.notna(x) and (x >= lo) and (x <= hi)

def _le(x: float, thr: float) -> bool:
    return pd.notna(x) and (x <= thr)

def _ge(x: float, thr: float) -> bool:
    return pd.notna(x) and (x >= thr)

def compute_performance_metrics(bt: pd.DataFrame) -> dict:
    if bt is None or bt.empty:
        return {
            "mae_mean": np.nan,
            "mae_home": np.nan,
            "mae_away": np.nan,
            "coverage80_home": np.nan,
            "coverage80_away": np.nan,
            "coverage80_mean": np.nan,
            "hit_le_050_mean": np.nan,
            "hit_le_075_mean": np.nan,
            "fold_mae_range": np.nan,
            "home_away_mae_gap": np.nan,
            "bias_abs_mean": np.nan,
            "n_folds": 0,
        }

    mae_mean = float(bt["mae_mean"].mean())
    mae_home = float(bt["mae_home"].mean())
    mae_away = float(bt["mae_away"].mean())
    cov_h = float(bt["coverage80_home"].mean())
    cov_a = float(bt["coverage80_away"].mean())
    hit50 = float(pd.concat([bt["hit_le_050_home"], bt["hit_le_050_away"]], ignore_index=True).mean())
    hit75 = float(pd.concat([bt["hit_le_075_home"], bt["hit_le_075_away"]], ignore_index=True).mean())
    fold_mae_range = float(bt["mae_mean"].max() - bt["mae_mean"].min()) if len(bt) > 1 else 0.0
    home_away_mae_gap = float(abs(mae_home - mae_away))
    bias_abs_mean = float(
        pd.concat([bt["bias_home"].abs(), bt["bias_away"].abs()], ignore_index=True).mean()
    )
    return {
        "mae_mean": mae_mean,
        "mae_home": mae_home,
        "mae_away": mae_away,
        "coverage80_home": cov_h,
        "coverage80_away": cov_a,
        "coverage80_mean": float(np.nanmean([cov_h, cov_a])),
        "hit_le_050_mean": hit50,
        "hit_le_075_mean": hit75,
        "fold_mae_range": fold_mae_range,
        "home_away_mae_gap": home_away_mae_gap,
        "bias_abs_mean": bias_abs_mean,
        "n_folds": int(len(bt)),
    }

def classify_stability(fold_mae_range: float) -> str:
    if pd.isna(fold_mae_range):
        return "indefinida"
    if fold_mae_range <= 0.04:
        return "excelente"
    if fold_mae_range <= 0.06:
        return "boa"
    if fold_mae_range <= 0.08:
        return "aceitavel"
    return "ruim"

def classify_balance(home_away_mae_gap: float) -> str:
    if pd.isna(home_away_mae_gap):
        return "indefinido"
    if home_away_mae_gap <= 0.03:
        return "excelente"
    if home_away_mae_gap <= 0.05:
        return "boa"
    if home_away_mae_gap <= 0.07:
        return "aceitavel"
    return "ruim"

def classify_bias(bias_abs_mean: float) -> str:
    if pd.isna(bias_abs_mean):
        return "indefinido"
    if bias_abs_mean <= 0.05:
        return "excelente"
    if bias_abs_mean <= 0.08:
        return "bom"
    if bias_abs_mean <= 0.10:
        return "aceitavel"
    return "ruim"

def classify_project_band(metrics: dict) -> tuple[str, list[str]]:
    mae = metrics["mae_mean"]
    hit50 = metrics["hit_le_050_mean"]
    hit75 = metrics["hit_le_075_mean"]
    cov_h = metrics["coverage80_home"]
    cov_a = metrics["coverage80_away"]
    fold_rng = metrics["fold_mae_range"]
    gap = metrics["home_away_mae_gap"]

    reasons = []

    elite = (
        _le(mae, 0.45)
        and _ge(hit75, 0.83)
        and _ge(hit50, 0.64)
        and _between(cov_h, 0.76, 0.82)
        and _between(cov_a, 0.76, 0.82)
        and _le(fold_rng, 0.06)
        and _le(gap, 0.05)
    )
    if elite:
        return "elite", ["MAE de elite com estabilidade e cobertura calibrada."]

    muito_forte = (
        _between(mae, 0.45, 0.49)
        and _ge(hit75, 0.78)
        and _ge(hit50, 0.58)
        and _between(cov_h, 0.74, 0.83)
        and _between(cov_a, 0.74, 0.83)
        and _le(fold_rng, 0.08)
        and _le(gap, 0.07)
    )
    if muito_forte:
        return "muito forte", ["MAE muito forte e robustez consistente no walk-forward."]

    forte = (
        _between(mae, 0.50, 0.55)
        and _ge(hit75, 0.72)
        and _ge(hit50, 0.52)
        and _between(cov_h, 0.72, 0.84)
        and _between(cov_a, 0.72, 0.84)
        and _le(fold_rng, 0.08)
        and _le(gap, 0.07)
    )
    if forte:
        return "forte", ["Modelo competitivo com potencial real de edge, sem sinais graves de instabilidade."]

    bom = (
        _between(mae, 0.56, 0.65)
        and _ge(hit75, 0.65)
        and _ge(hit50, 0.45)
        and _between(cov_h, 0.68, 0.84)
        and _between(cov_a, 0.68, 0.84)
    )
    if bom:
        if not _le(fold_rng, 0.08):
            reasons.append("classificado como bom, mas com estabilidade entre folds ainda fraca")
        if not _le(gap, 0.07):
            reasons.append("classificado como bom, mas com desequilíbrio relevante entre mandante e visitante")
        if not reasons:
            reasons.append("modelo já competitivo, porém ainda abaixo da faixa forte")
        return "bom", reasons

    reasons.append("modelo ainda abaixo do nível competitivo desejado na régua oficial")
    if pd.notna(mae) and mae > 0.65:
        reasons.append("MAE médio acima de 0.65")
    if pd.notna(hit75) and hit75 < 0.65:
        reasons.append("hit-rate com erro <= 0.75 xG ainda baixo")
    return "ruim", reasons

def build_performance_scale_sheet(metrics: dict, band: str, reasons: list[str]) -> pd.DataFrame:
    rows = [
        {"grupo": "classificacao", "item": "faixa_oficial", "valor": band},
        {"grupo": "classificacao", "item": "motivo_principal", "valor": " | ".join(reasons)},
        {"grupo": "metricas", "item": "mae_mean", "valor": round(metrics["mae_mean"], 4) if pd.notna(metrics["mae_mean"]) else np.nan},
        {"grupo": "metricas", "item": "mae_home", "valor": round(metrics["mae_home"], 4) if pd.notna(metrics["mae_home"]) else np.nan},
        {"grupo": "metricas", "item": "mae_away", "valor": round(metrics["mae_away"], 4) if pd.notna(metrics["mae_away"]) else np.nan},
        {"grupo": "metricas", "item": "coverage80_home", "valor": round(metrics["coverage80_home"], 4) if pd.notna(metrics["coverage80_home"]) else np.nan},
        {"grupo": "metricas", "item": "coverage80_away", "valor": round(metrics["coverage80_away"], 4) if pd.notna(metrics["coverage80_away"]) else np.nan},
        {"grupo": "metricas", "item": "hit_le_050_mean", "valor": round(metrics["hit_le_050_mean"], 4) if pd.notna(metrics["hit_le_050_mean"]) else np.nan},
        {"grupo": "metricas", "item": "hit_le_075_mean", "valor": round(metrics["hit_le_075_mean"], 4) if pd.notna(metrics["hit_le_075_mean"]) else np.nan},
        {"grupo": "estabilidade", "item": "fold_mae_range", "valor": round(metrics["fold_mae_range"], 4) if pd.notna(metrics["fold_mae_range"]) else np.nan},
        {"grupo": "estabilidade", "item": "fold_mae_range_classe", "valor": classify_stability(metrics["fold_mae_range"])},
        {"grupo": "estabilidade", "item": "home_away_mae_gap", "valor": round(metrics["home_away_mae_gap"], 4) if pd.notna(metrics["home_away_mae_gap"]) else np.nan},
        {"grupo": "estabilidade", "item": "home_away_mae_gap_classe", "valor": classify_balance(metrics["home_away_mae_gap"])},
        {"grupo": "estabilidade", "item": "bias_abs_mean", "valor": round(metrics["bias_abs_mean"], 4) if pd.notna(metrics["bias_abs_mean"]) else np.nan},
        {"grupo": "estabilidade", "item": "bias_abs_mean_classe", "valor": classify_bias(metrics["bias_abs_mean"])},
        {"grupo": "metas", "item": "meta_imediata", "valor": "entrar em forte: MAE<=0.55, hit<=0.75>=0.74, cobertura 0.72-0.84, fold_range<=0.06"},
        {"grupo": "metas", "item": "meta_intermediaria", "valor": "entrar em muito forte: MAE<=0.49, hit<=0.75>=0.78, hit<=0.50>=0.58"},
        {"grupo": "metas", "item": "meta_final", "valor": "elite: MAE<0.45, sem vazamento, estabilidade alta, cobertura 0.76-0.82"},
    ]
    return pd.DataFrame(rows)

# ==============================
# Pipeline principal
# ==============================
def main(passadas_file=PASSADAS_FILE, atuais_file=ATUAIS_FILE, proxima_file=PROXIMA_FILE, output_file=OUTPUT_FILE):
    print(SEP)
    print("MODELO PROFISSIONAL DE PREVISÃO DE xG")
    print(SEP)

    for fp in [passadas_file, atuais_file, proxima_file]:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Arquivo não encontrado: {fp}")

    print("\n1) Carregando planilhas...")
    passadas = prep_df(pd.read_excel(passadas_file), is_future=False)
    atuais = prep_df(pd.read_excel(atuais_file), is_future=False)
    proxima = prep_df(pd.read_excel(proxima_file), is_future=True)
    passadas["dataset_source"] = "passadas"
    atuais["dataset_source"] = "atuais"

    print("2) Validando data quality...")
    validate_history_df(passadas, "passadas")
    validate_history_df(atuais, "atuais")
    hist = pd.concat([passadas, atuais], ignore_index=True, sort=False)
    hist = hist.sort_values(["date", "ano", "_row"]).reset_index(drop=True)
    proxima = infer_missing_future_dates(hist, proxima)
    hist, proxima = derive_rounds_from_dates(hist, proxima)
    hist_teams = set(pd.concat([hist["home"], hist["away"]]).dropna().astype(str).unique())
    validate_future_df(proxima, hist_teams, "proxima")
    print(f"  Histórico: {len(hist)} jogos | anos={sorted(hist['ano'].dropna().astype(int).unique().tolist())}")
    print(f"  Próxima base: {len(proxima)} jogos | data_ref={proxima['date'].min().date() if proxima['date'].notna().any() else 'sem_data'}")

    print("3) Computando ratings estruturais de xG...")
    rating_snapshots = build_rating_snapshots(hist, proxima)
    final_ratings = compute_xg_ratings(hist)
    print(f"  Snapshots estruturais calculados para {len(rating_snapshots)} chaves temporais (date/ano/rodada_interna)")
    print(f"  Home advantage estrutural final: {math.exp(final_ratings[2]):.3f}x")

    print("4) Montando dataset de features sem vazamento temporal...")
    hist_f, next_f, ranking = build_datasets(hist, proxima, rating_snapshots, final_ratings)
    exclude = {"home", "away", "hxg", "axg", "dataset_source"}
    feat_cols = [c for c in hist_f.columns if c not in exclude and pd.api.types.is_numeric_dtype(hist_f[c])]
    print(f"  Jogos históricos com features: {len(hist_f)}")
    print(f"  Número de features candidatas: {len(feat_cols)}")

    print("5) Rodando backtest walk-forward...")
    bt = run_backtest(hist_f, feat_cols)
    perf = compute_performance_metrics(bt)
    bt_mae = perf["mae_mean"]
    bt_cov_h = perf["coverage80_home"]
    bt_cov_a = perf["coverage80_away"]
    bt_hit50 = perf["hit_le_050_mean"]
    bt_hit75 = perf["hit_le_075_mean"]
    perf_band, perf_reasons = classify_project_band(perf)
    perf_sheet = build_performance_scale_sheet(perf, perf_band, perf_reasons)
    print(
        f"  Backtest médio: MAE={bt_mae:.4f} | Cov80_home={bt_cov_h:.3f} | "
        f"Cov80_away={bt_cov_a:.3f} | Hit<=0.50={bt_hit50:.3f} | Hit<=0.75={bt_hit75:.3f}"
    )
    print(
        f"  Régua oficial: faixa={perf_band.upper()} | "
        f"estabilidade={classify_stability(perf['fold_mae_range'])} | "
        f"equilibrio={classify_balance(perf['home_away_mae_gap'])}"
    )

    print("6) Treinando modelo final com ensemble temporal e calibração conservadora...")
    final_model = fit_pair_model(hist_f, feat_cols)
    pred = final_model.predict(next_f)
    df_pred = pd.concat([next_f[["home", "away"]].reset_index(drop=True), pred], axis=1)
    df_pred["xg_home_rating"] = next_f["xg_rating_home"].values
    df_pred["xg_away_rating"] = next_f["xg_rating_away"].values
    df_pred["team_home_adv"] = next_f["team_home_adv"].values

    print("\nPrevisão da próxima rodada:")
    print(f"{'Mandante':<22} {'Visitante':<22} {'xG_M':>6} {'Q10':>6} {'Q90':>6} {'xG_V':>6} {'Q10':>6} {'Q90':>6}")
    print("-" * 92)
    for _, r in df_pred.iterrows():
        print(
            f"{r['home']:<22} {r['away']:<22} {r['xg_home']:>6.2f} {r['xg_home_q10']:>6.2f} {r['xg_home_q90']:>6.2f} "
            f"{r['xg_away']:>6.2f} {r['xg_away_q10']:>6.2f} {r['xg_away_q90']:>6.2f}"
        )

    info = pd.DataFrame(
        [
            {"item": "modelo", "value": final_model.model_name},
            {"item": "random_state", "value": RANDOM_STATE},
            {"item": "xg_only_pipeline", "value": True},
            {"item": "strict_missing_xg_break", "value": True},
            {"item": "league_priors_incremental", "value": True},
            {"item": "team_ha_incremental", "value": True},
            {"item": "ratings_optimizer_checked", "value": True},
            {"item": "temporal_calibration", "value": True},
            {"item": "round_level_snapshot_ratings", "value": True},
            {"item": "same_round_leakage_removed", "value": True},
            {"item": "backtest_mae_mean", "value": round(bt_mae, 4)},
            {"item": "backtest_coverage80_home", "value": round(bt_cov_h, 4)},
            {"item": "backtest_coverage80_away", "value": round(bt_cov_a, 4)},
            {"item": "backtest_hit_le_050_mean", "value": round(bt_hit50, 4)},
            {"item": "backtest_hit_le_075_mean", "value": round(bt_hit75, 4)},
            {"item": "performance_band", "value": perf_band},
            {"item": "fold_mae_range", "value": round(perf["fold_mae_range"], 4)},
            {"item": "home_away_mae_gap", "value": round(perf["home_away_mae_gap"], 4)},
            {"item": "bias_abs_mean", "value": round(perf["bias_abs_mean"], 4)},
            {"item": "stability_class", "value": classify_stability(perf["fold_mae_range"])},
            {"item": "balance_class", "value": classify_balance(perf["home_away_mae_gap"])},
            {"item": "bias_class", "value": classify_bias(perf["bias_abs_mean"])},
            {"item": "n_hist_games", "value": len(hist_f)},
            {"item": "historical_date_features", "value": True},
            {"item": "date_is_primary_time_axis", "value": True},
            {"item": "rodada_is_derived_from_date", "value": True},
            {"item": "n_features", "value": len(final_model.feat_cols)},
        ]
    )

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_pred.to_excel(writer, sheet_name="previsoes_xg", index=False)
        bt.to_excel(writer, sheet_name="backtest_xg", index=False)
        perf_sheet.to_excel(writer, sheet_name="regua_performance", index=False)
        if not ranking.empty:
            ranking.to_excel(writer, sheet_name="ranking_xg", index=False)
        info.to_excel(writer, sheet_name="info_modelo", index=False)

    print(f"\nArquivo gerado: {output_file}")
    return df_pred, bt, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--passadas", default=PASSADAS_FILE)
    parser.add_argument("--atuais", default=ATUAIS_FILE)
    parser.add_argument("--proxima", default=PROXIMA_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()
    main(args.passadas, args.atuais, args.proxima, args.output)
