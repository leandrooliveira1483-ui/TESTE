#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODELO PREDITIVO DE xG — VERSÃO LIMPA (xG PURO)                          ║
║  Target: xG esperado por time na próxima rodada                            ║
║                                                                              ║
║  AUDITORIA — 14 PROBLEMAS CORRIGIDOS:                                      ║
║                                                                              ║
║  #1  🔴  prep_df: sem proxy fallback — quebra se hxg/axg ausente           ║
║  #2  🔴  required: exige hxg e axg explicitamente                          ║
║  #3  🔴  SHRINK_PRIOR: removidos gf_pg e ga_pg                            ║
║  #4  🔴  shrunk_block_feats: removidos gf_pg/ga_pg das features            ║
║  #5  🔴  shrunk_recent_feats: removidos gf/ga das features                 ║
║  #6  🔴  Momentum: removido para gf_pg/ga_pg                              ║
║  #7  🔴  Liga: removidos lg_hg_pg e lg_ag_pg                              ║
║  #8  🔴  Ranking features: removidos gf/ga/gd do loop do modelo            ║
║  #9  🔴  Diffs: removido diff_gd (goal difference)                         ║
║  #10 🔴  prev_summary: removidos prev_gf_pg/prev_ga_pg                    ║
║  #11 🔴  compute_league_means: priors baseados em xG apenas               ║
║  #12 🔴  compute_team_home_advantages: usa hxg não hg                     ║
║  #13 🔴  Removida calibração de probabilidades (usava gols)                ║
║  #14 🟡  fit_quantiles: removidos parâmetros não usados                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import logging
import os
import re
import unicodedata
import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression

# Logging seletivo — mantém warnings inesperados visíveis
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
warnings.filterwarnings("ignore", message=".*lbfgs.*")
warnings.filterwarnings("ignore", message=".*line search.*")
pd.set_option("display.float_format", "{:.4f}".format)

# ══════════════════════════════════════════════════════════════════════════════
# E3: import no topo (não dentro de função chamada N vezes)
try:
    from xgboost import XGBRegressor as _XGBRegressor
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

# CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

PASSADAS_FILE  = "passadas.xlsx"
ATUAIS_FILE    = "atuais.xlsx"
PROXIMA_FILE   = "proxima.xlsx"
OUTPUT_FILE    = "previsao_xg.xlsx"
MODEL_VERSION  = "v5_away_regime"
BENCHMARK_PREFIX_DEFAULT = "benchmark_interno"
OFFICIAL_BENCHMARK_LABEL = "official_rolling_tweedie"
CHALLENGER_BENCHMARK_LABEL = "official_away_regime_v5"

# Gates de promoção do challenger -> official
PROMOTE_MAX_DELTA_MAE_MEAN = -0.0020
PROMOTE_MAX_DELTA_MAE_AWAY = -0.0030
PROMOTE_MAX_DELTA_MAE_HOME = 0.0010
PROMOTE_MIN_DELTA_CORR_AWAY = 0.0000
PROMOTE_MIN_FOLD_WINS_AWAY = 3
PROMOTE_MIN_PHASE_WINS_AWAY = 2

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Pesos de amostra
SEASON_DECAY         = 0.82
CURRENT_SEASON_BONUS = 1.35
ATUAIS_SOURCE_BONUS  = 1.10

# Clipping do target xG
# Máximo real histórico: 6.20 | p99: 4.20 — usar 6.5 para não truncar
XG_CLIP_MIN = 0.00
XG_CLIP_MAX = 6.5

# Forma recente
RECENT_WINDOW = 5
FORM_DECAY    = 0.85
H2H_GAMES            = 6
DC_HALFLIFE_TICKS    = 10.0  # meia-vida em rodadas observadas (índice sequencial)
DC_DECAY             = float(0.5 ** (1.0 / DC_HALFLIFE_TICKS))
DC_ROLL_WINDOW_TICKS = 30    # janela móvel máxima em rodadas observadas
DC_MIN_GAMES_SNAPSHOT = 20

# Tweedie power=1.23 calibrado empiricamente via log(Var)~log(mean) por faixa
# CV observado = 0.485 (entre Poisson=1.0 e Gamma=2.0)
TWEEDIE_POWER = 1.23

# H2H shrinkage: prior_n=3 minimiza MAE out-of-sample (calibrado)
H2H_PRIOR_N = 3

# Per-team home advantage shrinkage
TEAM_HA_PRIOR_N = 8

DEFAULT_LEAGUE_MEANS = {
    "h_home_xgf_pg": 1.45, "h_home_xga_pg": 1.15,
    "h_home_pts_pg": 1.55,
    "h_home_wr": 0.45, "h_home_dr": 0.28, "h_home_lr": 0.27,
    "a_away_xgf_pg": 1.15, "a_away_xga_pg": 1.45,
    "a_away_pts_pg": 1.10,
    "a_away_wr": 0.27, "a_away_dr": 0.28, "a_away_lr": 0.45,
}

# Splits para backtesting
BACKTEST_MIN_TRAIN = 500   # jogos mínimos no primeiro fold
BACKTEST_N_FOLDS   = 5

# Shrinkage contextual — prior_n por métrica (apenas xG e resultados)
SHRINK_PRIOR = {
    "xgf_pg": 5, "xga_pg": 10,
    "pts_pg": 10, "wr": 10, "dr": 10, "lr": 10,
}

SEP  = "═" * 80
SEP2 = "─" * 80


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(s))
                   if not unicodedata.combining(c))

def norm_col(c: str) -> str:
    c = strip_accents(c).lower().strip()
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", c)).strip("_")

def clean_team(x):
    return np.nan if pd.isna(x) else re.sub(r"\s+", " ", str(x).strip())

def safe_div(a, b, default=np.nan):
    if pd.isna(b) or b == 0: return default
    return float(a) / float(b)

def pts_result(hg, ag):
    """Resultado em pontos — usa gols reais (correto: pts são definidos por gols)."""
    if hg > ag: return 3, 0
    if hg < ag: return 0, 3
    return 1, 1

def banner(t): print(f"\n{SEP}\n  {t}\n{SEP2}")
def pct(v, d=1): return f"{v*100:.{d}f}%"


# ══════════════════════════════════════════════════════════════════════════════
# CARREGAMENTO — #1 #2: SEM PROXY, xG OBRIGATÓRIO
# ══════════════════════════════════════════════════════════════════════════════

def prep_df(df: pd.DataFrame, is_future: bool = False) -> pd.DataFrame:
    """
    #1: Se hxg ou axg estiverem ausentes → ValueError. Nunca usa gols como proxy.
    #2: hxg e axg adicionados às colunas obrigatórias para dados históricos.

    Temporalidade:
    - usa a coluna `date` como eixo principal de ordenação e causalidade
    - `ano` é derivado de `date` quando não vier no arquivo
    - `rodada` passa a ser um índice interno denso por data dentro do ano,
      usado apenas para métricas de fase/pesos sazonais e compatibilidade interna
    """
    df = df.copy()
    df.columns = [norm_col(c) for c in df.columns]

    rename = {
        "mandante": "home", "visitante": "away",
        "year": "ano",      "round": "rodada",
        "data": "date",     "match_date": "date",
        "home_xg": "hxg",   "away_xg": "axg",
        "xg_home": "hxg",   "xg_away": "axg",
        "hxg_": "hxg",      "axg_": "axg",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if is_future:
        required = ["date", "home", "away"]
    else:
        required = ["date", "home", "away", "hg", "ag", "hxg", "axg"]

    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {miss}\n"
            f"  Agora a coluna temporal obrigatória é `date`.\n"
            f"  hxg e axg são OBRIGATÓRIOS — não use gols como substituto."
        )

    df["home"] = df["home"].map(clean_team)
    df["away"] = df["away"].map(clean_team)

    for c in ["ano", "hg", "ag", "hxg", "axg"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    if df["date"].isna().any():
        bad = int(df["date"].isna().sum())
        raise ValueError(
            f"Coluna `date` contém {bad} valor(es) inválido(s).\n"
            f"  Padronize as datas antes de continuar (ex.: YYYY-MM-DD ou DD/MM/YYYY)."
        )
    df["date"] = df["date"].dt.normalize()

    if "ano" not in df.columns:
        df["ano"] = df["date"].dt.year.astype(int)
    else:
        ano_from_date = df["date"].dt.year.astype(int)
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce")
        df["ano"] = df["ano"].fillna(ano_from_date).astype(int)

    if not is_future:
        for col_xg, col_name in [("hxg", "hxg (xG mandante)"), ("axg", "axg (xG visitante)")]:
            n_nan = df[col_xg].isna().sum()
            if df[col_xg].isna().all():
                raise ValueError(
                    f"Coluna '{col_xg}' está completamente vazia.\n"
                    f"  Forneça dados reais de xG. O modelo NÃO substitui por gols."
                )
            if n_nan > 0:
                raise ValueError(
                    f"Coluna '{col_xg}' tem {n_nan} valores NaN "
                    f"({n_nan/len(df)*100:.1f}% dos jogos).\n"
                    f"  Preencha os valores de xG antes de continuar.\n"
                    f"  O modelo NÃO substitui xG ausente por gols."
                )

    df["_row"] = np.arange(len(df))
    df = df.sort_values(["date", "_row"]).reset_index(drop=True)

    # índice interno por data dentro do ano, sem depender de coluna rodada no arquivo
    df["rodada"] = (
        df.groupby("ano")["date"]
          .rank(method="dense", ascending=True)
          .astype(int)
    )
    df["tick"] = df["date"].map(pd.Timestamp.toordinal).astype(int)
    return df



# ══════════════════════════════════════════════════════════════════════════════
# LEAGUE MEANS DINÂMICO — #11: priors baseados em xG apenas
# ══════════════════════════════════════════════════════════════════════════════

def compute_league_means(df: pd.DataFrame) -> dict:
    """
    Priors de shrinkage baseados apenas no histórico já observado.
    Quando df está vazio, retorna priors neutros fixos.
    """
    if df is None or len(df) == 0:
        return DEFAULT_LEAGUE_MEANS.copy()

    wr_h = float((df["hg"] > df["ag"]).mean())
    dr   = float((df["hg"] == df["ag"]).mean())
    wr_a = float((df["ag"] > df["hg"]).mean())
    lr_h = 1 - wr_h - dr
    lr_a = 1 - wr_a - dr
    xgf_h = float(df["hxg"].mean())
    xga_h = float(df["axg"].mean())
    pts_h = wr_h * 3 + dr
    pts_a = wr_a * 3 + dr
    return {
        "h_home_xgf_pg": xgf_h, "h_home_xga_pg": xga_h,
        "h_home_pts_pg": pts_h,
        "h_home_wr": wr_h, "h_home_dr": dr, "h_home_lr": lr_h,
        "a_away_xgf_pg": xga_h, "a_away_xga_pg": xgf_h,
        "a_away_pts_pg": pts_a,
        "a_away_wr": wr_a, "a_away_dr": dr, "a_away_lr": lr_a,
    }


def running_league_means(lg: dict) -> dict:
    """Priors dinâmicos usando somente jogos passados já incorporados no estado."""
    g = int(lg.get("games", 0))
    if g <= 0:
        return DEFAULT_LEAGUE_MEANS.copy()

    wr_h = safe_div(lg.get("home_wins", 0.), g, DEFAULT_LEAGUE_MEANS["h_home_wr"])
    dr   = safe_div(lg.get("draws", 0.),     g, DEFAULT_LEAGUE_MEANS["h_home_dr"])
    wr_a = safe_div(lg.get("away_wins", 0.), g, DEFAULT_LEAGUE_MEANS["a_away_wr"])
    lr_h = max(0.0, 1 - wr_h - dr)
    lr_a = max(0.0, 1 - wr_a - dr)
    xgf_h = safe_div(lg.get("hxg", 0.), g, DEFAULT_LEAGUE_MEANS["h_home_xgf_pg"])
    xga_h = safe_div(lg.get("axg", 0.), g, DEFAULT_LEAGUE_MEANS["h_home_xga_pg"])
    pts_h = wr_h * 3 + dr
    pts_a = wr_a * 3 + dr
    return {
        "h_home_xgf_pg": float(xgf_h), "h_home_xga_pg": float(xga_h),
        "h_home_pts_pg": float(pts_h),
        "h_home_wr": float(wr_h), "h_home_dr": float(dr), "h_home_lr": float(lr_h),
        "a_away_xgf_pg": float(xga_h), "a_away_xga_pg": float(xgf_h),
        "a_away_pts_pg": float(pts_a),
        "a_away_wr": float(wr_a), "a_away_dr": float(dr), "a_away_lr": float(lr_a),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FORÇA ROLLING INCREMENTAL (substitui DC from-scratch)
# ══════════════════════════════════════════════════════════════════════════════

class RollingStrengthEngine:
    """
    Motor incremental de força ataque/defesa com:
    - janela móvel por datas observadas
    - decay exponencial por meia-vida
    - snapshots estritamente causais por tick

    As métricas de força usam gols (mais próximo do espírito DC), mas o ajuste
    específico de mando por time usa xG para reduzir ruído de conversão.
    """

    def __init__(self,
                 decay: float = DC_DECAY,
                 window_ticks: int = DC_ROLL_WINDOW_TICKS,
                 min_games_snapshot: int = DC_MIN_GAMES_SNAPSHOT,
                 team_ha_prior_n: float = TEAM_HA_PRIOR_N):
        self.decay = float(decay)
        self.window_ticks = int(window_ticks)
        self.min_games_snapshot = int(min_games_snapshot)
        self.team_ha_prior_n = float(team_ha_prior_n)
        self.current_tidx = None
        self.events = deque()
        self.league = dict(games=0.0, home_goals=0.0, away_goals=0.0, home_xg=0.0, away_xg=0.0)
        self.team = defaultdict(lambda: dict(
            scored=0.0, conceded=0.0, games=0.0,
            home_xg=0.0, home_games=0.0,
            away_xg=0.0, away_games=0.0,
        ))

    def _apply_decay(self, factor: float):
        if factor == 1.0:
            return
        for k in list(self.league.keys()):
            self.league[k] *= factor
        for t in list(self.team.keys()):
            st = self.team[t]
            for k in list(st.keys()):
                st[k] *= factor
            if st["games"] < 1e-8 and st["home_games"] < 1e-8 and st["away_games"] < 1e-8:
                del self.team[t]

    def _subtract_event(self, ev: dict, weight: float):
        self.league["games"] -= weight
        self.league["home_goals"] -= weight * ev["hg"]
        self.league["away_goals"] -= weight * ev["ag"]
        self.league["home_xg"] -= weight * ev["hxg"]
        self.league["away_xg"] -= weight * ev["axg"]

        hs = self.team[ev["home"]]
        hs["scored"] -= weight * ev["hg"]
        hs["conceded"] -= weight * ev["ag"]
        hs["games"] -= weight
        hs["home_xg"] -= weight * ev["hxg"]
        hs["home_games"] -= weight

        a_s = self.team[ev["away"]]
        a_s["scored"] -= weight * ev["ag"]
        a_s["conceded"] -= weight * ev["hg"]
        a_s["games"] -= weight
        a_s["away_xg"] -= weight * ev["axg"]
        a_s["away_games"] -= weight

    def advance_to(self, tidx: int):
        tidx = int(tidx)
        if self.current_tidx is None:
            self.current_tidx = tidx
            return
        if tidx < self.current_tidx:
            raise ValueError("RollingStrengthEngine recebeu tidx não monótono.")
        delta = tidx - self.current_tidx
        if delta > 0:
            factor = float(self.decay ** delta)
            self._apply_decay(factor)
            self.current_tidx = tidx
        while self.events and (tidx - self.events[0]["tidx"] > self.window_ticks):
            ev = self.events.popleft()
            w_ev = float(self.decay ** (tidx - ev["tidx"]))
            self._subtract_event(ev, w_ev)

    def add_batch(self, grp: pd.DataFrame, tidx: int):
        tidx = int(tidx)
        if self.current_tidx is None:
            self.current_tidx = tidx
        elif tidx != self.current_tidx:
            raise ValueError("add_batch deve ocorrer no mesmo tidx corrente após advance_to().")
        for _, row in grp.iterrows():
            ev = {
                "tidx": tidx,
                "home": row["home"], "away": row["away"],
                "hg": float(row["hg"]), "ag": float(row["ag"]),
                "hxg": float(row["hxg"]), "axg": float(row["axg"]),
            }
            self.events.append(ev)
            self.league["games"] += 1.0
            self.league["home_goals"] += ev["hg"]
            self.league["away_goals"] += ev["ag"]
            self.league["home_xg"] += ev["hxg"]
            self.league["away_xg"] += ev["axg"]

            hs = self.team[ev["home"]]
            hs["scored"] += ev["hg"]
            hs["conceded"] += ev["ag"]
            hs["games"] += 1.0
            hs["home_xg"] += ev["hxg"]
            hs["home_games"] += 1.0

            a_s = self.team[ev["away"]]
            a_s["scored"] += ev["ag"]
            a_s["conceded"] += ev["hg"]
            a_s["games"] += 1.0
            a_s["away_xg"] += ev["axg"]
            a_s["away_games"] += 1.0

    def snapshot(self):
        g = float(self.league.get("games", 0.0))
        if g < self.min_games_snapshot:
            base_home = float(np.log(DEFAULT_LEAGUE_MEANS["h_home_xgf_pg"]))
            base_away = float(np.log(DEFAULT_LEAGUE_MEANS["a_away_xgf_pg"]))
            return {}, {}, base_home, base_away, {}

        eps = 1e-6
        home_pg = max(self.league["home_goals"] / g, 0.2)
        away_pg = max(self.league["away_goals"] / g, 0.2)
        home_xg_pg = max(self.league["home_xg"] / g, 0.2)
        away_xg_pg = max(self.league["away_xg"] / g, 0.2)
        base_goal_pg = max((home_pg + away_pg) / 2.0, 0.2)
        base_home = float(np.log(home_pg + eps))
        base_away = float(np.log(away_pg + eps))
        global_ha_xg = float(np.log((home_xg_pg + eps) / (away_xg_pg + eps)))

        attack_raw = {}
        defense_raw = {}
        game_weights = {}
        for team, st in self.team.items():
            gm = float(st["games"])
            if gm <= 1e-8:
                continue
            scored_pg = max(st["scored"] / gm, 0.05)
            conceded_pg = max(st["conceded"] / gm, 0.05)
            attack_raw[team] = float(np.log((scored_pg + eps) / (base_goal_pg + eps)))
            defense_raw[team] = float(np.log((base_goal_pg + eps) / (conceded_pg + eps)))
            game_weights[team] = gm

        if not attack_raw:
            return {}, {}, base_home, base_away, {}

        w_sum = max(sum(game_weights.values()), 1e-8)
        att_mean = sum(attack_raw[t] * game_weights[t] for t in attack_raw) / w_sum
        def_mean = sum(defense_raw[t] * game_weights[t] for t in defense_raw) / w_sum
        attack = {t: float(v - att_mean) for t, v in attack_raw.items()}
        defense = {t: float(v - def_mean) for t, v in defense_raw.items()}

        team_ha = {}
        for team, st in self.team.items():
            hg = float(st["home_games"])
            ag = float(st["away_games"])
            if hg <= 1e-8 or ag <= 1e-8:
                team_ha[team] = base_home
                continue
            home_xg_t = max(st["home_xg"] / hg, 0.05)
            away_xg_t = max(st["away_xg"] / ag, 0.05)
            delta = float(np.log((home_xg_t + eps) / (away_xg_t + eps)) - global_ha_xg)
            eff_n = min(hg, ag)
            w_shrink = eff_n / (eff_n + self.team_ha_prior_n)
            team_ha[team] = float(base_home + w_shrink * delta)

        return attack, defense, base_home, base_away, team_ha


def build_incremental_strength_snapshots(hist_df: pd.DataFrame, snapshot_ticks) -> tuple[dict, dict, tuple]:
    """
    Constrói snapshots causais em grade de ticks sem reotimizar do zero.
    Usa índice temporal sequencial por data observada, evitando distorção por ano*1000+rodada.

    Returns:
      dc_snapshots[tick] -> (attack_r, defense_r, base_home_log, base_away_log)
      team_ha_snapshots[tick] -> {team: abs_home_bias_log}
      final_snapshot -> mesma tupla + team_ha final após último jogo histórico
    """
    req_ticks = sorted(set(int(x) for x in snapshot_ticks))
    if hist_df is None or len(hist_df) == 0:
        base_home = float(np.log(DEFAULT_LEAGUE_MEANS["h_home_xgf_pg"]))
        base_away = float(np.log(DEFAULT_LEAGUE_MEANS["a_away_xgf_pg"]))
        dc = {t: ({}, {}, base_home, base_away) for t in req_ticks}
        tha = {t: {} for t in req_ticks}
        return dc, tha, ({}, {}, base_home, base_away, {})

    hist_sorted = hist_df.sort_values(["tick", "_row"]).reset_index(drop=True)
    hist_ticks = sorted(pd.unique(hist_sorted["tick"].astype(int)))
    all_ticks = sorted(set(hist_ticks).union(req_ticks))
    tick_to_tidx = {t: i for i, t in enumerate(all_ticks)}
    groups = {int(t): g.copy() for t, g in hist_sorted.groupby("tick", sort=True)}

    engine = RollingStrengthEngine()
    dc_snapshots = {}
    team_ha_snapshots = {}
    last_hist_tick = max(hist_ticks)
    final_snapshot = None

    for t in all_ticks:
        engine.advance_to(tick_to_tidx[t])
        att, dfe, bh, ba, tha = engine.snapshot()
        if t in req_ticks:
            dc_snapshots[int(t)] = (att, dfe, bh, ba)
            team_ha_snapshots[int(t)] = tha
        if t in groups:
            engine.add_batch(groups[t], tick_to_tidx[t])
            if int(t) == int(last_hist_tick):
                final_snapshot = engine.snapshot()

    if final_snapshot is None:
        final_snapshot = engine.snapshot()
    return dc_snapshots, team_ha_snapshots, final_snapshot


# ══════════════════════════════════════════════════════════════════════════════
# ESTADO ACUMULADO
# ══════════════════════════════════════════════════════════════════════════════

def make_block():
    # Mantém gf/ga para ranking display e pts_result — não são features ML
    return dict(games=0, gf=0., ga=0., xgf=0., xga=0.,
                pts=0., wins=0., draws=0., losses=0.)

def update_block(b, gf, ga, xgf, xga, pts):
    b["games"] += 1; b["gf"] += gf; b["ga"] += ga
    b["xgf"] += xgf; b["xga"] += xga; b["pts"] += pts
    if pts == 3:   b["wins"]   += 1
    elif pts == 1: b["draws"]  += 1
    else:          b["losses"] += 1

def default_state():
    return {"overall": make_block(), "home": make_block(), "away": make_block()}

def make_rec():
    # #5: apenas xgf, xga, pts na forma recente (sem gf, ga)
    return {k: deque(maxlen=RECENT_WINDOW) for k in ["xgf", "xga", "pts"]}

def default_recent():
    return {"overall": make_rec(), "home": make_rec(), "away": make_rec()}

def update_recent(rec, xgf, xga, pts):
    for k, v in [("xgf", xgf), ("xga", xga), ("pts", pts)]:
        rec[k].append(float(v))

def build_rank_table(s_tbl, teams, prev_summary):
    rows = []
    for team in teams:
        b = s_tbl.get(team, {}).get("overall", make_block())
        g = b["games"]
        prev = prev_summary.get(team, {})
        rows.append({
            "team": team, "games": g,
            "points": b["pts"], "gf": b["gf"], "ga": b["ga"],
            "gd": b["gf"] - b["ga"],
            "xgf": b["xgf"], "xga": b["xga"],
            "xgd": b["xgf"] - b["xga"],
            "ppg": safe_div(b["pts"], g, 0),
            "prev_ppg": prev.get("prev_ppg", np.nan),
        })
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values(
        ["points", "xgd", "xgf"], ascending=False
    ).reset_index(drop=True)  # C6: tiebreaker por xGD (não goal difference)
    df["rank"] = np.arange(1, len(df) + 1)
    df["rank_pct"] = df["rank"] / len(df)
    return df[["team", "rank", "rank_pct", "games", "points",
               "gf", "ga", "gd", "xgf", "xga", "xgd", "ppg"]]

def finalize_season(s_tbl, teams, prev_summary):
    rdf = build_rank_table(s_tbl, teams, prev_summary)
    out = {}
    for _, r in rdf.iterrows():
        out[r["team"]] = {
            "prev_rank":      int(r["rank"]),
            "prev_rank_pct":  float(r["rank_pct"]),
            "prev_ppg":       float(r["ppg"]) if pd.notna(r["ppg"]) else np.nan,
            # #10: apenas xGD e xG por jogo — removidos prev_gf_pg/prev_ga_pg
            "prev_xgd_pg":    safe_div(r["xgd"], r["games"]),
            "prev_xgf_pg":    safe_div(r["xgf"], r["games"]),
            "prev_xga_pg":    safe_div(r["xga"], r["games"]),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SHRINKAGE CONTEXTUAL — #3 #4 #5: apenas xG e resultados
# ══════════════════════════════════════════════════════════════════════════════

def _get_lm(league_means: dict, side, ctx, metric):
    """Busca prior de shrinkage no dicionário passado explicitamente (não global)."""
    return league_means.get(f"{side}_{ctx}_{metric}", 0.0)

def _shrink(val, n_ctx, lm, prior_n):
    if pd.isna(val) or n_ctx == 0: return lm
    w = n_ctx / (n_ctx + prior_n)
    return float(w * val + (1 - w) * lm)

def shrunk_block_feats(pfx, block, side, ctx, league_means: dict):
    """
    Features de bloco com shrinkage — apenas xG e resultados.
    league_means passado explicitamente (sem estado global).
    """
    g = block["games"]
    def s(m, v):
        return _shrink(
            v if pd.notna(v) else 0, g,
            _get_lm(league_means, side, ctx, m), SHRINK_PRIOR.get(m, 8)
        )
    xgf = safe_div(block["xgf"], g); xga = safe_div(block["xga"], g)
    pts = safe_div(block["pts"],  g)
    wr  = safe_div(block["wins"],   g)
    dr  = safe_div(block["draws"],  g)
    lr  = safe_div(block["losses"], g)
    return {
        f"{pfx}_g":      g,
        f"{pfx}_xgf_pg": s("xgf_pg", xgf if pd.notna(xgf) else 0),
        f"{pfx}_xga_pg": s("xga_pg", xga if pd.notna(xga) else 0),
        f"{pfx}_pts_pg": s("pts_pg", pts if pd.notna(pts) else 0),
        f"{pfx}_wr":     s("wr",     wr  if pd.notna(wr)  else 0),
        f"{pfx}_dr":     s("dr",     dr  if pd.notna(dr)  else 0),
        f"{pfx}_lr":     s("lr",     lr  if pd.notna(lr)  else 0),
    }

def shrunk_recent_feats(pfx, rec, side, ctx, league_means: dict):
    """Forma recente com shrinkage — apenas xG e pontos."""
    def wm(q):
        n = len(q)
        if n == 0: return np.nan
        w = np.array([FORM_DECAY ** (n-1-i) for i in range(n)])
        return float(np.average(list(q), weights=w))
    n = len(rec["xgf"])
    xgf = wm(rec["xgf"]); xga = wm(rec["xga"]); pts = wm(rec["pts"])
    def s(m, v):
        return _shrink(
            v if pd.notna(v) else 0, n,
            _get_lm(league_means, side, ctx, m), SHRINK_PRIOR.get(m, 8)
        )
    return {
        f"{pfx}_xgf": s("xgf_pg", xgf if pd.notna(xgf) else 0),
        f"{pfx}_xga": s("xga_pg", xga if pd.notna(xga) else 0),
        f"{pfx}_pts": s("pts_pg", pts if pd.notna(pts) else 0),
        f"{pfx}_n":   float(n),
    }


# ══════════════════════════════════════════════════════════════════════════════
# HEAD-TO-HEAD — xG com shrinkage bayesiano
# ══════════════════════════════════════════════════════════════════════════════

def _h2h_key(a, b):
    return (min(a, b), max(a, b))

def get_h2h_fast(h2h_hist: dict, home: str, away: str, league_means: dict) -> dict:
    """
    H2H com shrinkage bayesiano.
    C2 CORRIGIDO: win rate por resultado real (hg/ag), não por xG.
    xG médio H2H ainda usa hxg/axg (correto — mede qualidade de chance histórica).
    league_means passado explicitamente (sem estado global).
    """
    null = {
        "h2h_h_xgf": np.nan, "h2h_a_xgf": np.nan,
        "h2h_h_wr": np.nan, "h2h_dr": np.nan, "h2h_a_wr": np.nan,
        "h2h_n": 0.
    }
    key = _h2h_key(home, away)
    entries = list(h2h_hist.get(key, []))
    if not entries: return null
    hw = dr = aw = 0
    hxgf_sum = axgf_sum = 0.
    for e in entries:
        if e["home"] == home:
            hxgf_sum += e["hxg"]; axgf_sum += e["axg"]
            # C2: win rate pelo placar real, não por quem criou mais xG
            if   e["hg"] > e["ag"]: hw += 1
            elif e["hg"] < e["ag"]: aw += 1
            else:                   dr += 1
        else:
            hxgf_sum += e["axg"]; axgf_sum += e["hxg"]
            if   e["ag"] > e["hg"]: hw += 1
            elif e["ag"] < e["hg"]: aw += 1
            else:                   dr += 1
    n = len(entries)
    lm_h = league_means.get("h_home_xgf_pg", 1.6)
    lm_a = league_means.get("a_away_xgf_pg", 1.2)
    w = n / (n + H2H_PRIOR_N)
    return {
        "h2h_h_xgf": w * (hxgf_sum / n) + (1-w) * lm_h,
        "h2h_a_xgf": w * (axgf_sum / n) + (1-w) * lm_a,
        "h2h_h_wr":  hw / n,
        "h2h_dr":    dr / n,
        "h2h_a_wr":  aw / n,
        "h2h_n":     float(n),
    }

def update_h2h_hist(h2h_hist, home, away, hg, ag, hxg, axg):
    """C2: armazena hg/ag (win rate real) e hxg/axg (xG médio H2H)."""
    key = _h2h_key(home, away)
    if key not in h2h_hist:
        h2h_hist[key] = deque(maxlen=H2H_GAMES)
    h2h_hist[key].append({"home": home, "away": away,
                           "hg": hg, "ag": ag, "hxg": hxg, "axg": axg})


# ══════════════════════════════════════════════════════════════════════════════
# FEATURES POR JOGO — #6 #7 #8 #9 #10: apenas xG, pontos, DC
# ══════════════════════════════════════════════════════════════════════════════

def make_match_features(home, away, ano, rodada, tick,
                        career, season, recent,
                        season_tbl, teams_by_year, prev_summary, lg,
                        h2h_hist, attack_r, defense_r, base_home_log, base_away_log, team_ha,
                        league_means: dict):
    f = {
        "ano":      float(ano),
        "rodada":   float(rodada),
        "is_early": 1. if rodada <= 5 else 0.,
    }

    # Blocos shrinkados — career + season + recent (xG e resultados apenas)
    f.update(shrunk_block_feats("hch_s", career[home]["home"], "h", "home", league_means))
    f.update(shrunk_block_feats("acw_s", career[away]["away"], "a", "away", league_means))
    f.update(shrunk_block_feats("hsh_s", season[home]["home"], "h", "home", league_means))
    f.update(shrunk_block_feats("asw_s", season[away]["away"], "a", "away", league_means))
    f.update(shrunk_recent_feats("hrh_s", recent[home]["home"], "h", "home", league_means))
    f.update(shrunk_recent_feats("arw_s", recent[away]["away"], "a", "away", league_means))

    # #6: Momentum apenas para xG e pontos (sem gols)
    for metric in ("xgf_pg", "xga_pg", "pts_pg"):
        for side, rp, sp in [("h", "hrh_s", "hsh_s"), ("a", "arw_s", "asw_s")]:
            v_rec = f.get(f"{rp}_{metric.replace('_pg', '')}", np.nan)
            v_szn = f.get(f"{sp}_{metric}", np.nan)
            f[f"mom_{side}_{metric}"] = (
                v_rec - v_szn
                if pd.notna(v_rec) and pd.notna(v_szn)
                else np.nan
            )

    # Ranking — #8: apenas colunas xG-based como features ML
    curr_df = build_rank_table(season_tbl, teams_by_year.get(ano, []), prev_summary)
    cr_map  = curr_df.set_index("team").to_dict("index") if not curr_df.empty else {}
    hr = cr_map.get(home, {}); ar = cr_map.get(away, {})
    # #8: removidos gf/ga/gd do loop de features
    for k in ["rank", "rank_pct", "games", "points", "xgf", "xga", "xgd", "ppg"]:
        f[f"hcr_{k}"] = hr.get(k, np.nan)
        f[f"acr_{k}"] = ar.get(k, np.nan)

    hp = prev_summary.get(home, {}); ap = prev_summary.get(away, {})
    # #10: apenas xG e ppg da temporada anterior
    for k in ["prev_rank", "prev_rank_pct", "prev_ppg", "prev_xgd_pg"]:
        f[f"h_{k}"] = hp.get(k, np.nan)
        f[f"a_{k}"] = ap.get(k, np.nan)

    # Liga — #7: apenas xG da liga (sem hg/ag médias)
    lg_n = lg["games"]
    f.update({
        "lg_hxg_pg": safe_div(lg["hxg"], lg_n, 1.642),
        "lg_axg_pg": safe_div(lg["axg"], lg_n, 1.310),
    })

    # DC como features (ratings de força)
    att_h = attack_r.get(home, 0.); def_h = defense_r.get(home, 0.)
    att_a = attack_r.get(away, 0.); def_a = defense_r.get(away, 0.)
    ha_home = team_ha.get(home, base_home_log)
    dc_h = float(np.exp(att_h - def_a + ha_home))
    dc_a = float(np.exp(att_a - def_h + base_away_log))
    f.update({
        "att_h": att_h, "def_h": def_h, "att_a": att_a, "def_a": def_a,
        "dc_lam_h": dc_h, "dc_lam_a": dc_a,
        "dc_lam_diff": dc_h - dc_a,
        "home_adv_h": ha_home,
        "home_adv_global": base_home_log - base_away_log,
        "dc_base_home": float(np.exp(base_home_log)),
        "dc_base_away": float(np.exp(base_away_log)),
    })

    # H2H
    f.update(get_h2h_fast(h2h_hist, home, away, league_means))

    # Diffs — #9: removido diff_gd
    diffs = [
        ("rank",       "hcr_rank",       "acr_rank"),
        ("ppg",        "hcr_ppg",        "acr_ppg"),
        ("xgd",        "hcr_xgd",        "acr_xgd"),
        ("dc",         "dc_lam_h",       "dc_lam_a"),
        ("prev_ppg",   "h_prev_ppg",     "a_prev_ppg"),
        ("shrunk_xgf", "hsh_s_xgf_pg",   "asw_s_xgf_pg"),
        ("shrunk_xga", "hsh_s_xga_pg",   "asw_s_xga_pg"),
        ("shrunk_pts", "hsh_s_pts_pg",   "asw_s_pts_pg"),
        ("mom_xgf",    "mom_h_xgf_pg",   "mom_a_xgf_pg"),
    ]
    for out, hk, ak in diffs:
        hv = f.get(hk, np.nan); av = f.get(ak, np.nan)
        f[f"diff_{out}"] = hv - av if pd.notna(hv) and pd.notna(av) else np.nan

    # Interação xG ataque × xGA defesa adversária
    h_xgf = f.get("hsh_s_xgf_pg", 1.)
    a_xga = max(f.get("asw_s_xga_pg", 1.) or 1., 0.5)
    a_xgf = f.get("asw_s_xgf_pg", 1.)
    h_xga = max(f.get("hsh_s_xga_pg", 1.) or 1., 0.5)
    f["xg_h_vs_def_a"] = np.clip((h_xgf or 0) / a_xga, 0, 5.)
    f["xg_a_vs_def_h"] = np.clip((a_xgf or 0) / h_xga, 0, 5.)
    return f


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUÇÃO DO DATASET
# ══════════════════════════════════════════════════════════════════════════════

def build_datasets(hist_df, next_df, dc_snapshots, team_ha_snapshots,
                   attack_r_final, defense_r_final, base_home_final, base_away_final, team_ha_final):
    hist_df = hist_df.sort_values(["date","_row"]).reset_index(drop=True)
    next_df = next_df.sort_values(["date","_row"]).reset_index(drop=True)

    teams_by_year = defaultdict(set)
    for df in [hist_df, next_df]:
        for _, r in df.iterrows():
            teams_by_year[int(r["ano"])].add(r["home"])
            teams_by_year[int(r["ano"])] .add(r["away"])
    teams_by_year = {k: sorted(v) for k, v in teams_by_year.items()}

    career  = defaultdict(default_state)
    season  = defaultdict(default_state)
    recent  = defaultdict(default_recent)
    s_table = defaultdict(default_state)
    prev_summary = {}
    lg = dict(games=0, hxg=0., axg=0., home_wins=0., away_wins=0., draws=0.)
    h2h_hist = {}
    cur_year = None; cur_teams = []; rows = []

    def roll_year(ny):
        nonlocal cur_year, season, s_table, prev_summary, cur_teams
        if cur_year is None:
            cur_year = ny; cur_teams = teams_by_year.get(ny, []); return
        if ny != cur_year:
            prev_summary = finalize_season(s_table, cur_teams, prev_summary)
            season  = defaultdict(default_state)
            s_table = defaultdict(default_state)
            cur_year = ny; cur_teams = teams_by_year.get(ny, [])

    print("  Iterando histórico cronologicamente por data (sem leakage intra-data)...")
    for (ano, match_date), grp in hist_df.groupby(["ano", "date"], sort=True):
        rod = int(grp["rodada"].iloc[0])
        ano = int(ano); rod = int(rod)
        roll_year(ano)
        league_means_cur = running_league_means(lg)
        tick_snapshot = int(grp["tick"].iloc[0])
        att_r, def_r, base_h, base_a = dc_snapshots.get(tick_snapshot, ({}, {}, float(np.log(DEFAULT_LEAGUE_MEANS['h_home_xgf_pg'])), float(np.log(DEFAULT_LEAGUE_MEANS['a_away_xgf_pg']))))
        team_ha_cur = team_ha_snapshots.get(tick_snapshot, {})

        batch_rows = []
        batch_updates = []
        for _, row in grp.iterrows():
            tick = int(row["tick"])
            home = row["home"]; away = row["away"]
            feats = make_match_features(
                home, away, ano, rod, tick,
                career, season, recent, s_table,
                teams_by_year, prev_summary, lg,
                h2h_hist, att_r, def_r, base_h, base_a, team_ha_cur,
                league_means_cur
            )
            feats.update({
                "date": pd.Timestamp(row["date"]).normalize(),
                "home": home, "away": away,
                "hg":  float(row["hg"]),  "ag":  float(row["ag"]),
                "hxg": float(row["hxg"]), "axg": float(row["axg"]),
                "dataset_source": str(row.get("dataset_source", "hist")),
            })
            batch_rows.append(feats)
            batch_updates.append(row)

        rows.extend(batch_rows)

        for row in batch_updates:
            home = row["home"]; away = row["away"]
            hg  = float(row["hg"]);  ag  = float(row["ag"])
            hxg = float(row["hxg"]); axg = float(row["axg"])
            ph, pa = pts_result(hg, ag)

            for st in [career, season, s_table]:
                update_block(st[home]["overall"], hg, ag, hxg, axg, ph)
                update_block(st[home]["home"],    hg, ag, hxg, axg, ph)
                update_block(st[away]["overall"], ag, hg, axg, hxg, pa)
                update_block(st[away]["away"],    ag, hg, axg, hxg, pa)

            update_recent(recent[home]["overall"], hxg, axg, ph)
            update_recent(recent[home]["home"],    hxg, axg, ph)
            update_recent(recent[away]["overall"], axg, hxg, pa)
            update_recent(recent[away]["away"],    axg, hxg, pa)

            lg["games"] += 1
            lg["hxg"] += hxg
            lg["axg"] += axg
            if hg > ag:
                lg["home_wins"] += 1
            elif hg < ag:
                lg["away_wins"] += 1
            else:
                lg["draws"] += 1

            update_h2h_hist(h2h_hist, home, away, hg, ag, hxg, axg)

        if len(rows) % 300 == 0:
            print(f"    [{len(rows)}/{len(hist_df)}] processados...")

    hist_feats = pd.DataFrame(rows)
    if "date" in hist_feats.columns:
        hist_feats["date"] = pd.to_datetime(hist_feats["date"], errors="coerce").dt.normalize()

    fut_year = cur_year; fut_tbl = s_table; fut_szn = season
    fut_prev = prev_summary; fut_teams = cur_teams
    next_rows = []
    league_means_final = running_league_means(lg)
    for (ano, match_date), grp in next_df.groupby(["ano", "date"], sort=True):
        rod = int(grp["rodada"].iloc[0])
        ano = int(ano); rod = int(rod)
        if fut_year is None:
            fut_year = ano; fut_teams = teams_by_year.get(ano, [])
        elif ano != fut_year:
            fut_prev = finalize_season(fut_tbl, fut_teams, fut_prev)
            fut_tbl  = defaultdict(default_state)
            fut_szn  = defaultdict(default_state)
            fut_year = ano; fut_teams = teams_by_year.get(ano, [])
        for _, row in grp.iterrows():
            tick = int(row["tick"])
            home = row["home"]; away = row["away"]
            att_f, def_f, base_h_f, base_a_f = dc_snapshots.get(tick, (attack_r_final, defense_r_final, base_home_final, base_away_final))
            team_ha_f = team_ha_snapshots.get(tick, team_ha_final)
            feats = make_match_features(
                home, away, ano, rod, tick,
                career, fut_szn, recent, fut_tbl,
                teams_by_year, fut_prev, lg,
                h2h_hist, att_f, def_f, base_h_f, base_a_f, team_ha_f,
                league_means_final
            )
            feats.update({"date": pd.Timestamp(row["date"]).normalize(), "home": home, "away": away})
            next_rows.append(feats)

    next_feats = pd.DataFrame(next_rows)
    if "date" in next_feats.columns:
        next_feats["date"] = pd.to_datetime(next_feats["date"], errors="coerce").dt.normalize()
    ranking = (
        build_rank_table(s_table, cur_teams, prev_summary)
        if cur_year else pd.DataFrame()
    )
    return hist_feats, next_feats, ranking


# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def sample_weights(df: pd.DataFrame) -> np.ndarray:
    years  = df["ano"].astype(float).values
    rounds = df["rodada"].astype(float).values
    max_yr = np.nanmax(years)
    # Decay entre temporadas
    sw = np.power(SEASON_DECAY, max_yr - years)
    sw = np.where(years == max_yr, sw * CURRENT_SEASON_BONUS, sw)
    # Bônus por posição na temporada [0.85, 1.15]
    max_rnd_per_year = df.groupby("ano")["rodada"].transform("max").values
    season_pos = rounds / np.maximum(max_rnd_per_year, 1)
    rndw = 0.85 + 0.30 * season_pos
    # Penalidade de fonte
    srcw = (
        np.where(df["dataset_source"].astype(str).values == "atuais",
                 ATUAIS_SOURCE_BONUS, 1.)
        if "dataset_source" in df.columns else np.ones(len(df))
    )
    return sw * srcw * rndw


# ══════════════════════════════════════════════════════════════════════════════
# MODELOS xG — Tweedie (pontual) + Quantile (IC 80%)
# ══════════════════════════════════════════════════════════════════════════════

def get_xg_model(quantile: float | None = None):
    """
    Tweedie power=1.23 para estimativa pontual (calibrado: CV≈0.485).
    reg:quantileerror para IC 80% (q10 e q90).
    E3: usa _XGBRegressor importado no topo do módulo (não reimporta em cada chamada).
    """
    if _HAS_XGBOOST:
        if quantile is not None:
            return _XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=float(quantile),
                n_estimators=500, learning_rate=0.04, max_depth=4,
                min_child_weight=3, subsample=0.80, colsample_bytree=0.75,
                reg_alpha=0.10, reg_lambda=1.50,
                random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
            ), f"xgb_q{int(quantile*100)}"
        else:
            return _XGBRegressor(
                objective="reg:tweedie",
                tweedie_variance_power=TWEEDIE_POWER,
                n_estimators=500, learning_rate=0.04, max_depth=4,
                min_child_weight=3, subsample=0.80, colsample_bytree=0.75,
                reg_alpha=0.10, reg_lambda=1.50, gamma=0.05,
                random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
            ), "xgb_tweedie"
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, max_depth=5,
            max_iter=300, min_samples_leaf=10, l2_regularization=0.1,
            random_state=RANDOM_STATE,
        ), "histgb"


# ══════════════════════════════════════════════════════════════════════════════
# V4 — CHALLENGER ASSIMÉTRICO PARA xG VISITANTE
# ══════════════════════════════════════════════════════════════════════════════

def augment_dual_away_v4_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features adicionais para reduzir subestimação do visitante sem quebrar causalidade."""
    out = df.copy()
    idx = out.index

    def s(col: str, default: float = np.nan) -> pd.Series:
        if col in out.columns:
            return out[col].astype(float)
        return pd.Series(default, index=idx, dtype=float)

    eps = 0.05
    dc_h = s("dc_lam_h", 1.45)
    dc_a = s("dc_lam_a", 1.15)
    ctx_h = s("xg_h_vs_def_a", dc_h)
    ctx_a = s("xg_a_vs_def_h", dc_a)
    lg_h = s("lg_hxg_pg", 1.40)
    lg_a = s("lg_axg_pg", 1.10)
    asw_xgf = s("asw_s_xgf_pg", ctx_a)
    asw_xga = s("asw_s_xga_pg", lg_h)
    hsh_xga = s("hsh_s_xga_pg", lg_a)
    hsh_xgf = s("hsh_s_xgf_pg", ctx_h)
    arw_xgf = s("arw_s_xgf", np.nan) / np.clip(s("arw_s_n", 1.0), 1.0, None)
    hrh_xga = s("hrh_s_xga", np.nan) / np.clip(s("hrh_s_n", 1.0), 1.0, None)
    mom_axgf = s("mom_a_xgf_pg", ctx_a)
    mom_hxga = s("mom_h_xga_pg", lg_a)
    rank_edge = s("hcr_rank_pct", 0.50) - s("acr_rank_pct", 0.50)
    xgd_edge = s("acr_xgd", 0.0) - s("hcr_xgd", 0.0)
    diff_dc = s("diff_dc", 0.0)
    diff_xga = s("diff_shrunk_xga", 0.0)

    out["v4_dc_sum"] = dc_h + dc_a
    out["v4_dc_gap_a_h"] = dc_a - dc_h
    out["v4_dc_ratio_a_h"] = dc_a / np.clip(dc_h, eps, None)
    out["v4_ctx_sum_xg"] = ctx_h + ctx_a
    out["v4_ctx_gap_a_h"] = ctx_a - ctx_h
    out["v4_ctx_ratio_a_dc"] = ctx_a / np.clip(dc_a, eps, None)
    out["v4_away_shape_edge"] = asw_xgf - hsh_xga
    out["v4_away_recent_edge"] = arw_xgf - hrh_xga
    out["v4_away_form_edge"] = mom_axgf - mom_hxga
    out["v4_open_game_index"] = (dc_h + dc_a) * (lg_h + lg_a) / 2.0
    out["v4_away_upside_gate"] = np.maximum(ctx_a - lg_a, 0.0) * np.maximum(dc_h - lg_h, 0.0)
    out["v4_away_transition_edge"] = 0.55 * (ctx_a - hsh_xgf) + 0.45 * (asw_xgf - asw_xga)
    out["v4_away_quality_edge"] = 0.60 * rank_edge + 0.40 * np.tanh(xgd_edge / 8.0)
    out["v4_away_pressure_relief"] = np.maximum(diff_xga, 0.0) * np.maximum(-diff_dc, 0.0)
    out["v4_away_blend_core"] = 0.45 * ctx_a + 0.35 * dc_a + 0.20 * asw_xgf
    return out


def prepare_dual_away_v4_frames(hist_f: pd.DataFrame, next_f: pd.DataFrame | None = None):
    hist_aug = augment_dual_away_v4_features(hist_f)
    next_aug = augment_dual_away_v4_features(next_f) if next_f is not None else None
    feat_cols_home = get_feature_columns(hist_f)
    feat_cols_away = get_feature_columns(hist_aug)
    return hist_aug, next_aug, feat_cols_home, feat_cols_away


def sample_weights_side_v4(df: pd.DataFrame, side: str = "home") -> np.ndarray:
    w = sample_weights(df).astype(float)
    if side != "away":
        return w
    y = df["axg"].astype(float).to_numpy()
    q70 = float(np.nanquantile(y, 0.70))
    q85 = float(np.nanquantile(y, 0.85))
    hi = np.where(y >= q70, 1.18, 1.0) * np.where(y >= q85, 1.10, 1.0)

    open_idx = df.get("v4_open_game_index", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
    open_thr = float(np.nanmedian(open_idx)) if np.isfinite(open_idx).any() else np.nan
    open_boost = np.where(np.isfinite(open_thr) & (open_idx >= open_thr), 1.04, 1.0)

    upside = df.get("v4_away_upside_gate", pd.Series(0.0, index=df.index)).astype(float).to_numpy()
    up_thr = float(np.nanquantile(upside, 0.75)) if np.isfinite(upside).any() else np.nan
    up_boost = np.where(np.isfinite(up_thr) & (upside >= up_thr), 1.05, 1.0)
    return w * hi * open_boost * up_boost


def get_xg_model_side_v4(side: str = "home"):
    if side != "away":
        return get_xg_model()
    if _HAS_XGBOOST:
        return _XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=TWEEDIE_POWER,
            n_estimators=650, learning_rate=0.035, max_depth=5,
            min_child_weight=2.5, subsample=0.86, colsample_bytree=0.86,
            reg_alpha=0.06, reg_lambda=1.20, gamma=0.03,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        ), "xgb_tweedie_away_v4"
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.045, max_depth=6,
        max_iter=360, min_samples_leaf=8, l2_regularization=0.08,
        random_state=RANDOM_STATE,
    ), "histgb_away_v4"


def run_dual_away_v4_backtest_detailed(hist_f: pd.DataFrame, n_folds: int = 5, min_train: int = 500,
                                      benchmark_label: str = CHALLENGER_BENCHMARK_LABEL):
    hist_aug, _, feat_cols_home, feat_cols_away = prepare_dual_away_v4_frames(hist_f, None)
    _, plan = build_backtest_plan(hist_aug, n_folds=n_folds, min_train=min_train)
    results = []
    oof_rows = []
    print(f"\n  Walk-forward challenger away-gated ({len(plan)} folds):")
    for spec in plan:
        tr = hist_aug.iloc[spec["train_idx"]]
        te = hist_aug.iloc[spec["test_idx"]]

        Xh_tr = tr[feat_cols_home].fillna(tr[feat_cols_home].median())
        Xh_te = te[feat_cols_home].fillna(Xh_tr.median())
        Xa_tr = tr[feat_cols_away].fillna(tr[feat_cols_away].median())
        Xa_te = te[feat_cols_away].fillna(Xa_tr.median())

        y_h = tr["hxg"].clip(lower=XG_CLIP_MIN, upper=XG_CLIP_MAX)
        y_a = tr["axg"].clip(lower=XG_CLIP_MIN, upper=XG_CLIP_MAX)
        w_h = sample_weights_side_v4(tr, "home")
        w_a = sample_weights_side_v4(tr, "away")

        mh, _ = get_xg_model_side_v4("home")
        ma, _ = get_xg_model_side_v4("away")
        mh.fit(Xh_tr, y_h, sample_weight=w_h)
        ma.fit(Xa_tr, y_a, sample_weight=w_a)

        pred_h = np.clip(mh.predict(Xh_te), XG_CLIP_MIN, XG_CLIP_MAX)
        pred_a = np.clip(ma.predict(Xa_te), XG_CLIP_MIN, XG_CLIP_MAX)
        real_h = te["hxg"].to_numpy(dtype=float)
        real_a = te["axg"].to_numpy(dtype=float)
        metrics = _metrics_from_predictions(real_h, pred_h, real_a, pred_a)

        results.append({
            "benchmark_model": benchmark_label,
            "fold": spec["fold"],
            "n_train": len(tr),
            "n_test": len(te),
            **{k: round(v, 4) if pd.notna(v) else np.nan for k, v in metrics.items()},
            "test_date_start": pd.Timestamp.fromordinal(int(spec["test_keys"].min())).strftime("%Y-%m-%d"),
            "test_date_end": pd.Timestamp.fromordinal(int(spec["test_keys"].max())).strftime("%Y-%m-%d"),
        })
        oof_rows.append(pd.DataFrame({
            "benchmark_model": benchmark_label,
            "fold": spec["fold"],
            "row_idx": spec["test_idx"],
            "pred_home": pred_h,
            "real_home": real_h,
            "pred_away": pred_a,
            "real_away": real_a,
        }))
        print(
            f"    fold {spec['fold']}: MAE_h={metrics['mae_home']:.4f} "
            f"MAE_a={metrics['mae_away']:.4f} "
            f"bias_h={metrics['bias_home']:+.4f} bias_a={metrics['bias_away']:+.4f}"
        )

    df_bt = pd.DataFrame(results)
    oof_df = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    if not df_bt.empty:
        print(
            f"  → Challenger MAE médio: {df_bt['mae_mean'].mean():.4f} | "
            f"Bias_h: {df_bt['bias_home'].mean():+.4f} | "
            f"Bias_a: {df_bt['bias_away'].mean():+.4f}"
        )
    return df_bt, oof_df, {"home": feat_cols_home, "away": feat_cols_away}


def evaluate_challenger_promotion(summary_df: pd.DataFrame, compare_df: pd.DataFrame,
                                  fold_df: pd.DataFrame | None = None, phase_df: pd.DataFrame | None = None,
                                  official_label: str = OFFICIAL_BENCHMARK_LABEL,
                                  challenger_label: str = CHALLENGER_BENCHMARK_LABEL) -> dict:
    decision = {
        "challenger_label": challenger_label,
        "official_label": official_label,
        "promoted": False,
        "reason": "dados_insuficientes",
    }
    if summary_df is None or compare_df is None or summary_df.empty or compare_df.empty:
        return decision
    if official_label not in set(summary_df["benchmark_model"]) or challenger_label not in set(summary_df["benchmark_model"]):
        return decision

    cmp_row = compare_df.loc[compare_df["benchmark_model"] == challenger_label]
    if cmp_row.empty:
        return decision
    cmp_row = cmp_row.iloc[0]

    delta_mean = float(cmp_row.get(f"delta_vs_{official_label}_mae_mean", np.nan))
    delta_home = float(cmp_row.get(f"delta_vs_{official_label}_mae_home", np.nan))
    delta_away = float(cmp_row.get(f"delta_vs_{official_label}_mae_away", np.nan))
    delta_corr_away = float(cmp_row.get(f"delta_vs_{official_label}_corr_away", np.nan))

    wins_away = wins_mean = 0
    if fold_df is not None and not fold_df.empty:
        pivot_away = fold_df.pivot(index="fold", columns="benchmark_model", values="mae_away")
        pivot_mean = fold_df.pivot(index="fold", columns="benchmark_model", values="mae_mean")
        if official_label in pivot_away.columns and challenger_label in pivot_away.columns:
            wins_away = int((pivot_away[challenger_label] < pivot_away[official_label]).sum())
        if official_label in pivot_mean.columns and challenger_label in pivot_mean.columns:
            wins_mean = int((pivot_mean[challenger_label] < pivot_mean[official_label]).sum())

    phase_wins_away = 0
    if phase_df is not None and not phase_df.empty:
        pvt = phase_df.pivot(index="phase", columns="benchmark_model", values="mae_away")
        if official_label in pvt.columns and challenger_label in pvt.columns:
            phase_wins_away = int((pvt[challenger_label] < pvt[official_label]).sum())

    rules = {
        "mae_mean_gate": bool(np.isfinite(delta_mean) and delta_mean <= PROMOTE_MAX_DELTA_MAE_MEAN),
        "mae_away_gate": bool(np.isfinite(delta_away) and delta_away <= PROMOTE_MAX_DELTA_MAE_AWAY),
        "mae_home_guard": bool(np.isfinite(delta_home) and delta_home <= PROMOTE_MAX_DELTA_MAE_HOME),
        "corr_away_guard": bool(np.isfinite(delta_corr_away) and delta_corr_away >= PROMOTE_MIN_DELTA_CORR_AWAY),
        "fold_wins_away_guard": wins_away >= PROMOTE_MIN_FOLD_WINS_AWAY,
        "phase_wins_away_guard": phase_wins_away >= PROMOTE_MIN_PHASE_WINS_AWAY,
    }
    promoted = all(rules.values())
    decision.update({
        "promoted": promoted,
        "reason": "promovido" if promoted else "gates_nao_satisfeitos",
        "delta_mae_mean": round(delta_mean, 4) if np.isfinite(delta_mean) else np.nan,
        "delta_mae_home": round(delta_home, 4) if np.isfinite(delta_home) else np.nan,
        "delta_mae_away": round(delta_away, 4) if np.isfinite(delta_away) else np.nan,
        "delta_corr_away": round(delta_corr_away, 4) if np.isfinite(delta_corr_away) else np.nan,
        "fold_wins_away": int(wins_away),
        "fold_wins_mean": int(wins_mean),
        "phase_wins_away": int(phase_wins_away),
        **rules,
    })
    return decision


# ══════════════════════════════════════════════════════════════════════════════
# BACKTESTING WALK-FORWARD xG
# ══════════════════════════════════════════════════════════════════════════════



# ══════════════════════════════════════════════════════════════════════════════
# V5 — CHALLENGER ESTRUTURAL PARA xG VISITANTE (BASE + RESIDUAL DE REGIME)
# ══════════════════════════════════════════════════════════════════════════════

def augment_away_regime_v5_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index
    def s(col: str, default: float = np.nan) -> pd.Series:
        if col in out.columns:
            return out[col].astype(float)
        return pd.Series(default, index=idx, dtype=float)

    eps = 0.05
    dc_h = s("dc_lam_h", 1.45)
    dc_a = s("dc_lam_a", 1.15)
    ctx_h = s("xg_h_vs_def_a", dc_h)
    ctx_a = s("xg_a_vs_def_h", dc_a)
    lg_h = s("lg_hxg_pg", 1.40)
    lg_a = s("lg_axg_pg", 1.10)
    asw_xgf = s("asw_s_xgf_pg", ctx_a)
    asw_xga = s("asw_s_xga_pg", lg_h)
    hsh_xga = s("hsh_s_xga_pg", lg_a)
    hsh_xgf = s("hsh_s_xgf_pg", ctx_h)
    arw_xgf = s("arw_s_xgf", np.nan) / np.clip(s("arw_s_n", 1.0), 1.0, None)
    hrh_xga = s("hrh_s_xga", np.nan) / np.clip(s("hrh_s_n", 1.0), 1.0, None)
    mom_axgf = s("mom_a_xgf_pg", ctx_a)
    mom_hxga = s("mom_h_xga_pg", lg_a)
    rank_home = s("hcr_rank_pct", 0.50)
    rank_away = s("acr_rank_pct", 0.50)
    xgd_home = s("hcr_xgd", 0.0)
    xgd_away = s("acr_xgd", 0.0)
    diff_dc = s("diff_dc", 0.0)
    diff_xga = s("diff_shrunk_xga", 0.0)

    out["v5_open_game_index"] = 0.55 * (dc_h + dc_a) + 0.45 * (lg_h + lg_a)
    out["v5_home_strength_index"] = 0.55 * rank_home + 0.45 * np.tanh(xgd_home / 8.0)
    out["v5_away_strength_index"] = 0.55 * rank_away + 0.45 * np.tanh(xgd_away / 8.0)
    out["v5_away_transition_edge"] = 0.55 * (ctx_a - hsh_xgf) + 0.45 * (asw_xgf - asw_xga)
    out["v5_away_counter_window"] = 0.60 * (asw_xgf - hsh_xga) + 0.40 * (arw_xgf - hrh_xga)
    out["v5_away_form_edge"] = mom_axgf - mom_hxga
    out["v5_away_dc_edge"] = dc_a - dc_h
    out["v5_away_ctx_edge"] = ctx_a - ctx_h
    out["v5_away_ratio_ctx_dc"] = ctx_a / np.clip(dc_a, eps, None)
    out["v5_away_pressure_relief"] = np.maximum(diff_xga, 0.0) * np.maximum(-diff_dc, 0.0)
    out["v5_away_upside_gate"] = np.maximum(ctx_a - lg_a, 0.0) * np.maximum(dc_h - lg_h, 0.0)
    out["v5_away_base_anchor"] = 0.40 * ctx_a + 0.35 * dc_a + 0.25 * asw_xgf
    out["v5_away_regime_score"] = (
        0.32 * out["v5_open_game_index"]
        + 0.24 * out["v5_away_transition_edge"]
        + 0.18 * out["v5_away_counter_window"]
        + 0.14 * out["v5_away_form_edge"]
        - 0.24 * out["v5_home_strength_index"]
        + 0.12 * out["v5_away_strength_index"]
        + 0.08 * out["v5_away_pressure_relief"]
    )
    return out


def prepare_away_regime_v5_frames(hist_f: pd.DataFrame, next_f: pd.DataFrame | None = None):
    hist_aug = augment_away_regime_v5_features(hist_f)
    next_aug = augment_away_regime_v5_features(next_f) if next_f is not None else None
    feat_cols_home = get_feature_columns(hist_f)
    feat_cols_away_base = get_feature_columns(hist_aug)
    feat_cols_away_resid = sorted(set(feat_cols_away_base).union({"away_base_pred", "v5_regime_id"}))
    return hist_aug, next_aug, feat_cols_home, feat_cols_away_base, feat_cols_away_resid


def sample_weights_side_v5(df: pd.DataFrame, side: str = "home", residual_target: np.ndarray | None = None) -> np.ndarray:
    w = sample_weights(df).astype(float)
    if side != "away":
        return w
    y = df["axg"].astype(float).to_numpy()
    q75 = float(np.nanquantile(y, 0.75)) if len(y) else np.nan
    hi = np.where(np.isfinite(q75) & (y >= q75), 1.10, 1.0)
    open_idx = df.get("v5_open_game_index", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
    open_thr = float(np.nanmedian(open_idx)) if np.isfinite(open_idx).any() else np.nan
    open_boost = np.where(np.isfinite(open_thr) & (open_idx >= open_thr), 1.03, 1.0)
    upside = df.get("v5_away_upside_gate", pd.Series(0.0, index=df.index)).astype(float).to_numpy()
    up_thr = float(np.nanquantile(upside, 0.75)) if np.isfinite(upside).any() else np.nan
    up_boost = np.where(np.isfinite(up_thr) & (upside >= up_thr), 1.04, 1.0)
    out = w * hi * open_boost * up_boost
    if residual_target is not None:
        r = np.asarray(residual_target, dtype=float)
        out = out * np.where(r > 0.10, 1.08, 1.0) * np.where(r < -0.10, 1.03, 1.0)
    return out


def get_xg_model_side_v5(side: str = "home", stage: str = "base"):
    if side != "away" or stage == "base":
        return get_xg_model()
    if _HAS_XGBOOST:
        return _XGBRegressor(
            objective="reg:squarederror",
            n_estimators=320, learning_rate=0.035, max_depth=3,
            min_child_weight=5.0, subsample=0.82, colsample_bytree=0.78,
            reg_alpha=0.12, reg_lambda=1.80, gamma=0.02,
            random_state=RANDOM_STATE, verbosity=0, n_jobs=-1,
        ), "xgb_sqerr_resid_v5"
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.04, max_depth=3,
        max_iter=220, min_samples_leaf=18, l2_regularization=0.14,
        random_state=RANDOM_STATE,
    ), "histgb_resid_v5"


def _build_v5_regime_thresholds(df: pd.DataFrame) -> dict:
    score = df["v5_away_regime_score"].astype(float).to_numpy()
    score = score[np.isfinite(score)]
    if len(score) < 30:
        return {"q1": -0.10, "q2": 0.10}
    q1, q2 = np.nanquantile(score, [0.33, 0.67])
    if not np.isfinite(q1) or not np.isfinite(q2) or q2 <= q1:
        med = float(np.nanmedian(score)) if len(score) else 0.0
        return {"q1": med - 0.05, "q2": med + 0.05}
    return {"q1": float(q1), "q2": float(q2)}


def _assign_v5_regime_ids(score: np.ndarray, thresholds: dict) -> np.ndarray:
    q1 = float(thresholds.get("q1", -0.10))
    q2 = float(thresholds.get("q2", 0.10))
    s = np.asarray(score, dtype=float)
    return np.where(s <= q1, 0, np.where(s <= q2, 1, 2)).astype(int)


def fit_away_regime_v5_bundle(train_df: pd.DataFrame, feat_cols_base: list[str], feat_cols_resid: list[str]):
    tr = train_df.copy()
    Xa_tr = tr[feat_cols_base].fillna(tr[feat_cols_base].median())
    y_a = tr["axg"].clip(lower=XG_CLIP_MIN, upper=XG_CLIP_MAX).to_numpy(dtype=float)
    w_a = sample_weights_side_v5(tr, "away")
    base_model, base_name = get_xg_model_side_v5("away", "base")
    base_model.fit(Xa_tr, y_a, sample_weight=w_a)
    base_pred_tr = np.clip(base_model.predict(Xa_tr), XG_CLIP_MIN, XG_CLIP_MAX)

    tr["away_base_pred"] = base_pred_tr
    thresholds = _build_v5_regime_thresholds(tr)
    tr["v5_regime_id"] = _assign_v5_regime_ids(tr["v5_away_regime_score"].astype(float).to_numpy(), thresholds)

    resid_y = y_a - base_pred_tr
    Xr_tr = tr[feat_cols_resid].fillna(tr[feat_cols_resid].median())
    w_r = sample_weights_side_v5(tr, "away", residual_target=resid_y)
    global_model, resid_name = get_xg_model_side_v5("away", "residual")
    global_model.fit(Xr_tr, resid_y, sample_weight=w_r)

    regime_models = {}
    regime_alpha = {}
    for regime_id in [0, 1, 2]:
        mask = (tr["v5_regime_id"].to_numpy(dtype=int) == regime_id)
        n_reg = int(mask.sum())
        if n_reg < 120:
            regime_alpha[regime_id] = 0.0
            continue
        X_reg = Xr_tr.loc[mask]
        y_reg = resid_y[mask]
        w_reg = w_r[mask]
        m_reg, _ = get_xg_model_side_v5("away", "residual")
        m_reg.fit(X_reg, y_reg, sample_weight=w_reg)
        regime_models[regime_id] = m_reg
        regime_alpha[regime_id] = float(np.clip(0.20 + 0.35 * (n_reg - 120) / 260.0, 0.20, 0.55))

    return {
        "base_model": base_model,
        "base_name": base_name,
        "resid_global_model": global_model,
        "resid_name": resid_name,
        "regime_models": regime_models,
        "regime_alpha": regime_alpha,
        "thresholds": thresholds,
        "feat_cols_base": feat_cols_base,
        "feat_cols_resid": feat_cols_resid,
        "base_fill": tr[feat_cols_base].median(),
        "resid_fill": tr[feat_cols_resid].median(),
    }


def predict_away_regime_v5(bundle: dict, df: pd.DataFrame):
    work = df.copy()
    Xa = work[bundle["feat_cols_base"]].fillna(bundle["base_fill"])
    base_pred = np.clip(bundle["base_model"].predict(Xa), XG_CLIP_MIN, XG_CLIP_MAX)
    work["away_base_pred"] = base_pred
    work["v5_regime_id"] = _assign_v5_regime_ids(work["v5_away_regime_score"].astype(float).to_numpy(), bundle["thresholds"])
    Xr = work[bundle["feat_cols_resid"]].fillna(bundle["resid_fill"])
    resid_global = np.asarray(bundle["resid_global_model"].predict(Xr), dtype=float)
    resid_final = resid_global.copy()
    reg_ids = work["v5_regime_id"].to_numpy(dtype=int)
    for regime_id, model in bundle["regime_models"].items():
        mask = (reg_ids == regime_id)
        if not mask.any():
            continue
        resid_reg = np.asarray(model.predict(Xr.loc[mask]), dtype=float)
        alpha = float(bundle["regime_alpha"].get(regime_id, 0.0))
        resid_final[mask] = (1.0 - alpha) * resid_global[mask] + alpha * resid_reg
    resid_final = np.clip(resid_final, -0.60, 0.85)
    pred_final = np.clip(base_pred + resid_final, XG_CLIP_MIN, XG_CLIP_MAX)
    return base_pred, pred_final, reg_ids


def run_away_regime_v5_backtest_detailed(hist_f: pd.DataFrame, n_folds: int = 5, min_train: int = 500,
                                         benchmark_label: str = CHALLENGER_BENCHMARK_LABEL):
    hist_aug, _, feat_cols_home, feat_cols_away_base, feat_cols_away_resid = prepare_away_regime_v5_frames(hist_f, None)
    _, plan = build_backtest_plan(hist_aug, n_folds=n_folds, min_train=min_train)
    results = []
    oof_rows = []
    print(f"\n  Walk-forward challenger away-regime-v5 ({len(plan)} folds):")
    for spec in plan:
        tr = hist_aug.iloc[spec["train_idx"]].copy()
        te = hist_aug.iloc[spec["test_idx"]].copy()
        Xh_tr = tr[feat_cols_home].fillna(tr[feat_cols_home].median())
        Xh_te = te[feat_cols_home].fillna(Xh_tr.median())
        y_h = tr["hxg"].clip(lower=XG_CLIP_MIN, upper=XG_CLIP_MAX)
        w_h = sample_weights_side_v5(tr, "home")
        mh, _ = get_xg_model_side_v5("home", "base")
        mh.fit(Xh_tr, y_h, sample_weight=w_h)
        pred_h = np.clip(mh.predict(Xh_te), XG_CLIP_MIN, XG_CLIP_MAX)

        away_bundle = fit_away_regime_v5_bundle(tr, feat_cols_away_base, feat_cols_away_resid)
        pred_a_base, pred_a, reg_ids = predict_away_regime_v5(away_bundle, te)
        real_h = te["hxg"].to_numpy(dtype=float)
        real_a = te["axg"].to_numpy(dtype=float)
        metrics = _metrics_from_predictions(real_h, pred_h, real_a, pred_a)
        results.append({
            "benchmark_model": benchmark_label,
            "fold": spec["fold"],
            "n_train": len(tr),
            "n_test": len(te),
            **{k: round(v, 4) if pd.notna(v) else np.nan for k, v in metrics.items()},
            "test_date_start": pd.Timestamp.fromordinal(int(spec["test_keys"].min())).strftime("%Y-%m-%d"),
            "test_date_end": pd.Timestamp.fromordinal(int(spec["test_keys"].max())).strftime("%Y-%m-%d"),
        })
        oof_rows.append(pd.DataFrame({
            "benchmark_model": benchmark_label,
            "fold": spec["fold"],
            "row_idx": spec["test_idx"],
            "pred_home": pred_h,
            "real_home": real_h,
            "pred_away": pred_a,
            "pred_away_base": pred_a_base,
            "real_away": real_a,
            "ano": te["ano"].to_numpy(dtype=int),
            "date": te["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "away_regime_id": reg_ids,
        }))
        print(
            f"    Fold {spec['fold']}: {pd.Timestamp.fromordinal(int(spec['test_keys'].min())).strftime('%Y-%m-%d')}→{pd.Timestamp.fromordinal(int(spec['test_keys'].max())).strftime('%Y-%m-%d')} | "
            f"MAE mean={metrics['mae_mean']:.4f} | MAE away={metrics['mae_away']:.4f} | corr away={metrics['corr_away']:.4f}"
        )
    df_bt = pd.DataFrame(results)
    oof_df = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    feat_pack = {"home": feat_cols_home, "away_base": feat_cols_away_base, "away_resid": feat_cols_away_resid}
    return df_bt, oof_df, feat_pack

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {"home", "away", "hg", "ag", "hxg", "axg", "dataset_source", "date"}
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def build_backtest_plan(hist_f: pd.DataFrame, n_folds: int = 5, min_train: int = 500):
    """Constroi folds walk-forward sem dividir a mesma data entre treino e teste."""
    date_key = hist_f["date"].map(pd.Timestamp.toordinal).astype(int)
    uniq = pd.Index(pd.unique(date_key))
    counts = pd.Series(date_key).value_counts().reindex(uniq).fillna(0).astype(int)
    cum_counts = counts.cumsum()
    start_pos = int(np.searchsorted(cum_counts.values, min_train, side="left"))
    keys_after = uniq[start_pos + 1:]
    folds_effective = max(1, min(n_folds, len(keys_after))) if len(keys_after) > 0 else 1
    chunks = np.array_split(np.array(keys_after, dtype=int), folds_effective)

    plan = []
    for fold, test_keys in enumerate(chunks, start=1):
        if len(test_keys) == 0:
            continue
        te_mask = date_key.isin(test_keys)
        te_idx = np.where(te_mask)[0]
        if len(te_idx) == 0:
            continue
        te_s = int(te_idx.min())
        te_e = int(te_idx.max()) + 1
        tr_idx = np.arange(0, te_s, dtype=int)
        if len(tr_idx) < min_train or te_e <= te_s:
            continue
        plan.append({
            "fold": fold,
            "test_keys": np.array(test_keys, dtype=int),
            "train_idx": tr_idx,
            "test_idx": np.arange(te_s, te_e, dtype=int),
        })
    return date_key, plan


def _safe_corr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 4:
        return np.nan
    if np.nanstd(y_true) <= 1e-12 or np.nanstd(y_pred) <= 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _metrics_from_predictions(real_h, pred_h, real_a, pred_a) -> dict:
    real_h = np.asarray(real_h, dtype=float)
    pred_h = np.asarray(pred_h, dtype=float)
    real_a = np.asarray(real_a, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    mae_h = float(mean_absolute_error(real_h, pred_h))
    mae_a = float(mean_absolute_error(real_a, pred_a))
    return {
        "mae_home": mae_h,
        "mae_away": mae_a,
        "mae_mean": (mae_h + mae_a) / 2.0,
        "rmse_home": float(np.sqrt(mean_squared_error(real_h, pred_h))),
        "rmse_away": float(np.sqrt(mean_squared_error(real_a, pred_a))),
        "bias_home": float(np.mean(pred_h - real_h)),
        "bias_away": float(np.mean(pred_a - real_a)),
        "corr_home": _safe_corr(real_h, pred_h),
        "corr_away": _safe_corr(real_a, pred_a),
    }


def run_xg_backtest_detailed(hist_f: pd.DataFrame, n_folds: int = 5, min_train: int = 500,
                             benchmark_label: str = OFFICIAL_BENCHMARK_LABEL):
    """Backtest detalhado do modelo oficial, com OOF por linha para benchmark interno."""
    feat_cols = get_feature_columns(hist_f)
    _, plan = build_backtest_plan(hist_f, n_folds=n_folds, min_train=min_train)

    results = []
    oof_rows = []
    print(f"\n  Walk-forward xG backtest por data ({len(plan)} folds):")
    for spec in plan:
        tr = hist_f.iloc[spec["train_idx"]]
        te = hist_f.iloc[spec["test_idx"]]
        X_tr = tr[feat_cols].fillna(tr[feat_cols].median())
        X_te = te[feat_cols].fillna(X_tr.median())
        y_h = tr["hxg"].clip(lower=XG_CLIP_MIN, upper=XG_CLIP_MAX)
        y_a = tr["axg"].clip(lower=XG_CLIP_MIN, upper=XG_CLIP_MAX)
        w = sample_weights(tr)

        mh, _ = get_xg_model()
        ma, _ = get_xg_model()
        mh.fit(X_tr, y_h, sample_weight=w)
        ma.fit(X_tr, y_a, sample_weight=w)

        pred_h = np.clip(mh.predict(X_te), XG_CLIP_MIN, XG_CLIP_MAX)
        pred_a = np.clip(ma.predict(X_te), XG_CLIP_MIN, XG_CLIP_MAX)
        real_h = te["hxg"].to_numpy(dtype=float)
        real_a = te["axg"].to_numpy(dtype=float)
        metrics = _metrics_from_predictions(real_h, pred_h, real_a, pred_a)

        results.append({
            "benchmark_model": benchmark_label,
            "fold": spec["fold"],
            "n_train": len(tr),
            "n_test": len(te),
            **{k: round(v, 4) if pd.notna(v) else np.nan for k, v in metrics.items()},
            "test_date_start": pd.Timestamp.fromordinal(int(spec["test_keys"].min())).strftime("%Y-%m-%d"),
            "test_date_end": pd.Timestamp.fromordinal(int(spec["test_keys"].max())).strftime("%Y-%m-%d"),
        })

        fold_rows = pd.DataFrame({
            "benchmark_model": benchmark_label,
            "fold": spec["fold"],
            "row_idx": spec["test_idx"],
            "pred_home": pred_h,
            "real_home": real_h,
            "pred_away": pred_a,
            "real_away": real_a,
        })
        oof_rows.append(fold_rows)
        print(
            f"    fold {spec['fold']}: MAE_h={metrics['mae_home']:.4f} "
            f"MAE_a={metrics['mae_away']:.4f} "
            f"bias_h={metrics['bias_home']:+.4f} bias_a={metrics['bias_away']:+.4f}"
        )

    df_bt = pd.DataFrame(results)
    oof_df = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    if not df_bt.empty:
        print(
            f"  → MAE médio: {df_bt['mae_mean'].mean():.4f} | "
            f"Bias_h: {df_bt['bias_home'].mean():+.4f} | "
            f"Bias_a: {df_bt['bias_away'].mean():+.4f}"
        )
    return df_bt, oof_df, feat_cols


def run_xg_backtest(hist_f: pd.DataFrame, n_folds: int = 5, min_train: int = 500):
    df_bt, oof_df, _ = run_xg_backtest_detailed(hist_f, n_folds=n_folds, min_train=min_train)
    return (
        df_bt,
        oof_df["pred_home"].to_numpy(dtype=float),
        oof_df["real_home"].to_numpy(dtype=float),
        oof_df["pred_away"].to_numpy(dtype=float),
        oof_df["real_away"].to_numpy(dtype=float),
    )


def run_feature_baseline_backtest(hist_f: pd.DataFrame, pred_home_col: str, pred_away_col: str,
                                  label: str, n_folds: int = 5, min_train: int = 500):
    """Benchmark simples usando colunas causais já montadas como baseline fixo."""
    _, plan = build_backtest_plan(hist_f, n_folds=n_folds, min_train=min_train)
    results = []
    oof_rows = []
    for spec in plan:
        tr = hist_f.iloc[spec["train_idx"]]
        te = hist_f.iloc[spec["test_idx"]]
        fill_h = float(tr["hxg"].mean())
        fill_a = float(tr["axg"].mean())
        pred_h = te[pred_home_col].astype(float).fillna(fill_h).clip(XG_CLIP_MIN, XG_CLIP_MAX).to_numpy()
        pred_a = te[pred_away_col].astype(float).fillna(fill_a).clip(XG_CLIP_MIN, XG_CLIP_MAX).to_numpy()
        real_h = te["hxg"].to_numpy(dtype=float)
        real_a = te["axg"].to_numpy(dtype=float)
        metrics = _metrics_from_predictions(real_h, pred_h, real_a, pred_a)
        results.append({
            "benchmark_model": label,
            "fold": spec["fold"],
            "n_train": len(tr),
            "n_test": len(te),
            **{k: round(v, 4) if pd.notna(v) else np.nan for k, v in metrics.items()},
            "test_date_start": pd.Timestamp.fromordinal(int(spec["test_keys"].min())).strftime("%Y-%m-%d"),
            "test_date_end": pd.Timestamp.fromordinal(int(spec["test_keys"].max())).strftime("%Y-%m-%d"),
        })
        oof_rows.append(pd.DataFrame({
            "benchmark_model": label,
            "fold": spec["fold"],
            "row_idx": spec["test_idx"],
            "pred_home": pred_h,
            "real_home": real_h,
            "pred_away": pred_a,
            "real_away": real_a,
        }))
    return pd.DataFrame(results), (pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame())


def run_league_mean_backtest(hist_f: pd.DataFrame, label: str = "baseline_league_mean",
                             n_folds: int = 5, min_train: int = 500):
    """Baseline mínimo: média histórica do treino em cada fold."""
    _, plan = build_backtest_plan(hist_f, n_folds=n_folds, min_train=min_train)
    results = []
    oof_rows = []
    for spec in plan:
        tr = hist_f.iloc[spec["train_idx"]]
        te = hist_f.iloc[spec["test_idx"]]
        pred_h = np.full(len(te), float(tr["hxg"].mean()))
        pred_a = np.full(len(te), float(tr["axg"].mean()))
        real_h = te["hxg"].to_numpy(dtype=float)
        real_a = te["axg"].to_numpy(dtype=float)
        metrics = _metrics_from_predictions(real_h, pred_h, real_a, pred_a)
        results.append({
            "benchmark_model": label,
            "fold": spec["fold"],
            "n_train": len(tr),
            "n_test": len(te),
            **{k: round(v, 4) if pd.notna(v) else np.nan for k, v in metrics.items()},
            "test_date_start": pd.Timestamp.fromordinal(int(spec["test_keys"].min())).strftime("%Y-%m-%d"),
            "test_date_end": pd.Timestamp.fromordinal(int(spec["test_keys"].max())).strftime("%Y-%m-%d"),
        })
        oof_rows.append(pd.DataFrame({
            "benchmark_model": label,
            "fold": spec["fold"],
            "row_idx": spec["test_idx"],
            "pred_home": pred_h,
            "real_home": real_h,
            "pred_away": pred_a,
            "real_away": real_a,
        }))
    return pd.DataFrame(results), (pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame())


def build_phase_metrics(hist_f: pd.DataFrame, oof_all: pd.DataFrame) -> pd.DataFrame:
    if oof_all is None or oof_all.empty:
        return pd.DataFrame()
    meta = hist_f[["ano", "date", "rodada"]].copy().reset_index().rename(columns={"index": "row_idx"})
    max_step = hist_f.groupby("ano")["rodada"].max().rename("max_step").reset_index()
    meta = meta.merge(max_step, on="ano", how="left")
    merged = oof_all.merge(meta, on="row_idx", how="left")
    ratio = merged["rodada"] / merged["max_step"].clip(lower=1)
    merged["phase"] = np.where(ratio <= 1/3, "inicio", np.where(ratio <= 2/3, "meio", "fim"))

    rows = []
    for (model, phase), g in merged.groupby(["benchmark_model", "phase"], sort=True):
        metrics = _metrics_from_predictions(g["real_home"], g["pred_home"], g["real_away"], g["pred_away"])
        rows.append({
            "benchmark_model": model,
            "phase": phase,
            "n": int(len(g)),
            **{k: round(v, 4) if pd.notna(v) else np.nan for k, v in metrics.items()},
        })
    return pd.DataFrame(rows).sort_values(["benchmark_model", "phase"]).reset_index(drop=True)



def build_benchmark_summary(fold_df: pd.DataFrame, oof_all: pd.DataFrame) -> pd.DataFrame:
    if fold_df is None or fold_df.empty:
        return pd.DataFrame()
    metric_cols = ["mae_home", "mae_away", "mae_mean", "rmse_home", "rmse_away", "bias_home", "bias_away", "corr_home", "corr_away"]
    summary = fold_df.groupby("benchmark_model", as_index=False)[metric_cols].mean()
    counts = fold_df.groupby("benchmark_model").size().rename("folds").reset_index()
    rows = oof_all.groupby("benchmark_model").size().rename("rows_oof").reset_index() if oof_all is not None and not oof_all.empty else pd.DataFrame(columns=["benchmark_model", "rows_oof"])
    summary = summary.merge(counts, on="benchmark_model", how="left").merge(rows, on="benchmark_model", how="left")
    summary = summary.sort_values(["mae_mean", "mae_home", "mae_away"]).reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    for c in metric_cols:
        summary[c] = summary[c].round(4)
    return summary


def build_compare_vs_official(summary_df: pd.DataFrame, official_label: str = OFFICIAL_BENCHMARK_LABEL) -> pd.DataFrame:
    if summary_df is None or summary_df.empty or official_label not in set(summary_df["benchmark_model"]):
        return pd.DataFrame()
    metrics = ["mae_home", "mae_away", "mae_mean", "rmse_home", "rmse_away", "bias_home", "bias_away", "corr_home", "corr_away"]
    official = summary_df.loc[summary_df["benchmark_model"] == official_label, metrics].iloc[0]
    compare = summary_df.copy()
    for c in metrics:
        compare[f"delta_vs_{official_label}_{c}"] = (compare[c] - official[c]).round(4)
    return compare


def write_benchmark_report(summary_df: pd.DataFrame, compare_df: pd.DataFrame, output_txt: str,
                           fold_df: pd.DataFrame | None = None, phase_df: pd.DataFrame | None = None):
    lines = []
    lines.append(f"Benchmark interno | versão oficial: {MODEL_VERSION}")
    lines.append(f"Modelo oficial: {OFFICIAL_BENCHMARK_LABEL}")
    lines.append(f"Challenger monitorado: {CHALLENGER_BENCHMARK_LABEL}")
    lines.append("")
    if summary_df is None or summary_df.empty:
        lines.append("Sem resultados de benchmark.")
    else:
        winner = summary_df.iloc[0]
        lines.append(f"Vencedor por MAE médio: {winner['benchmark_model']} | mae_mean={winner['mae_mean']:.4f}")
        if OFFICIAL_BENCHMARK_LABEL in set(summary_df["benchmark_model"]):
            official = summary_df.loc[summary_df["benchmark_model"] == OFFICIAL_BENCHMARK_LABEL].iloc[0]
            lines.append(f"Oficial: mae_mean={official['mae_mean']:.4f} | mae_home={official['mae_home']:.4f} | mae_away={official['mae_away']:.4f}")
        lines.append("")
        lines.append("Resumo:")
        for _, row in summary_df.iterrows():
            lines.append(
                f"- {row['benchmark_model']}: mae_mean={row['mae_mean']:.4f} | "
                f"mae_home={row['mae_home']:.4f} | mae_away={row['mae_away']:.4f} | "
                f"corr_home={row['corr_home']:.4f} | corr_away={row['corr_away']:.4f}"
            )
        if compare_df is not None and not compare_df.empty:
            lines.append("")
            lines.append(f"Deltas vs {OFFICIAL_BENCHMARK_LABEL} (mae_mean negativo = melhor):")
            for _, row in compare_df.iterrows():
                lines.append(
                    f"- {row['benchmark_model']}: Δmae_mean={row[f'delta_vs_{OFFICIAL_BENCHMARK_LABEL}_mae_mean']:+.4f} | "
                    f"Δmae_home={row[f'delta_vs_{OFFICIAL_BENCHMARK_LABEL}_mae_home']:+.4f} | "
                    f"Δmae_away={row[f'delta_vs_{OFFICIAL_BENCHMARK_LABEL}_mae_away']:+.4f} | "
                    f"Δcorr_away={row[f'delta_vs_{OFFICIAL_BENCHMARK_LABEL}_corr_away']:+.4f}"
                )

        decision = evaluate_challenger_promotion(summary_df, compare_df, fold_df=fold_df, phase_df=phase_df)
        lines.append("")
        lines.append("Gate de promoção challenger → official:")
        lines.append(f"- decisão: {'PROMOVER' if decision['promoted'] else 'MANTER official atual'}")
        lines.append(f"- motivo: {decision['reason']}")
        if 'delta_mae_mean' in decision:
            lines.append(
                f"- deltas challenger: mae_mean={decision['delta_mae_mean']:+.4f} | "
                f"mae_home={decision['delta_mae_home']:+.4f} | "
                f"mae_away={decision['delta_mae_away']:+.4f} | "
                f"corr_away={decision['delta_corr_away']:+.4f}"
            )
            lines.append(
                f"- wins: away_folds={decision['fold_wins_away']} | mean_folds={decision['fold_wins_mean']} | "
                f"away_phases={decision['phase_wins_away']}"
            )
            lines.append("- critérios:")
            lines.append(f"  * Δmae_mean <= {PROMOTE_MAX_DELTA_MAE_MEAN:+.4f}: {decision['mae_mean_gate']}")
            lines.append(f"  * Δmae_away <= {PROMOTE_MAX_DELTA_MAE_AWAY:+.4f}: {decision['mae_away_gate']}")
            lines.append(f"  * Δmae_home <= {PROMOTE_MAX_DELTA_MAE_HOME:+.4f}: {decision['mae_home_guard']}")
            lines.append(f"  * Δcorr_away >= {PROMOTE_MIN_DELTA_CORR_AWAY:+.4f}: {decision['corr_away_guard']}")
            lines.append(f"  * wins away folds >= {PROMOTE_MIN_FOLD_WINS_AWAY}: {decision['fold_wins_away_guard']}")
            lines.append(f"  * wins away phases >= {PROMOTE_MIN_PHASE_WINS_AWAY}: {decision['phase_wins_away_guard']}")
    Path(output_txt).write_text("\n".join(lines), encoding="utf-8")



def run_native_benchmark_suite(hist_f: pd.DataFrame, output_prefix: str = BENCHMARK_PREFIX_DEFAULT,
                               n_folds: int = BACKTEST_N_FOLDS, min_train: int = BACKTEST_MIN_TRAIN,
                               official_cache: tuple[pd.DataFrame, pd.DataFrame] | None = None):
    """Benchmark nativo: official vs challenger away-regime-v5 vs baselines fixas."""
    banner("BENCHMARK INTERNO — OFFICIAL VS CHALLENGER VS BASELINES")
    fold_frames = []
    oof_frames = []

    if official_cache is None:
        df_off, oof_off, _ = run_xg_backtest_detailed(hist_f, n_folds=n_folds, min_train=min_train)
    else:
        df_off, oof_off = official_cache
    if df_off is not None and not df_off.empty:
        fold_frames.append(df_off.copy())
    if oof_off is not None and not oof_off.empty:
        oof_frames.append(oof_off.copy())

    print(f"  Rodando {CHALLENGER_BENCHMARK_LABEL}...")
    df_ch, oof_ch, _ = run_away_regime_v5_backtest_detailed(hist_f, n_folds=n_folds, min_train=min_train)
    if df_ch is not None and not df_ch.empty:
        fold_frames.append(df_ch.copy())
    if oof_ch is not None and not oof_ch.empty:
        oof_frames.append(oof_ch.copy())

    baseline_specs = []
    if {"xg_h_vs_def_a", "xg_a_vs_def_h"}.issubset(hist_f.columns):
        baseline_specs.append(("baseline_contextual_xg", "xg_h_vs_def_a", "xg_a_vs_def_h"))
    if {"dc_lam_h", "dc_lam_a"}.issubset(hist_f.columns):
        baseline_specs.append(("baseline_dc_ref", "dc_lam_h", "dc_lam_a"))

    for label, ph, pa in baseline_specs:
        print(f"  Rodando {label}...")
        df_b, oof_b = run_feature_baseline_backtest(hist_f, ph, pa, label=label, n_folds=n_folds, min_train=min_train)
        if not df_b.empty:
            fold_frames.append(df_b)
        if not oof_b.empty:
            oof_frames.append(oof_b)

    print("  Rodando baseline_league_mean...")
    df_lg, oof_lg = run_league_mean_backtest(hist_f, label="baseline_league_mean", n_folds=n_folds, min_train=min_train)
    if not df_lg.empty:
        fold_frames.append(df_lg)
    if not oof_lg.empty:
        oof_frames.append(oof_lg)

    fold_all = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    oof_all = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    phase_df = build_phase_metrics(hist_f, oof_all)
    summary_df = build_benchmark_summary(fold_all, oof_all)
    compare_df = build_compare_vs_official(summary_df)
    promotion_decision = evaluate_challenger_promotion(summary_df, compare_df, fold_df=fold_all, phase_df=phase_df)

    fold_path = f"{output_prefix}_fold_metrics.csv"
    phase_path = f"{output_prefix}_phase_metrics.csv"
    summary_path = f"{output_prefix}_summary.csv"
    compare_path = f"{output_prefix}_compare.csv"
    report_path = f"{output_prefix}_report.txt"

    if not fold_all.empty:
        fold_all.to_csv(fold_path, index=False)
    if not phase_df.empty:
        phase_df.to_csv(phase_path, index=False)
    if not summary_df.empty:
        summary_df.to_csv(summary_path, index=False)
    if not compare_df.empty:
        compare_df.to_csv(compare_path, index=False)
    write_benchmark_report(summary_df, compare_df, report_path, fold_df=fold_all, phase_df=phase_df)

    if not summary_df.empty:
        print("\n  Ranking do benchmark interno:")
        for _, row in summary_df.iterrows():
            print(f"    {int(row['rank'])}. {row['benchmark_model']:<24} mae_mean={row['mae_mean']:.4f}")
        print(
            f"  Gate de promoção challenger: "
            f"{'PROMOVER' if promotion_decision['promoted'] else 'MANTER official'}"
        )

    return {
        "fold": fold_all,
        "phase": phase_df,
        "summary": summary_df,
        "compare": compare_df,
        "promotion_decision": promotion_decision,
        "paths": {
            "fold": fold_path,
            "phase": phase_path,
            "summary": summary_path,
            "compare": compare_path,
            "report": report_path,
        }
    }


# ══════════════════════════════════════════════════════════════════════════════
# CORREÇÃO ISOTÔNICA DE BIAS OOF

# ══════════════════════════════════════════════════════════════════════════════
# CORREÇÃO ISOTÔNICA DE BIAS OOF
# ══════════════════════════════════════════════════════════════════════════════

class XGBiasCorrector:
    """Calibração isotônica do ponto central usando previsões OOF."""

    def __init__(self):
        self._iso_h = None; self._iso_a = None
        self.bias_before_h = np.nan; self.bias_after_h  = np.nan
        self.bias_before_a = np.nan; self.bias_after_a  = np.nan
        self.mae_before_h  = np.nan; self.mae_after_h   = np.nan
        self.mae_before_a  = np.nan; self.mae_after_a   = np.nan

    def fit(self, oof_pred_h, oof_real_h, oof_pred_a, oof_real_a):
        self._iso_h = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._iso_h.fit(oof_pred_h, oof_real_h)
        self._iso_a = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._iso_a.fit(oof_pred_a, oof_real_a)

        corr_h = self._iso_h.predict(oof_pred_h)
        corr_a = self._iso_a.predict(oof_pred_a)
        self.bias_before_h = float(np.mean(oof_pred_h - oof_real_h))
        self.bias_after_h  = float(np.mean(corr_h - oof_real_h))
        self.bias_before_a = float(np.mean(oof_pred_a - oof_real_a))
        self.bias_after_a  = float(np.mean(corr_a - oof_real_a))
        self.mae_before_h  = float(mean_absolute_error(oof_real_h, oof_pred_h))
        self.mae_after_h   = float(mean_absolute_error(oof_real_h, corr_h))
        self.mae_before_a  = float(mean_absolute_error(oof_real_a, oof_pred_a))
        self.mae_after_a   = float(mean_absolute_error(oof_real_a, corr_a))

    def correct_h(self, p):
        arr = np.asarray(p, dtype=float)
        return np.clip(self._iso_h.predict(arr), XG_CLIP_MIN, XG_CLIP_MAX) if self._iso_h else arr

    def correct_a(self, p):
        arr = np.asarray(p, dtype=float)
        return np.clip(self._iso_a.predict(arr), XG_CLIP_MIN, XG_CLIP_MAX) if self._iso_a else arr

    def report(self):
        print("\n  Correção isotônica OOF:")
        print(f"    Home | Bias: {self.bias_before_h:+.4f} → {self.bias_after_h:+.4f} | MAE: {self.mae_before_h:.4f} → {self.mae_after_h:.4f}")
        print(f"    Away | Bias: {self.bias_before_a:+.4f} → {self.bias_after_a:+.4f} | MAE: {self.mae_before_a:.4f} → {self.mae_after_a:.4f}")


class XGIntervalCalibrator:
    """
    Intervalos conformais assimétricos por faixas de predição.
    Usa resíduos OOF do ponto central já corrigido por isotônica.
    """

    def __init__(self, n_bins: int = 4, min_bin_size: int = 60):
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self._spec_h = None
        self._spec_a = None

    def _fit_side(self, pred: np.ndarray, real: np.ndarray) -> dict:
        pred = np.asarray(pred, dtype=float)
        real = np.asarray(real, dtype=float)
        resid = real - pred
        ql_g = float(np.quantile(resid, 0.10))
        qh_g = float(np.quantile(resid, 0.90))
        edges = np.unique(np.quantile(pred, np.linspace(0, 1, self.n_bins + 1)))
        if len(edges) < 2:
            edges = np.array([pred.min(), pred.max() + 1e-9])
        bins = []
        for i in range(len(edges) - 1):
            lo = float(edges[i]); hi = float(edges[i + 1])
            if i == len(edges) - 2:
                mask = (pred >= lo) & (pred <= hi)
            else:
                mask = (pred >= lo) & (pred < hi)
            n = int(mask.sum())
            if n >= self.min_bin_size:
                ql = float(np.quantile(resid[mask], 0.10))
                qh = float(np.quantile(resid[mask], 0.90))
            else:
                ql, qh = ql_g, qh_g
            bins.append({"lo": lo, "hi": hi, "ql": ql, "qh": qh, "n": n})
        return {"global": {"ql": ql_g, "qh": qh_g}, "bins": bins}

    def fit(self, pred_h: np.ndarray, real_h: np.ndarray,
            pred_a: np.ndarray, real_a: np.ndarray):
        self._spec_h = self._fit_side(pred_h, real_h)
        self._spec_a = self._fit_side(pred_a, real_a)

    def _apply_side(self, pred: np.ndarray, spec: dict):
        pred = np.asarray(pred, dtype=float)
        low = np.empty_like(pred)
        high = np.empty_like(pred)
        bins = spec["bins"] if spec else []
        g = spec["global"] if spec else {"ql": 0.0, "qh": 0.0}
        for i, p in enumerate(pred):
            chosen = None
            for b in bins:
                is_last = (b is bins[-1])
                if (p >= b["lo"] and p < b["hi"]) or (is_last and p >= b["lo"] and p <= b["hi"]):
                    chosen = b
                    break
            if chosen is None:
                ql, qh = g["ql"], g["qh"]
            else:
                ql, qh = chosen["ql"], chosen["qh"]
            low[i] = np.clip(p + ql, XG_CLIP_MIN, XG_CLIP_MAX)
            high[i] = np.clip(p + qh, XG_CLIP_MIN, XG_CLIP_MAX)
        low = np.minimum(low, pred)
        high = np.maximum(high, pred)
        return low, high

    def interval_h(self, pred: np.ndarray):
        return self._apply_side(pred, self._spec_h)

    def interval_a(self, pred: np.ndarray):
        return self._apply_side(pred, self._spec_a)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN / CLI
# ══════════════════════════════════════════════════════════════════════════════

from pathlib import Path


def load_input_frames(passadas_file: str = PASSADAS_FILE,
                      atuais_file: str = ATUAIS_FILE,
                      proxima_file: str = PROXIMA_FILE):
    for f in [passadas_file, atuais_file, proxima_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Arquivo não encontrado: {f}")

    print("\n📂 Carregando dados...")
    passadas = prep_df(pd.read_excel(passadas_file), is_future=False)
    atuais = prep_df(pd.read_excel(atuais_file), is_future=False)
    proxima = prep_df(pd.read_excel(proxima_file), is_future=True)

    passadas["dataset_source"] = "passadas"
    atuais["dataset_source"] = "atuais"
    hist = pd.concat([passadas, atuais], ignore_index=True, sort=False)
    hist = hist.sort_values(["date", "_row"]).reset_index(drop=True)
    return passadas, atuais, proxima, hist


def build_feature_pack(hist: pd.DataFrame, proxima: pd.DataFrame):
    print(f"   {len(hist)} jogos | Anos: {sorted(hist['ano'].unique())}")
    print(f"   Próxima data {proxima['date'].dt.strftime('%Y-%m-%d').iloc[0]}: {len(proxima)} jogos")
    print("\n  Jogos a prever:")
    for _, r in proxima.iterrows():
        print(f"    {r['home']:<24} vs  {r['away']}")

    league_means = compute_league_means(hist)
    print(
        f"\n  League means finais (xG): hxG_casa={league_means['h_home_xgf_pg']:.3f} | "
        f"axG_fora={league_means['a_away_xgf_pg']:.3f}"
    )

    banner("STEP 1/5 — Snapshots causais de força")
    print("  Construindo snapshots rolling incrementais (janela móvel + meia-vida)...")
    snapshot_ticks = sorted(set(hist["tick"].astype(int)).union(set(proxima["tick"].astype(int))))
    dc_snapshots, team_ha_snapshots, final_snapshot = build_incremental_strength_snapshots(hist, snapshot_ticks)
    attack_r, defense_r, base_home_log, base_away_log, team_ha = final_snapshot
    print(f"  Snapshots gerados: {len(dc_snapshots)} ticks")
    print(f"  Base final home={np.exp(base_home_log):.3f} | away={np.exp(base_away_log):.3f}")
    top_ha = sorted(team_ha.items(), key=lambda x: -x[1])[:4]
    if top_ha:
        print(f"  Per-team HA final (top): {', '.join(f'{t}={np.exp(v):.2f}x' for t, v in top_ha)}")

    banner("STEP 2/5 — Montagem de features (causal por data)")
    hist_f, next_f, ranking = build_datasets(
        hist, proxima, dc_snapshots, team_ha_snapshots,
        attack_r, defense_r, base_home_log, base_away_log, team_ha,
    )
    feat_cols = get_feature_columns(hist_f)
    print(f"  ✅ {len(hist_f)} jogos | {len(feat_cols)} features | {len(next_f)} para prever")

    if not ranking.empty:
        print(f"\n  TABELA DO CAMPEONATO")
        print(f"  {'#':>3}  {'Time':<24} {'Pts':>5} {'J':>3} {'xGD':>6} {'ppg':>5}")
        print(f"  {SEP2}")
        for _, r in ranking.iterrows():
            print(
                f"  {int(r['rank']):>3}  {r['team']:<24} {int(r['points']):>5} {int(r['games']):>3} "
                f"{r['xgd']:>6.2f} {r['ppg']:>5.2f}"
            )

    return {
        "hist": hist,
        "proxima": proxima,
        "hist_f": hist_f,
        "next_f": next_f,
        "ranking": ranking,
        "feat_cols": feat_cols,
        "attack_r": attack_r,
        "defense_r": defense_r,
        "base_home_log": base_home_log,
        "base_away_log": base_away_log,
        "team_ha": team_ha,
    }


def build_info_frame(mh_name: str, hist_f: pd.DataFrame, feat_cols: list[str], df_bt: pd.DataFrame,
                     bias_corrector: XGBiasCorrector, benchmark_summary: pd.DataFrame | None = None,
                     promotion_decision: dict | None = None):
    bt_mae = df_bt["mae_mean"].mean() if not df_bt.empty else np.nan
    bt_bias = df_bt["bias_home"].mean() if not df_bt.empty else np.nan
    winner = None
    if benchmark_summary is not None and not benchmark_summary.empty:
        winner = str(benchmark_summary.iloc[0]["benchmark_model"])
    promoted = None
    if promotion_decision is not None:
        promoted = promotion_decision.get("promoted")
    rows = [
        {"item": "model_version",              "value": MODEL_VERSION},
        {"item": "modelo_xg",                  "value": mh_name},
        {"item": "official_benchmark_label",   "value": OFFICIAL_BENCHMARK_LABEL},
        {"item": "challenger_benchmark_label", "value": CHALLENGER_BENCHMARK_LABEL},
        {"item": "benchmark_winner",           "value": winner},
        {"item": "promotion_decision",         "value": promoted},
        {"item": "tweedie_power",              "value": TWEEDIE_POWER},
        {"item": "xg_clip_max",                "value": XG_CLIP_MAX},
        {"item": "proxy_fallback_removido",    "value": True},
        {"item": "goal_features_removidas",    "value": True},
        {"item": "calibracao_gols_removida",   "value": True},
        {"item": "team_ha_usa_hxg",            "value": True},
        {"item": "league_means_xg_apenas",     "value": True},
        {"item": "intervalo_conformal_oof",    "value": True},
        {"item": "dc_snapshot_por_tick",       "value": True},
        {"item": "dc_engine",                  "value": "rolling_incremental"},
        {"item": "away_engine_v5",             "value": True},
        {"item": "away_engine_v5_type",        "value": "base_plus_regime_residual"},
        {"item": "away_promotion_gates",       "value": True},
        {"item": "dc_halflife_ticks",          "value": DC_HALFLIFE_TICKS},
        {"item": "dc_roll_window_ticks",       "value": DC_ROLL_WINDOW_TICKS},
        {"item": "h2h_prior_n",                "value": H2H_PRIOR_N},
        {"item": "backtest_mae_xg",            "value": round(bt_mae, 4)},
        {"item": "backtest_bias",              "value": round(bt_bias, 4)},
        {"item": "bias_home_antes_correcao",   "value": round(bias_corrector.bias_before_h, 4)},
        {"item": "bias_home_apos_correcao",    "value": round(bias_corrector.bias_after_h, 4)},
        {"item": "bias_away_antes_correcao",   "value": round(bias_corrector.bias_before_a, 4)},
        {"item": "bias_away_apos_correcao",    "value": round(bias_corrector.bias_after_a, 4)},
        {"item": "mae_home_antes_correcao",    "value": round(bias_corrector.mae_before_h, 4)},
        {"item": "mae_home_apos_correcao",     "value": round(bias_corrector.mae_after_h, 4)},
        {"item": "mae_away_antes_correcao",    "value": round(bias_corrector.mae_before_a, 4)},
        {"item": "mae_away_apos_correcao",     "value": round(bias_corrector.mae_after_a, 4)},
        {"item": "season_decay",               "value": SEASON_DECAY},
        {"item": "dc_decay",                   "value": DC_DECAY},
        {"item": "random_state_global",        "value": RANDOM_STATE},
        {"item": "jogos_historico",            "value": len(hist_f)},
        {"item": "n_features",                 "value": len(feat_cols)},
    ]
    if promotion_decision is not None:
        for k in ["delta_mae_mean", "delta_mae_home", "delta_mae_away", "delta_corr_away", "fold_wins_away", "phase_wins_away"]:
            if k in promotion_decision:
                rows.append({"item": f"promotion_{k}", "value": promotion_decision[k]})
    return pd.DataFrame(rows)


def export_prediction_workbook(output_file: str, df_pred: pd.DataFrame, df_bt: pd.DataFrame,
                               ranking: pd.DataFrame, info: pd.DataFrame,
                               benchmark_summary: pd.DataFrame | None = None):
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_pred.to_excel(writer, sheet_name="previsoes_xg", index=False)
        df_bt.to_excel(writer, sheet_name="backtest_xg", index=False)
        if ranking is not None and not ranking.empty:
            ranking.to_excel(writer, sheet_name="ranking", index=False)
        info.to_excel(writer, sheet_name="info_modelo", index=False)
        if benchmark_summary is not None and not benchmark_summary.empty:
            benchmark_summary.to_excel(writer, sheet_name="benchmark_resumo", index=False)


def run_official_prediction_pipeline(pack: dict, n_folds: int = BACKTEST_N_FOLDS,
                                     min_train: int = BACKTEST_MIN_TRAIN):
    hist_f = pack["hist_f"]
    next_f = pack["next_f"]
    proxima = pack["proxima"]

    hist_aug, next_aug, feat_cols_home, feat_cols_away_base, feat_cols_away_resid = prepare_away_regime_v5_frames(hist_f, next_f)
    feat_union = sorted(set(feat_cols_home).union(set(feat_cols_away_base)).union(set(feat_cols_away_resid)))

    banner("STEP 3/5 — Backtesting xG + calibração OOF (v5 away-regime)")
    df_bt, oof_df, feat_pack = run_away_regime_v5_backtest_detailed(hist_f, n_folds=n_folds, min_train=min_train)

    bias_corrector = XGBiasCorrector()
    bias_corrector.fit(
        oof_df["pred_home"].to_numpy(dtype=float),
        oof_df["real_home"].to_numpy(dtype=float),
        oof_df["pred_away"].to_numpy(dtype=float),
        oof_df["real_away"].to_numpy(dtype=float),
    )
    bias_corrector.report()

    banner("STEP 4/5 — Treino final (home official + away v5 regime)")
    Xh_all = hist_aug[feat_pack["home"]].fillna(hist_aug[feat_pack["home"]].median())
    Xh_next = next_aug[feat_pack["home"]].fillna(Xh_all.median())

    y_hxg = hist_aug["hxg"].clip(lower=XG_CLIP_MIN, upper=XG_CLIP_MAX)
    w_h = sample_weights_side_v5(hist_aug, "home")

    mh, mh_name = get_xg_model_side_v5("home", "base")
    mh.fit(Xh_all, y_hxg, sample_weight=w_h)

    away_bundle = fit_away_regime_v5_bundle(hist_aug, feat_pack["away_base"], feat_pack["away_resid"])

    xg_h_raw = np.clip(mh.predict(Xh_next), XG_CLIP_MIN, XG_CLIP_MAX)
    _, xg_a_raw, reg_ids_next = predict_away_regime_v5(away_bundle, next_aug)
    xg_h = bias_corrector.correct_h(xg_h_raw)
    xg_a = bias_corrector.correct_a(xg_a_raw)
    print(f"  Modelo home: {mh_name} | modelo away: {away_bundle["base_name"]}+{away_bundle["resid_name"]}")
    print(
        f"  Correção isotônica: home raw={xg_h_raw.mean():.3f} → {xg_h.mean():.3f} | "
        f"away raw={xg_a_raw.mean():.3f} → {xg_a.mean():.3f}"
    )

    print("  Ajustando IC 80% com conformal OOF por faixas de predição...")
    oof_h_corr = bias_corrector.correct_h(oof_df["pred_home"].to_numpy(dtype=float))
    oof_a_corr = bias_corrector.correct_a(oof_df["pred_away"].to_numpy(dtype=float))
    interval_cal = XGIntervalCalibrator(n_bins=4, min_bin_size=60)
    interval_cal.fit(
        oof_h_corr,
        oof_df["real_home"].to_numpy(dtype=float),
        oof_a_corr,
        oof_df["real_away"].to_numpy(dtype=float),
    )
    xg_h_q10, xg_h_q90 = interval_cal.interval_h(xg_h)
    xg_a_q10, xg_a_q90 = interval_cal.interval_a(xg_a)

    banner("STEP 5/5 — Previsão da próxima rodada")
    rows = []
    for i, (_, row) in enumerate(proxima.iterrows()):
        rows.append({
            "home": row["home"],
            "away": row["away"],
            "xg_home": round(float(xg_h[i]), 3),
            "xg_away": round(float(xg_a[i]), 3),
            "xg_home_raw": round(float(xg_h_raw[i]), 3),
            "xg_away_raw": round(float(xg_a_raw[i]), 3),
            "xg_home_q10": round(float(xg_h_q10[i]), 3),
            "xg_home_q90": round(float(xg_h_q90[i]), 3),
            "xg_away_q10": round(float(xg_a_q10[i]), 3),
            "xg_away_q90": round(float(xg_a_q90[i]), 3),
            "away_regime_v5": int(reg_ids_next[i]),
            "dc_ref_home": round(float(np.exp(
                pack["attack_r"].get(row["home"], 0.0) - pack["defense_r"].get(row["away"], 0.0)
                + pack["team_ha"].get(row["home"], pack["base_home_log"])
            )), 3),
            "dc_ref_away": round(float(np.exp(
                pack["attack_r"].get(row["away"], 0.0) - pack["defense_r"].get(row["home"], 0.0)
                + pack["base_away_log"]
            )), 3),
            "home_adv_fator": round(float(np.exp(
                pack["team_ha"].get(row["home"], pack["base_home_log"]) - pack["base_away_log"]
            )), 3),
        })
    df_pred = pd.DataFrame(rows)

    print(f"\n  {'Mandante':<24} {'Visitante':<24} {'xG_M':>6} {'IC80_M':>14} {'xG_V':>6} {'IC80_V':>14} {'HA':>6}")
    print(f"  {SEP2}")
    for _, r in df_pred.iterrows():
        ic_h = f"[{r['xg_home_q10']:.2f},{r['xg_home_q90']:.2f}]"
        ic_a = f"[{r['xg_away_q10']:.2f},{r['xg_away_q90']:.2f}]"
        print(
            f"  {r['home']:<24} {r['away']:<24} {r['xg_home']:>6.3f} {ic_h:>14} "
            f"{r['xg_away']:>6.3f} {ic_a:>14} {r['home_adv_fator']:>6.3f}x"
        )

    return {
        "df_pred": df_pred,
        "df_bt": df_bt,
        "oof_df": oof_df,
        "bias_corrector": bias_corrector,
        "model_name": f"{mh_name}+{away_bundle["base_name"]}+{away_bundle["resid_name"]}",
        "feat_cols": feat_union,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Modelo oficial de previsão de xG com benchmark interno.")
    parser.add_argument("--mode", choices=["predict", "benchmark", "all"], default="predict")
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--benchmark-prefix", default=BENCHMARK_PREFIX_DEFAULT)
    parser.add_argument("--n-folds", type=int, default=BACKTEST_N_FOLDS)
    parser.add_argument("--min-train", type=int, default=BACKTEST_MIN_TRAIN)
    parser.add_argument("--passadas", default=PASSADAS_FILE)
    parser.add_argument("--atuais", default=ATUAIS_FILE)
    parser.add_argument("--proxima", default=PROXIMA_FILE)
    return parser.parse_args()


def main():
    args = parse_args()
    banner(f"MODELO PREDITIVO DE xG — VERSÃO OFICIAL ({MODEL_VERSION})")

    _, _, proxima, hist = load_input_frames(args.passadas, args.atuais, args.proxima)
    pack = build_feature_pack(hist, proxima)

    benchmark_result = None
    official_result = None

    if args.mode in {"predict", "all"}:
        official_result = run_official_prediction_pipeline(pack, n_folds=args.n_folds, min_train=args.min_train)

    if args.mode in {"benchmark", "all"}:
        benchmark_result = run_native_benchmark_suite(
            pack["hist_f"],
            output_prefix=args.benchmark_prefix,
            n_folds=args.n_folds,
            min_train=args.min_train,
            official_cache=None,
        )

    if official_result is not None:
        info = build_info_frame(
            official_result["model_name"],
            pack["hist_f"],
            pack["feat_cols"],
            official_result["df_bt"],
            official_result["bias_corrector"],
            benchmark_summary=(benchmark_result["summary"] if benchmark_result is not None else None),
            promotion_decision=(benchmark_result["promotion_decision"] if benchmark_result is not None else None),
        )
        export_prediction_workbook(
            args.output,
            official_result["df_pred"],
            official_result["df_bt"],
            pack["ranking"],
            info,
            benchmark_summary=(benchmark_result["summary"] if benchmark_result is not None else None),
        )
        bt_mae = official_result["df_bt"]["mae_mean"].mean() if not official_result["df_bt"].empty else np.nan
        bt_bias = official_result["df_bt"]["bias_home"].mean() if not official_result["df_bt"].empty else np.nan
        print(f"\n{SEP}")
        print(f"  ✅ '{args.output}' gerado!")
        abas = "previsoes_xg | backtest_xg | ranking | info_modelo"
        if benchmark_result is not None and benchmark_result.get("summary") is not None and not benchmark_result["summary"].empty:
            abas += " | benchmark_resumo"
        print(f"     Abas: {abas}")
        print(f"     MAE xG backtest: {bt_mae:.4f} | Bias_h: {bt_bias:+.4f}")
        print(
            f"     Correção isotônica: bias {official_result['bias_corrector'].bias_before_h:+.4f} "
            f"→ {official_result['bias_corrector'].bias_after_h:+.4f} | "
            f"MAE {official_result['bias_corrector'].mae_before_h:.4f} "
            f"→ {official_result['bias_corrector'].mae_after_h:.4f}"
        )
        if benchmark_result is not None:
            dec = benchmark_result.get("promotion_decision", {})
            print(f"     Gate challenger: {'PROMOVER' if dec.get('promoted') else 'MANTER official'}")
        print(SEP)

    if benchmark_result is not None:
        print("\n  Arquivos do benchmark interno:")
        for k, v in benchmark_result["paths"].items():
            if os.path.exists(v):
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()