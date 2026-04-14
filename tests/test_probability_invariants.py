from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    path = Path("colab_pipeline/03_train_predict_markets.py")
    spec = importlib.util.spec_from_file_location("mkt", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_probability_invariants():
    mod = _load_module()

    home_lambda = np.array([0.8, 1.2, 2.0, 1.6], dtype=float)
    away_lambda = np.array([0.7, 1.1, 1.5, 1.8], dtype=float)

    out = mod.market_probabilities(home_lambda, away_lambda, max_goals=8, rho=0.03)

    s_1x2 = out["p_home_win"] + out["p_draw"] + out["p_away_win"]
    assert np.allclose(s_1x2, 1.0, atol=1e-6)

    assert np.allclose(out["p_over_1_5"] + out["p_under_1_5"], 1.0, atol=1e-6)
    assert np.allclose(out["p_over_2_5"] + out["p_under_2_5"], 1.0, atol=1e-6)
    assert np.allclose(out["p_over_3_5"] + out["p_under_3_5"], 1.0, atol=1e-6)

    assert np.allclose(out["p_btts_yes"] + out["p_btts_no"], 1.0, atol=1e-6)

    for k, v in out.items():
        assert np.all(np.isfinite(v)), f"Non-finite in {k}"
        assert np.all((v >= -1e-9) & (v <= 1 + 1e-9)), f"Out of range in {k}"


if __name__ == "__main__":
    test_probability_invariants()
    print("OK - probability invariants")
