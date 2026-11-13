"""Backtest a ridge-regression vote-share model against a "no change" baseline.

Idea: learn how a seat's vote share moves from one election to the next using
the previous shares and simple demographics. We only have two transitions
(2013 -> 2018 and 2018 -> 2021), so we backtest by training on one and testing
on the other.

RESULT (see data/processed/backtest.json): the model does NOT beat the baseline
reliably, so simulate.py does not use it. This script is kept as the evidence for
that choice, and its residual spread (15-17pp) informed the noise level in simulate.py.

Output: data/processed/backtest.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

OUT = Path("data/processed")
BLOCS = ["BN", "PH", "PN"]
DEMOGRAPHICS = ["malay", "chinese", "urban", "age21_29"]
PERCENT_COLS = ["malay", "chinese", "age21_29"]  # stored as 0-100, scaled to 0-1 before fitting
FEATURES = [f"prev_{b}" for b in BLOCS] + DEMOGRAPHICS
RIDGE_ALPHA = 3.0  # strong-ish shrinkage because there are only 56 training rows


def make_transition(feats, prev_year, cur_year=None):
    """One row per seat: previous-election shares + demographics (+ next-election shares as targets)."""
    df = feats[["seat", "name"] + DEMOGRAPHICS].copy()
    for b in BLOCS:
        df[f"prev_{b}"] = feats[f"{b}_{prev_year}"]
        if cur_year is not None:
            df[f"y_{b}"] = feats[f"{b}_{cur_year}"]
    return df


def scaled(df):
    X = df[FEATURES].copy()
    X[PERCENT_COLS] /= 100
    return X


def fit(train):
    """One ridge regression per bloc."""
    X = scaled(train)
    return {b: Ridge(alpha=RIDGE_ALPHA).fit(X, train[f"y_{b}"]) for b in BLOCS}


def predict(models, df):
    """Predicted shares, floored at 1% and renormalised to sum to 1 across the three blocs."""
    X = scaled(df)
    pred = pd.DataFrame({b: models[b].predict(X) for b in BLOCS}, index=df.index).clip(lower=0.01)
    return pred.div(pred.sum(axis=1), axis=0)


def normalise(df, prefix):
    """Pull `prefix`+bloc columns out as a bloc-named frame whose rows sum to 1."""
    shares = df[[f"{prefix}{b}" for b in BLOCS]].set_axis(BLOCS, axis=1)
    return shares.div(shares.sum(axis=1), axis=0)


def score(train, test):
    pred = predict(fit(train), test)
    actual = normalise(test, "y_")
    baseline = normalise(test, "prev_")  # "nothing changes since last election"
    winner = actual.idxmax(axis=1)
    return {
        "model_MAE_pp": round(float((pred - actual).abs().mean().mean() * 100), 2),
        "baseline_MAE_pp": round(float((baseline - actual).abs().mean().mean() * 100), 2),
        "model_winner_acc": round(float((pred.idxmax(axis=1) == winner).mean()), 3),
        "baseline_winner_acc": round(float((baseline.idxmax(axis=1) == winner).mean()), 3),
        "resid_sd_pp": round(float((pred - actual).stack().std() * 100), 2),
    }


def main():
    feats = pd.read_csv(OUT / "seats_features.csv")
    t_2013_18 = make_transition(feats, 2013, 2018)
    t_2018_21 = make_transition(feats, 2018, 2021)

    report = {
        "train 2013->18, test 2018->21": score(t_2013_18, t_2018_21),
        "train 2018->21, test 2013->18": score(t_2018_21, t_2013_18),
    }
    (OUT / "backtest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
