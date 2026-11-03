"""
Step 08: machine-learning models in the same walk-forward backtest (Type 1).

How the ML forecast works (a "direct" multi-horizon approach):
  * For each horizon h = 1..12 a separate model is trained.
  * What it predicts: the log year-on-year change of the target month,
        log(y[t+h]) - log(y[t+h-12]).
    The forecast is then  y[t+h-12] * exp(predicted change),  i.e. "same month last
    year, adjusted by a learned growth rate". Level and trend are handled by that base.
  * Features, all known at the forecast origin t:
        - the latest three year-on-year changes (and their average)
        - where last year's base month sat relative to the latest 12-month average
        - month of the target (as sin/cos)
        - calendar differences between the target month and its base month
          (days, weekdays, Mondays)
  * Training rows: every earlier origin whose target month was already known at t.
    Nothing from after t is ever used. Months Nov 2010 - May 2015 (estimated from
    weekly data) are allowed in training, never in scoring.

Models:
  ridge   standardised features + ridge regression (a simple linear check)
  gbm     histogram gradient boosting with shallow trees and strong regularisation

Settings were fixed in advance and NOT tuned on the backtest results.
No COVID masking here: the ML models see the pandemic months as they were.

Results are compared with the step 07 results saved in outputs/backtest_models_v2_type1.csv,
so run step 07 first.

Run from the project folder, with the virtual environment active:
    python scripts\08_models_ml.py
"""

import time
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ae_backtest as bt

warnings.filterwarnings("ignore")

TARGET = "type1"
MIN_POS = 14  # earliest origin position with all lag features available
PREVIOUS = bt.OUT_DIR / "backtest_models_v2_type1.csv"
SHOW = ["snaive", "ets_mul", "sarima", "sarima_masked", "ridge", "gbm"]


@lru_cache(maxsize=None)
def month_calendar(period):
    """(days, weekdays, mondays) in the month starting at `period`."""
    days = pd.date_range(period, period + pd.offsets.MonthEnd(0))
    dow = days.day_of_week
    return len(days), int((dow < 5).sum()), int((dow == 0).sum())


def features(logy, first, s, h):
    """Feature vector for origin position s and horizon h, using data up to position s only."""

    def yoy(p):
        return logy[p] - logy[p - 12]

    f0, f1, f2 = yoy(s), yoy(s - 1), yoy(s - 2)
    base_pos = s + h - 12
    target_month = first + pd.DateOffset(months=int(s + h))
    base_month = first + pd.DateOffset(months=int(base_pos))
    ct, cb = month_calendar(target_month), month_calendar(base_month)
    m = target_month.month
    return [
        f0,
        f1,
        f2,
        (f0 + f1 + f2) / 3,
        logy[base_pos] - logy[s - 11 : s + 1].mean(),
        np.sin(2 * np.pi * m / 12),
        np.cos(2 * np.pi * m / 12),
        ct[0] - cb[0],
        ct[1] - cb[1],
        ct[2] - cb[2],
    ]


def ml_method(make_model):
    def run(hist, max_h):
        logy = np.log(hist.to_numpy(dtype=float))
        n = len(logy)
        first = hist.index[0]
        out = []
        for h in range(1, max_h + 1):
            positions = range(MIN_POS, n - h)  # targets at s+h <= n-1: already known
            X = np.array([features(logy, first, s, h) for s in positions])
            t = np.array([logy[s + h] - logy[s + h - 12] for s in positions])
            model = make_model()
            model.fit(X, t)
            x_new = np.array([features(logy, first, n - 1, h)])
            base = hist.iloc[n - 1 + h - 12]
            out.append(base * np.exp(model.predict(x_new)[0]))
        return pd.Series(out, index=bt.future_index(hist, max_h))

    return run


def make_ridge():
    return make_pipeline(StandardScaler(), Ridge(alpha=3.0))


def make_gbm():
    return HistGradientBoostingRegressor(
        max_depth=2,
        learning_rate=0.05,
        max_iter=200,
        min_samples_leaf=5,
        l2_regularization=1.0,
        random_state=0,
    )


failures = {}


def guarded(name, fn, total):
    state = {"calls": 0}
    failures[name] = 0

    def run(hist, max_h):
        state["calls"] += 1
        if state["calls"] % 25 == 0:
            print(f"  {name}: {state['calls']}/{total} origins done", flush=True)
        try:
            return fn(hist, max_h)
        except Exception as exc:  # noqa: BLE001
            failures[name] += 1
            if failures[name] == 1:
                print(f"  {name}: first failure at origin {hist.index[-1]:%b %Y}: {type(exc).__name__}: {exc}")
            return pd.Series(np.nan, index=bt.future_index(hist, max_h))

    return run


tbl = bt.load_table()
y = tbl[TARGET]
n_origins = int((y.index >= pd.Timestamp("2015-05-01")).sum())

raw = {"ridge": ml_method(make_ridge), "gbm": ml_method(make_gbm)}

# ---- fail-fast check: one fit of each model on all the data, timed and sanity-checked
print(f"Target: {TARGET}  |  origins: {n_origins}")
print("Quick check: one 12-month forecast from each model using all the data")
total_seconds = 0.0
for name, fn in raw.items():
    fn(y, 12)  # warm-up
    t0 = time.time()
    fc = fn(y, 12)
    seconds = time.time() - t0
    total_seconds += seconds
    print(f"  {name:<6} {seconds:5.1f}s   next month {fc.iloc[0]:>12,.0f}   in 12 months {fc.iloc[-1]:>12,.0f}")
    assert np.isfinite(fc.to_numpy()).all(), f"{name} produced non-finite forecasts"
    assert fc.between(0.5 * y.iloc[-1], 2.0 * y.iloc[-1]).all(), f"{name} produced wild forecasts"
print(f"  Latest actual month ({y.index[-1]:%b %Y}): {y.iloc[-1]:,.0f}")
print(f"  Rough total time for the full test: about {total_seconds * n_origins / 60:.0f} minutes")

methods = {name: guarded(name, fn, n_origins) for name, fn in raw.items()}
print("\nFitting... (progress lines appear below)")
res = bt.walk_forward(y, methods, tbl["scoreable"])

n_all = len(res)
res = res.dropna(subset=["forecast"])
print(f"\nForecast rows: {n_all:,}  |  dropped because a model failed: {n_all - len(res):,}")
print("Failed fits per model:", dict(failures))
bt.OUT_DIR.mkdir(parents=True, exist_ok=True)
res.to_csv(bt.OUT_DIR / "backtest_ml_type1.csv", index=False)

# ---- compare with the step 07 results
if not PREVIOUS.exists():
    raise SystemExit(f"Cannot find {PREVIOUS}. Run step 07 first.")
prev = pd.read_csv(PREVIOUS, parse_dates=["origin", "target"])
everything = pd.concat([prev, res], ignore_index=True)
everything = everything[everything["method"].isin(SHOW)]

summary = bt.summarise(everything)
bt.print_tables(summary, SHOW)

mape = summary.pivot_table(index=["horizon", "phase"], columns="method", values="mape", observed=True)[SHOW]
ratio = mape.div(mape["snaive"], axis=0)
print("\n=== MAPE relative to seasonal naive (below 1.00 = better than 'same month last year') ===")
print(ratio.to_string(float_format=lambda x: f"{x:,.2f}"))
print(f"\nSaved ML forecasts: {bt.OUT_DIR / 'backtest_ml_type1.csv'}")
