"""
Shared model code for the NHS A&E forecasting project.

These are the SAME models that were backtested in steps 07 (ETS, masked SARIMA) and
10 (COVID-aware ridge), gathered in one importable place so that the final forecast
(step 13) and later the API use exactly the tested code. Step 13 proves this by
re-running one past forecast and comparing it with the saved backtest.

Every model has the shape  model(hist, max_h) -> pandas Series  where `hist` is the
monthly series up to "now" and the result holds the next max_h months.

Nothing here runs when imported; statsmodels is only loaded when a model is used.
"""

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import ae_backtest as bt

COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-05-01")
MIN_POS = 14  # earliest origin position with all lag features available
MIN_ROWS = 24  # refuse to fit the ML models on fewer training rows than this

nonconverged = []  # (model name, origin) for SARIMA fits whose optimiser did not converge


def _out(hist, max_h, values):
    return pd.Series(np.asarray(values, dtype=float), index=bt.future_index(hist, max_h))


# ---------------------------------------------------------------- statistical models
def ets(seasonal):
    """Holt-Winters with a damped additive trend; seasonal is 'add' or 'mul'."""

    def run(hist, max_h):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        model = ExponentialSmoothing(
            hist.to_numpy(dtype=float),
            trend="add",
            damped_trend=True,
            seasonal=seasonal,
            seasonal_periods=12,
            initialization_method="estimated",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit()
        return _out(hist, max_h, fit.forecast(max_h))

    return run


def sarima(mask_covid, name):
    """SARIMA(1,1,1)(0,1,1)12 on log(attendances); optionally treats the COVID months as missing."""

    def run(hist, max_h):
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        z = np.log(hist.astype(float))
        if mask_covid:
            in_window = (z.index >= COVID_START) & (z.index <= COVID_END)
            z = z.where(~in_window)  # these months become NaN (missing)
        model = SARIMAX(
            z.to_numpy(),
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 12),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(disp=False, maxiter=300)
        if not fit.mle_retvals.get("converged", True):
            nonconverged.append((name, hist.index[-1]))
        return _out(hist, max_h, np.exp(fit.forecast(max_h)))

    return run


# ---------------------------------------------------------------- machine-learning models
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


def training_positions(n, h, bad, exclude_covid):
    """Origin positions usable for training at horizon h (targets already known)."""
    positions = []
    for s in range(MIN_POS, n - h):
        if exclude_covid and (bad[s + h] or bad[s - 14 : s + 1].any()):
            continue
        positions.append(s)
    return positions


def ml_method(make_model, exclude_covid):
    """Direct multi-horizon model of the log year-on-year change; see step 08 and step 10."""

    def run(hist, max_h):
        logy = np.log(hist.to_numpy(dtype=float))
        n = len(logy)
        first = hist.index[0]
        bad = np.asarray((hist.index >= COVID_START) & (hist.index <= COVID_END))
        out = []
        for h in range(1, max_h + 1):
            positions = training_positions(n, h, bad, exclude_covid)
            if len(positions) < MIN_ROWS:
                raise ValueError(f"only {len(positions)} training rows at horizon {h}")
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


# ---------------------------------------------------------------- the three models we ship
def shipped_models():
    """The three models that are averaged into the final forecast ('combo3')."""
    return {
        "ets_mul": ets("mul"),
        "sarima_masked": sarima(True, "sarima_masked"),
        "ridge_cx": ml_method(make_ridge, True),
    }
