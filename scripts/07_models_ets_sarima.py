"""
Step 07: add classical statistical models to the walk-forward backtest (Type 1).

Models (each is refitted from scratch at every forecast origin, using only data up to it):
  ets_add        Holt-Winters: damped trend + ADDITIVE seasonality
  ets_mul        Holt-Winters: damped trend + MULTIPLICATIVE seasonality
  sarima         SARIMA(1,1,1)(0,1,1)12 fitted on log(attendances)
  sarima_masked  same, but the COVID months (Mar 2020 - May 2021) are treated as MISSING
                 when fitting, so the model does not try to learn from the pandemic.
Reference points: naive and snaive (same month last year).

Caveat to state in the write-up: the COVID window used for masking was chosen with
hindsight. A live system would need a rule for spotting anomalous months.

Expect this to take several minutes (it fits roughly 550 models).

Version 2: SARIMA fits may use up to 300 optimiser iterations (the default of 50 was hit by
about 1 in 5 fits in version 1), and fits that still do not converge are counted and saved.

Run from the project folder, with the virtual environment active:
    python scripts\07_models_ets_sarima.py
"""

import time
import warnings

import numpy as np
import pandas as pd

import ae_backtest as bt

# statsmodels prints many convergence notes; we count real failures instead
warnings.filterwarnings("ignore")

TARGET = "type1"
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-05-01")


def _out(hist, max_h, values):
    return pd.Series(np.asarray(values, dtype=float), index=bt.future_index(hist, max_h))


def ets(seasonal):
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


nonconverged = []  # (model name, origin) for SARIMA fits that hit the iteration limit


def sarima(mask_covid, name):
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
            fit = model.fit(disp=False, maxiter=300)  # default is 50; about 1 in 5 fits hit that limit
        if not fit.mle_retvals.get("converged", True):
            nonconverged.append((name, hist.index[-1]))
        return _out(hist, max_h, np.exp(fit.forecast(max_h)))

    return run


failures = {}


def guarded(name, fn, total):
    """Wrap a method: print progress, and on an error record it and return NaN forecasts."""
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

raw_models = {
    "ets_add": ets("add"),
    "ets_mul": ets("mul"),
    "sarima": sarima(False, "sarima"),
    "sarima_masked": sarima(True, "sarima_masked"),
}

# ---- fail-fast check: fit each model ONCE on all the data, time it, and look at the forecast.
# This is deliberately unguarded, so a real problem stops us here with a clear error
# instead of after several minutes of fitting.
print(f"Target: {TARGET}  |  origins: {n_origins}")
print("Quick check: one fit of each model on all the data (this also times it)")
total_seconds = 0.0
for name, fn in raw_models.items():
    fn(y, 12)  # warm-up: the first call includes one-off start-up time
    t0 = time.time()
    fc = fn(y, 12)
    seconds = time.time() - t0
    total_seconds += seconds
    print(f"  {name:<14} {seconds:5.1f}s   next month {fc.iloc[0]:>12,.0f}   in 12 months {fc.iloc[-1]:>12,.0f}")
    assert np.isfinite(fc.to_numpy()).all(), f"{name} produced non-finite forecasts"
    assert fc.between(0.5 * y.iloc[-1], 2.0 * y.iloc[-1]).all(), f"{name} produced wild forecasts"
print(f"  Latest actual month ({y.index[-1]:%b %Y}): {y.iloc[-1]:,.0f}")
print(f"  Rough total time for the full test: about {total_seconds * n_origins / 60:.0f} minutes")
nonconverged.clear()  # the quick check above does not count towards the backtest

methods = {
    "naive": bt.naive,
    "snaive": bt.snaive,
    "ets_add": guarded("ets_add", raw_models["ets_add"], n_origins),
    "ets_mul": guarded("ets_mul", raw_models["ets_mul"], n_origins),
    "sarima": guarded("sarima", raw_models["sarima"], n_origins),
    "sarima_masked": guarded("sarima_masked", raw_models["sarima_masked"], n_origins),
}

print(f"\nMethods: {', '.join(methods)}")
print("Fitting... (progress lines appear below)")
res = bt.walk_forward(y, methods, tbl["scoreable"])

n_all = len(res)
res = res.dropna(subset=["forecast"])
print(f"\nForecast rows: {n_all:,}  |  dropped because a model failed: {n_all - len(res):,}")
print("Failed fits per model:", {k: v for k, v in failures.items()})

bt.OUT_DIR.mkdir(parents=True, exist_ok=True)
res.to_csv(bt.OUT_DIR / "backtest_models_v2_type1.csv", index=False)

summary = bt.summarise(res)
bt.print_tables(summary, methods)

mape = summary.pivot_table(index=["horizon", "phase"], columns="method", values="mape", observed=True)[list(methods)]
ratio = mape.div(mape["snaive"], axis=0)
print("\n=== MAPE relative to seasonal naive (below 1.00 = better than 'same month last year') ===")
print(ratio.to_string(float_format=lambda x: f"{x:,.2f}"))
nc = pd.DataFrame(nonconverged, columns=["model", "origin"])
print("\n=== SARIMA fits that hit the iteration limit (did not converge) ===")
print(nc.groupby("model").size().reindex(["sarima", "sarima_masked"], fill_value=0).to_string())
print(f"(out of {n_origins} fits per model)")
nc.to_csv(bt.OUT_DIR / "sarima_nonconverged.csv", index=False)

print(f"\nSaved all forecasts: {bt.OUT_DIR / 'backtest_models_v2_type1.csv'}")
