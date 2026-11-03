"""
Step 10: COVID-aware versions of the ML models (ridge_cx, gbm_cx).

THESE VARIANTS WERE ADDED AFTER SEEING THE STEP 08/09 RESULTS. The COVID window below
(Mar 2020 - May 2021) is a hindsight choice, so report them as such in the write-up.

What changes compared with step 08: when training, any row that touches the COVID
window is left out. A training row (origin s, horizon h) is dropped if any of these
months falls in the window:
    * the target month (s+h)
    * any month from s-14 to s  (this covers the year-on-year features, the base month
      and the 12-month average the features are built from)
Everything else is identical to step 08 (same features, same fixed model settings,
same walk-forward rules, nothing tuned).

What this CANNOT fix: forecasts made from origins in 2021-22 still have to use the
latest months as inputs, and those are COVID-affected.

Built-in consistency check: for origins up to Feb 2020 no training row can touch the
window, so ridge_cx must give exactly the same forecasts as ridge, and gbm_cx the same
as gbm. The script checks this against the step 08 results.

Run steps 07 and 08 first. Then, from the project folder with the virtual environment active:
    python scripts\10_models_ml_covidaware.py
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
MIN_ROWS = 24  # refuse to fit on fewer training rows than this
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-05-01")

PREV_STATS = bt.OUT_DIR / "backtest_models_v2_type1.csv"
PREV_ML = bt.OUT_DIR / "backtest_ml_type1.csv"
SHOW = ["snaive", "ets_mul", "sarima_masked", "ridge", "ridge_cx", "gbm", "gbm_cx"]
SHOW_GROUPS = ["ets_mul", "sarima_masked", "ridge", "ridge_cx", "gbm", "gbm_cx"]


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


for f in (PREV_STATS, PREV_ML):
    if not f.exists():
        raise SystemExit(f"Cannot find {f}. Run steps 07 and 08 first.")

tbl = bt.load_table()
y = tbl[TARGET]
n_origins = int((y.index >= pd.Timestamp("2015-05-01")).sum())

raw = {
    "ridge_cx": ml_method(make_ridge, True),
    "gbm_cx": ml_method(make_gbm, True),
}

# ---- quick check: how many training rows survive, plus one timed fit on all the data
print(f"Target: {TARGET}  |  origins: {n_origins}")
bad_all = np.asarray((y.index >= COVID_START) & (y.index <= COVID_END))
print("Training rows available (all data, i.e. the latest origin):")
for h in (1, 6, 12):
    keep = len(training_positions(len(y), h, bad_all, True))
    allrows = len(training_positions(len(y), h, bad_all, False))
    print(f"  horizon {h:>2}: step 08 used {allrows}  |  COVID-aware uses {keep}  (left out: {allrows - keep})")

print("Quick check: one 12-month forecast from each model using all the data")
total_seconds = 0.0
for name, fn in raw.items():
    fn(y, 12)  # warm-up
    t0 = time.time()
    fc = fn(y, 12)
    seconds = time.time() - t0
    total_seconds += seconds
    print(f"  {name:<9} {seconds:5.1f}s   next month {fc.iloc[0]:>12,.0f}   in 12 months {fc.iloc[-1]:>12,.0f}")
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
res.to_csv(bt.OUT_DIR / "backtest_ml_cx_type1.csv", index=False)

# ---- consistency check against step 08
ml_prev = pd.read_csv(PREV_ML, parse_dates=["origin", "target"])
print("\n=== CONSISTENCY CHECK (origins up to Feb 2020 must match step 08 exactly) ===")
for new, old in [("ridge_cx", "ridge"), ("gbm_cx", "gbm")]:
    a = res[res["method"] == new][["origin", "target", "horizon", "forecast"]]
    b = ml_prev[ml_prev["method"] == old][["origin", "target", "horizon", "forecast"]]
    m = a.merge(b, on=["origin", "target", "horizon"], suffixes=("_cx", "_old"))
    rel = (m["forecast_cx"] - m["forecast_old"]).abs() / m["forecast_old"]
    early = rel[m["origin"] <= pd.Timestamp("2020-02-01")]
    later = rel[m["origin"] > pd.Timestamp("2020-02-01")]
    status = "OK" if early.max() < 1e-6 else "MISMATCH - tell Claude"
    print(f"  {new:<9} early origins: max relative difference {early.max():.2e} over {len(early):,} forecasts -> {status}")
    print(f"  {'':<9} later origins: {(later > 1e-9).mean() * 100:.0f}% of {len(later):,} forecasts changed")

# ---- comparison with everything else
prev = pd.read_csv(PREV_STATS, parse_dates=["origin", "target"])
everything = pd.concat([prev, ml_prev, res], ignore_index=True)
everything = everything[everything["method"].isin(SHOW)]

summary = bt.summarise(everything)
bt.print_tables(summary, SHOW)

mape = summary.pivot_table(index=["horizon", "phase"], columns="method", values="mape", observed=True)[SHOW]
ratio = mape.div(mape["snaive"], axis=0)
print("\n=== MAPE relative to seasonal naive (below 1.00 = better than 'same month last year') ===")
print(ratio.to_string(float_format=lambda x: f"{x:,.2f}"))

# ---- the step 09 split, now including the COVID-aware variants
settled = everything[everything["phase"] == "settled"].copy()


def origin_group(origin):
    if origin < pd.Timestamp("2022-06-01"):
        return "a) origin up to May 2022"
    if origin < pd.Timestamp("2023-06-01"):
        return "b) origin Jun 2022 - May 2023"
    return "c) origin Jun 2023 on"


settled["origin_group"] = settled["origin"].map(origin_group)
sub = settled[settled["horizon"].isin([1, 3, 6, 12])]
grp = sub.groupby(["horizon", "origin_group", "method"])["ape"].mean().reset_index()
print("\n=== MAPE (%) in the settled period, split by WHEN the forecast was made ===")
print(grp.pivot_table(index=["horizon", "origin_group"], columns="method", values="ape")[SHOW_GROUPS]
      .to_string(float_format=lambda x: f"{x:,.1f}"))
print("(forecast counts per row are the same as in step 09: 1-6 / 12 / 12 / 28-39)")
print(f"\nSaved COVID-aware forecasts: {bt.OUT_DIR / 'backtest_ml_cx_type1.csv'}")
