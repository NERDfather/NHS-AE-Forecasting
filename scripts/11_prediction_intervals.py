"""
Step 11 (v2): prediction intervals with an honest coverage check (Type 1).

Idea: build each interval from the model's OWN past errors, then check how often the
real value landed inside. No new models are fitted: this reads the forecasts saved by
steps 07, 08 and 10.

How an interval is built (for one model, one horizon h, one forecast made at origin o):
  1. Look only at that model's earlier forecasts at the same horizon whose TARGET month
     had already happened by o (target <= o). Nothing from after o is used.
  2. Take the size of each of those errors on the log scale: |log(actual / forecast)|.
  3. Take the k-th smallest of them, where k = ceil((n + 1) x level).
     (This is the finite-sample rule used in conformal prediction. With fewer than
     24 past errors, or if k > n, no interval is made.)
  4. interval = forecast x exp(-q)  to  forecast x exp(+q)

Three versions of the error history:
  all        every past error, including the huge COVID-era ones
  no_covid   errors whose target month fell in Mar 2020 - May 2021 are left out
  stable     errors whose target month fell in Mar 2020 - May 2022 are left out
             (COVID plus the recovery year, when "same month last year" was itself
             COVID-affected). ADDED AFTER SEEING THE v1 RESULTS; a hindsight choice.

Forecasts that are zero or negative (ETS produced 10 of these after the April 2020
collapse) get no interval, are not used as calibration errors, and are counted below.

Then: coverage = share of real values that fell inside the interval.
An 80% interval should cover about 80%, a 95% interval about 95%.
Expect trouble during COVID, when errors suddenly became far larger than any history.

Run steps 07, 08 and 10 first. Then, from the project folder with the virtual environment active:
    python scripts\11_prediction_intervals.py
"""

import numpy as np
import pandas as pd

import ae_backtest as bt

METHODS = ["snaive", "ets_mul", "sarima_masked", "ridge_cx"]
LEVELS = [0.80, 0.95]
HORIZONS = [1, 3, 6, 12]
VARIANTS = ["all", "no_covid", "stable"]
PRINT_VARIANTS = ["no_covid", "stable"]  # 'all' is saved, and shown in the width table only
MIN_CAL = 24
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2021-05-01")
STABLE_END = pd.Timestamp("2022-05-01")
EXCLUDE_END = {"all": None, "no_covid": COVID_END, "stable": STABLE_END}

FILES = [
    bt.OUT_DIR / "backtest_models_v2_type1.csv",
    bt.OUT_DIR / "backtest_ml_cx_type1.csv",
]


def conformal_q(abs_err, level):
    """k-th smallest past error, k = ceil((n+1) x level); NaN if there is too little history."""
    n = len(abs_err)
    k = int(np.ceil((n + 1) * level))
    if n < MIN_CAL or k > n:
        return np.nan
    return float(np.sort(abs_err)[k - 1])


def make_intervals(sub, variant, level):
    """
    `sub`: one model's forecasts at one horizon (columns origin, target, forecast, actual),
    sorted by target. Returns the same rows with q, lo, hi, covered added.
    Rows whose forecast (or actual) is not positive get no interval.
    """
    targets = sub["target"].to_numpy()
    origins = sub["origin"].to_numpy()
    actual = sub["actual"].to_numpy(dtype=float)
    forecast = sub["forecast"].to_numpy(dtype=float)

    valid = (forecast > 0) & (actual > 0)
    ratio = np.where(valid, actual / np.where(valid, forecast, 1.0), 1.0)
    abs_err = np.where(valid, np.abs(np.log(ratio)), np.nan)

    end = EXCLUDE_END[variant]
    if end is None:
        excluded = np.zeros(len(sub), dtype=bool)
    else:
        excluded = ((sub["target"] >= COVID_START) & (sub["target"] <= end)).to_numpy()

    q = np.full(len(sub), np.nan)
    for i in range(len(sub)):
        if not valid[i]:
            continue
        k = np.searchsorted(targets, origins[i], side="right")  # forecasts whose target <= origin
        cal = abs_err[:k]
        cal = cal[~excluded[:k] & ~np.isnan(cal)]
        q[i] = conformal_q(cal, level)

    out = sub.copy()
    out["q"] = q
    out["lo"] = out["forecast"] * np.exp(-q)
    out["hi"] = out["forecast"] * np.exp(q)
    out["covered"] = np.where(np.isnan(q), np.nan, (out["actual"] >= out["lo"]) & (out["actual"] <= out["hi"]))
    out["rel_width"] = (out["hi"] - out["lo"]) / out["forecast"]
    return out


def run_all(res):
    parts = []
    for method in METHODS:
        for h in HORIZONS:
            sub = res[(res["method"] == method) & (res["horizon"] == h)].sort_values("target").reset_index(drop=True)
            for variant in VARIANTS:
                for level in LEVELS:
                    iv = make_intervals(sub, variant, level)
                    iv["method"], iv["variant"], iv["level"], iv["horizon"] = method, variant, level, h
                    parts.append(iv)
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    for f in FILES:
        if not f.exists():
            raise SystemExit(f"Cannot find {f}. Run steps 07, 08 and 10 first.")

    res = pd.concat([pd.read_csv(f, parse_dates=["origin", "target"]) for f in FILES], ignore_index=True)
    res = res[res["method"].isin(METHODS)]
    missing = set(METHODS) - set(res["method"].unique())
    assert not missing, f"missing methods in the saved forecasts: {missing}"

    bad = res[~(res["forecast"] > 0)]
    print(f"Forecasts that are zero or negative (no interval is made for them): {len(bad)}")
    if len(bad):
        print(bad.groupby(["method", "horizon"]).size().to_string())
        print(f"  origins: {bad['origin'].min():%b %Y} to {bad['origin'].max():%b %Y}  |  "
              f"targets: {bad['target'].min():%b %Y} to {bad['target'].max():%b %Y}")

    iv = run_all(res)
    iv.to_csv(bt.OUT_DIR / "intervals_empirical_type1.csv", index=False)
    valid = iv.dropna(subset=["q"]).copy()
    print(f"\nInterval rows: {len(iv):,}  |  with an interval: {len(valid):,}")

    clean = valid[valid["phase"].isin(["pre_covid", "settled"])].copy()
    clean["phase"] = "clean"
    both = pd.concat([valid, clean], ignore_index=True)
    PHASES = ["pre_covid", "covid", "recovery", "settled", "clean"]
    both["phase"] = pd.Categorical(both["phase"], PHASES, ordered=True)

    cover = (both.groupby(["variant", "level", "horizon", "phase", "method"], observed=True)
             .agg(coverage=("covered", "mean"), n=("covered", "size"), width=("rel_width", "median"))
             .reset_index())
    cover["coverage"] *= 100
    cover["width"] *= 100

    for variant in PRINT_VARIANTS:
        for level in LEVELS:
            t = cover[(cover["variant"] == variant) & (cover["level"] == level)]
            print(f"\n=== COVERAGE (%) of the {int(level * 100)}% interval | error history: {variant} "
                  f"(nominal = {int(level * 100)}) ===")
            print(t.pivot_table(index=["horizon", "phase"], columns="method", values="coverage", observed=True)[METHODS]
                  .to_string(float_format=lambda x: f"{x:,.0f}"))

    print("\n=== MEDIAN INTERVAL WIDTH (% of the forecast), clean periods (pre-COVID + settled) ===")
    w = cover[cover["phase"] == "clean"]
    print(w.pivot_table(index=["variant", "level", "horizon"], columns="method", values="width", observed=True)[METHODS]
          .to_string(float_format=lambda x: f"{x:,.1f}"))

    c = cover[(cover["variant"] == "stable") & (cover["level"] == 0.80) & (cover["method"] == "snaive")]
    print("\nNumber of intervals scored per cell (same for every model and level except where noted):")
    print(c.pivot_table(index="horizon", columns="phase", values="n", observed=True)
          .to_string(float_format=lambda x: f"{x:,.0f}"))

    print(f"\nSaved all intervals: {bt.OUT_DIR / 'intervals_empirical_type1.csv'}")
