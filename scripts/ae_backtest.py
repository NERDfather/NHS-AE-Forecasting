"""
Reusable walk-forward backtest helpers for the NHS A&E forecasting project.

A forecasting method is any function with this shape:
    method(hist, max_h) -> pandas Series
where `hist` is the monthly series up to and including the forecast origin
(nothing later is ever passed in) and the returned Series holds forecasts for
the next `max_h` months, indexed by month start.

Scored months are split into periods by the month being forecast:
    pre_covid   Jun 2015 - Feb 2020
    covid       Mar 2020 - May 2021   (provisional judgement, stated in the write-up)
    recovery    Jun 2021 - May 2022   (same month a year earlier was COVID-affected)
    settled     Jun 2022 onwards
    clean       pre_covid + settled   (the headline comparison)
"""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
MODEL_TABLE = PROJECT / "data" / "processed" / "ae_model_table.csv"
OUT_DIR = PROJECT / "outputs"

PHASES = ["pre_covid", "covid", "recovery", "settled", "clean"]


def load_table():
    """Load the model-ready table, indexed by month, and check months are continuous."""
    df = pd.read_csv(MODEL_TABLE, parse_dates=["period"]).sort_values("period").reset_index(drop=True)
    idx = pd.DatetimeIndex(df["period"])
    assert (idx == pd.date_range(idx[0], idx[-1], freq="MS")).all(), "months are not continuous"
    return df.set_index("period")


def label_phase(target):
    if target < pd.Timestamp("2020-03-01"):
        return "pre_covid"
    if target < pd.Timestamp("2021-06-01"):
        return "covid"
    if target < pd.Timestamp("2022-06-01"):
        return "recovery"
    return "settled"


def future_index(hist, max_h):
    start = hist.index[-1] + pd.DateOffset(months=1)
    return pd.date_range(start, periods=max_h, freq="MS")


# ---------------------------------------------------------------- baselines
def naive(hist, max_h):
    """Next months = the latest month."""
    return pd.Series(hist.iloc[-1], index=future_index(hist, max_h))


def snaive(hist, max_h):
    """Same month last year."""
    assert max_h <= 12, "seasonal naive here only supports horizons up to 12"
    idx = future_index(hist, max_h)
    return pd.Series([hist.loc[t - pd.DateOffset(months=12)] for t in idx], index=idx)


def snaive_growth(hist, max_h):
    """Same month last year, scaled by (latest 12 months) / (the 12 months before)."""
    base = snaive(hist, max_h)
    return base * hist.iloc[-12:].mean() / hist.iloc[-24:-12].mean()


BASELINES = {"naive": naive, "snaive": snaive, "snaive_growth": snaive_growth}


# ---------------------------------------------------------------- walk-forward
def walk_forward(y, methods, scoreable, max_h=12, first_origin="2015-05-01"):
    """
    For every origin month, hand each method ONLY the data up to that month,
    collect its 1..max_h month forecasts, and keep those whose target month is scoreable.
    """
    rows = []
    for origin in y.index[y.index >= pd.Timestamp(first_origin)]:
        hist = y.loc[:origin]
        for name, fn in methods.items():
            fc = fn(hist, max_h)
            for h, (target, value) in enumerate(fc.items(), start=1):
                if target not in y.index or not scoreable[target]:
                    continue
                rows.append(
                    {
                        "origin": origin,
                        "target": target,
                        "horizon": h,
                        "method": name,
                        "actual": y[target],
                        "forecast": value,
                    }
                )
    res = pd.DataFrame(rows)
    res["error"] = res["forecast"] - res["actual"]
    res["abs_err"] = res["error"].abs()
    res["ape"] = res["abs_err"] / res["actual"] * 100
    res["phase"] = res["target"].map(label_phase)
    return res


def summarise(res, horizons=(1, 3, 6, 12)):
    """Average errors by horizon, period and method (adds the combined 'clean' period)."""
    clean = res[res["phase"].isin(["pre_covid", "settled"])].copy()
    clean["phase"] = "clean"
    both = pd.concat([res, clean], ignore_index=True)
    both["phase"] = pd.Categorical(both["phase"], PHASES, ordered=True)
    sel = both[both["horizon"].isin(horizons)]
    return (
        sel.groupby(["horizon", "phase", "method"], observed=True)
        .agg(n=("abs_err", "size"), mae=("abs_err", "mean"), mape=("ape", "mean"), bias=("error", "mean"))
        .reset_index()
    )


def print_tables(summary, methods):
    order = list(methods)
    print("\n=== MAPE (%): average absolute % error, by horizon (months ahead) and period ===")
    print(summary.pivot_table(index=["horizon", "phase"], columns="method", values="mape", observed=True)[order]
          .to_string(float_format=lambda x: f"{x:,.1f}"))
    print("\n=== MAE: average absolute error in attendances ===")
    print(summary.pivot_table(index=["horizon", "phase"], columns="method", values="mae", observed=True)[order]
          .to_string(float_format=lambda x: f"{x:,.0f}"))
    print("\n=== Number of forecasts behind each cell ===")
    print(summary[summary["method"] == order[0]]
          .pivot_table(index="horizon", columns="phase", values="n", observed=True)
          .to_string(float_format=lambda x: f"{x:,.0f}"))
