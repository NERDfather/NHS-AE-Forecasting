"""
Step 05: baseline forecasts + walk-forward backtest (Type 1 attendances).

Walk-forward means: at each forecast origin (a month), we pretend that month is
"now", use ONLY data up to and including it, forecast the next 1-12 months,
then score against what really happened. Then we move the origin forward one
month and repeat. Nothing from the future is ever used.

Baselines (no model fitting, so nothing to overfit):
  naive           next months = the latest month
  snaive          same month last year
  snaive_growth   same month last year x (latest 12 months / the 12 months before)

Only months from June 2015 (real monthly collection) are ever scored.
Errors are reported separately for pre-COVID, COVID and post-COVID months,
plus 'non_covid' (pre + post together).

Run from the project folder, with the virtual environment active:
    python scripts\05_baselines_backtest.py
"""

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "data" / "processed" / "ae_model_table.csv"
OUT_DIR = PROJECT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "type1"
HORIZONS = range(1, 13)
SHOW_HORIZONS = [1, 3, 6, 12]
FIRST_ORIGIN = pd.Timestamp("2015-05-01")  # first target month is then Jun 2015

pd.set_option("display.width", 200)
pd.options.display.float_format = "{:,.1f}".format

df = pd.read_csv(SRC, parse_dates=["period"]).sort_values("period").reset_index(drop=True)
y = df.set_index("period")[TARGET]
phase = df.set_index("period")["phase"]
scoreable = df.set_index("period")["scoreable"]

# the series must be a clean monthly sequence
assert (y.index == pd.date_range(y.index[0], y.index[-1], freq="MS")).all(), "months are not continuous"


# ---------------------------------------------------------------- baselines
def f_naive(hist, target):
    return hist.iloc[-1]


def f_snaive(hist, target):
    return hist.loc[target - pd.DateOffset(months=12)]


def f_snaive_growth(hist, target):
    base = hist.loc[target - pd.DateOffset(months=12)]
    last12 = hist.iloc[-12:].mean()
    prev12 = hist.iloc[-24:-12].mean()
    return base * last12 / prev12


METHODS = {"naive": f_naive, "snaive": f_snaive, "snaive_growth": f_snaive_growth}

# ---------------------------------------------------------------- walk-forward
rows = []
for origin in y.index[y.index >= FIRST_ORIGIN]:
    hist = y.loc[:origin]  # ONLY data up to the origin
    for h in HORIZONS:
        target = origin + pd.DateOffset(months=h)
        if target not in y.index or not scoreable[target]:
            continue
        for name, fn in METHODS.items():
            rows.append(
                {
                    "origin": origin,
                    "target": target,
                    "horizon": h,
                    "method": name,
                    "actual": y[target],
                    "forecast": fn(hist, target),
                    "phase": phase[target],
                }
            )

res = pd.DataFrame(rows)
res["error"] = res["forecast"] - res["actual"]
res["abs_err"] = res["error"].abs()
res["ape"] = res["abs_err"] / res["actual"] * 100
res.to_csv(OUT_DIR / "backtest_baselines_type1.csv", index=False)

print("=== BACKTEST SETUP ===")
print(f"Target: {TARGET}  |  forecast rows: {len(res):,}  |  methods: {', '.join(METHODS)}")
print(f"Origins: {res['origin'].min():%b %Y} to {res['origin'].max():%b %Y}")
print(f"Targets scored: {res['target'].min():%b %Y} to {res['target'].max():%b %Y}")

# a worked example so the numbers are not a black box
ex = res[(res["origin"] == "2026-08-01") & (res["horizon"] == 1)]
print("\nWorked example: standing in Aug 2026, forecast Sep 2026 (1 month ahead)")
print(f"  actual Sep 2026 = {ex['actual'].iloc[0]:,.0f}")
for _, r in ex.iterrows():
    print(f"  {r['method']:<14} forecast {r['forecast']:>12,.0f}   error {r['error']:>+10,.0f}  ({r['ape']:.1f}%)")

# ---------------------------------------------------------------- summaries
nc = res[res["phase"] != "covid"].copy()
nc["phase"] = "non_covid"
allres = pd.concat([res, nc], ignore_index=True)
order = ["pre_covid", "covid", "post_covid", "non_covid"]
allres["phase"] = pd.Categorical(allres["phase"], order, ordered=True)
sel = allres[allres["horizon"].isin(SHOW_HORIZONS)]

grp = sel.groupby(["horizon", "phase", "method"], observed=True)
summary = grp.agg(n=("abs_err", "size"), mae=("abs_err", "mean"), mape=("ape", "mean")).reset_index()
meth_order = list(METHODS)

print("\n=== MAPE (%): average absolute % error, by horizon (months ahead) and period ===")
print(summary.pivot_table(index=["horizon", "phase"], columns="method", values="mape", observed=True)[meth_order].to_string())

print("\n=== MAE: average absolute error in attendances ===")
print(summary.pivot_table(index=["horizon", "phase"], columns="method", values="mae", observed=True)[meth_order]
      .to_string(float_format=lambda x: f"{x:,.0f}"))

print("\n=== Number of forecasts behind each cell ===")
print(summary[summary["method"] == "naive"].pivot_table(index="horizon", columns="phase", values="n", observed=True)
      .to_string(float_format=lambda x: f"{x:,.0f}"))

print(f"\nSaved all forecasts: {OUT_DIR / 'backtest_baselines_type1.csv'}")
