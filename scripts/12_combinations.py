"""
Step 12: does averaging models beat picking one? (read-only: fits nothing)

The models disagree about the future: for 12 months ahead, the latest forecasts of the
three models below span about 3% (and about 4.5% if unmasked SARIMA is included), which
is as large as the 80% interval itself. The backtest cannot separate the best ones.
A standard answer is to average them.

Two combinations, DECLARED BEFORE LOOKING at their results, equal weights, nothing tuned:
  combo2   average of ets_mul and sarima_masked
  combo3   average of ets_mul, sarima_masked and ridge_cx   (the one we would ship)

Everything is scored on the SAME forecasts, using only forecast rows where every
component produced a positive number (this drops the 4 negative ETS forecasts).

Decision rule, declared in advance: ship combo3 unless it is worse than the best single
model by more than 0.2 percentage points of average error at THREE OR MORE of the four
horizons (1, 3, 6, 12 months) in the clean periods (pre-COVID + settled).

Then the step 11 interval method (the 'stable' error history) is applied to each model
and each combination, so we can see whether the combination also gives better intervals.

Run steps 07, 08, 10 and 11 first. Then, from the project folder with the virtual environment active:
    python scripts\12_combinations.py
"""

import importlib

import numpy as np
import pandas as pd

import ae_backtest as bt

m11 = importlib.import_module("11_prediction_intervals")

COMPONENTS = ["ets_mul", "sarima_masked", "ridge_cx"]
SHOW = ["snaive"] + COMPONENTS + ["combo2", "combo3"]
INTERVAL_METHODS = COMPONENTS + ["combo2", "combo3"]
HORIZONS = [1, 3, 6, 12]
MARGIN = 0.2
FILES = [
    bt.OUT_DIR / "backtest_models_v2_type1.csv",
    bt.OUT_DIR / "backtest_ml_cx_type1.csv",
]

for f in FILES:
    if not f.exists():
        raise SystemExit(f"Cannot find {f}. Run steps 07, 08 and 10 first.")

raw = pd.concat([pd.read_csv(f, parse_dates=["origin", "target"]) for f in FILES], ignore_index=True)
raw = raw[raw["method"].isin(["snaive"] + COMPONENTS)]
assert not raw.duplicated(["origin", "target", "horizon", "method"]).any(), "duplicate forecasts in the saved files"

key = ["origin", "target", "horizon"]
wide = raw.pivot(index=key, columns="method", values="forecast")
actual = raw.drop_duplicates(key).set_index(key)["actual"]
assert set(["snaive"] + COMPONENTS) <= set(wide.columns), "a model is missing from the saved forecasts"

ok = wide[["snaive"] + COMPONENTS].gt(0).all(axis=1)  # NaN counts as not ok
print(f"Forecast rows (origin, target, horizon): {len(wide):,}")
print(f"Dropped because a component was missing or not positive: {int((~ok).sum())}")
wide = wide[ok].copy()
wide["combo2"] = wide[["ets_mul", "sarima_masked"]].mean(axis=1)
wide["combo3"] = wide[COMPONENTS].mean(axis=1)

long = (wide[SHOW].stack().rename("forecast").reset_index().rename(columns={"level_3": "method"}))
long["actual"] = long.set_index(key).index.map(actual)
long["error"] = long["forecast"] - long["actual"]
long["abs_err"] = long["error"].abs()
long["ape"] = long["abs_err"] / long["actual"] * 100
long["phase"] = long["target"].map(bt.label_phase)
assert long["forecast"].gt(0).all() and long["actual"].gt(0).all()

summary = bt.summarise(long)
mape = summary.pivot_table(index=["horizon", "phase"], columns="method", values="mape", observed=True)[SHOW]
print("\n=== MAPE (%): average absolute % error, by horizon (months ahead) and period ===")
print(mape.to_string(float_format=lambda x: f"{x:,.1f}"))
print("\n=== MAPE relative to seasonal naive (below 1.00 = better than 'same month last year') ===")
print(mape.div(mape["snaive"], axis=0).to_string(float_format=lambda x: f"{x:,.2f}"))
print("\n=== Number of forecasts behind each cell ===")
print(summary[summary["method"] == "snaive"].pivot_table(index="horizon", columns="phase", values="n", observed=True)
      .to_string(float_format=lambda x: f"{x:,.0f}"))

# ---- the declared decision rule
cl = mape.xs("clean", level="phase")
best_single = cl[COMPONENTS].min(axis=1)
gap = cl["combo3"] - best_single
worse = int((gap > MARGIN).sum())
print("\n=== DECISION RULE (clean periods) ===")
print(pd.DataFrame({"best single model": cl[COMPONENTS].idxmin(axis=1), "best single MAPE": best_single,
                    "combo3 MAPE": cl["combo3"], "combo3 minus best (pts)": gap}).to_string(float_format=lambda x: f"{x:,.2f}"))
print(f"Horizons where combo3 is worse than the best single model by more than {MARGIN} points: {worse} of 4")
print("VERDICT:", "single model preferred (rule says combo3 is clearly worse)" if worse >= 3
      else "ship combo3 (the rule does not find it clearly worse)")

# ---- intervals, 'stable' error history, for every model and both combinations
parts = []
for method in INTERVAL_METHODS:
    for h in HORIZONS:
        sub = long[(long["method"] == method) & (long["horizon"] == h)].sort_values("target").reset_index(drop=True)
        for level in m11.LEVELS:
            iv = m11.make_intervals(sub, "stable", level)
            iv["level"], iv["horizon"], iv["method"] = level, h, method
            parts.append(iv)
iv = pd.concat(parts, ignore_index=True)
iv.to_csv(bt.OUT_DIR / "intervals_combos_type1.csv", index=False)
v = iv.dropna(subset=["q"]).copy()
v["group"] = np.where(v["phase"] == "settled", "settled", "other")

for label, mask in [("settled period only", v["phase"] == "settled"),
                    ("clean periods (pre-COVID + settled)", v["phase"].isin(["pre_covid", "settled"]))]:
    t = v[mask].groupby(["level", "horizon", "method"]).agg(cov=("covered", "mean"), n=("covered", "size")).reset_index()
    t["cov"] *= 100
    print(f"\n=== COVERAGE (%) of the 'stable' intervals, {label} (nominal 80 / 95) ===")
    print(t.pivot_table(index=["level", "horizon"], columns="method", values="cov")[INTERVAL_METHODS]
          .to_string(float_format=lambda x: f"{x:,.0f}"))

w = v[v["phase"].isin(["pre_covid", "settled"])].groupby(["level", "horizon", "method"])["rel_width"].median().reset_index()
w["rel_width"] *= 100
print("\n=== MEDIAN INTERVAL WIDTH (% of the forecast), clean periods ===")
print(w.pivot_table(index=["level", "horizon"], columns="method", values="rel_width")[INTERVAL_METHODS]
      .to_string(float_format=lambda x: f"{x:,.1f}"))

out = long[long["method"].isin(["combo2", "combo3"])]
out.to_csv(bt.OUT_DIR / "backtest_combos_type1.csv", index=False)
print(f"\nSaved: {bt.OUT_DIR / 'backtest_combos_type1.csv'}  and  {bt.OUT_DIR / 'intervals_combos_type1.csv'}")
