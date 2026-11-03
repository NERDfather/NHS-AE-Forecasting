"""
Step 09: where do the ML models lose? (READ-ONLY: fits nothing, changes nothing.)

Hypothesis from step 08 (not yet tested): ridge and gbm do badly in the 'settled' period
because many of their forecasts were made from origins in 2021-22, when their features
(year-on-year changes) and their training rows were still full of COVID distortion.

If that is right:
  * their errors should shrink for forecasts made from LATER origins, and
  * their errors should lean in one direction (a consistent bias), not be random.

Reads the saved forecasts from steps 07 and 08, so run those first.

Run from the project folder, with the virtual environment active:
    python scripts\09_diagnose_ml.py
"""

import pandas as pd

import ae_backtest as bt

SHOW = ["snaive", "ets_mul", "sarima", "sarima_masked", "ridge", "gbm"]
HORIZONS = [1, 3, 6, 12]

files = [bt.OUT_DIR / "backtest_models_v2_type1.csv", bt.OUT_DIR / "backtest_ml_type1.csv"]
for f in files:
    if not f.exists():
        raise SystemExit(f"Cannot find {f}. Run steps 07 and 08 first.")

res = pd.concat([pd.read_csv(f, parse_dates=["origin", "target"]) for f in files], ignore_index=True)
res = res[res["method"].isin(SHOW)].copy()
res["pe"] = (res["forecast"] - res["actual"]) / res["actual"] * 100  # signed % error (+ = forecast too high)

settled = res[res["phase"] == "settled"].copy()
print(f"Settled-period forecasts (targets Jun 2022 onwards): {len(settled):,}")


def origin_group(origin):
    if origin < pd.Timestamp("2022-06-01"):
        return "a) origin up to May 2022"
    if origin < pd.Timestamp("2023-06-01"):
        return "b) origin Jun 2022 - May 2023"
    return "c) origin Jun 2023 on"


settled["origin_group"] = settled["origin"].map(origin_group)
sub = settled[settled["horizon"].isin(HORIZONS)]

grp = sub.groupby(["horizon", "origin_group", "method"]).agg(mape=("ape", "mean"), n=("ape", "size")).reset_index()

print("\n=== 1. MAPE (%) in the settled period, split by WHEN the forecast was made ===")
print(grp.pivot_table(index=["horizon", "origin_group"], columns="method", values="mape")[SHOW]
      .to_string(float_format=lambda x: f"{x:,.1f}"))
print("\nNumber of forecasts behind each row:")
print(grp[grp["method"] == "snaive"].pivot_table(index="horizon", columns="origin_group", values="n")
      .to_string(float_format=lambda x: f"{x:,.0f}"))

bias = (settled[settled["horizon"].isin([1, 6, 12])]
        .groupby(["horizon", "method"])["pe"].mean().unstack()[SHOW])
print("\n=== 2. AVERAGE SIGNED ERROR (%) in the settled period (+ = forecasts too high, - = too low) ===")
print(bias.to_string(float_format=lambda x: f"{x:+,.1f}"))

share = (settled[settled["horizon"].isin([1, 6, 12])]
         .assign(too_high=lambda d: d["pe"] > 0)
         .groupby(["horizon", "method"])["too_high"].mean().unstack()[SHOW] * 100)
print("\n=== 3. SHARE OF FORECASTS THAT WERE TOO HIGH (%) (50 = no lean either way) ===")
print(share.to_string(float_format=lambda x: f"{x:,.0f}"))
