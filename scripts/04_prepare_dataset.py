"""
Step 04: build the model-ready table.

What this does:
  1. Looks at the odd Jan 2016 row (a stray fractional count after real monthly data began).
  2. Replaces the crude 'fractional' flag with the one the Notes sheet actually states:
     months Nov 2010 - May 2015 were estimated from weekly data.
  3. Marks which months we are willing to SCORE forecasts on (June 2015 onwards).
  4. Marks a provisional COVID period (Mar 2020 - May 2021). This is a judgement call
     and will be stated openly in the write-up.
  5. Adds simple calendar features (weekdays, Mondays, weekend days in each month).
  6. Saves data/processed/ae_model_table.csv

Run from the project folder, with the virtual environment active:
    python scripts\04_prepare_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
SRC = PROJECT / "data" / "processed" / "ae_activity_clean.csv"
OUT = PROJECT / "data" / "processed" / "ae_model_table.csv"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

df = pd.read_csv(SRC, parse_dates=["period"])

OLD_FLAGS = ["estimated_from_weekly", "booked_included", "cyber_attack_window", "days_in_month"]
base_cols = [c for c in df.columns if c not in ["period"] + OLD_FLAGS]

# ------------------------------------------------------------ 1. stray fractional months
frac_any = ((df[base_cols] % 1).abs() > 1e-9).any(axis=1)
in_notes_block = (df["period"] >= "2010-11-01") & (df["period"] <= "2015-05-01")

print("=== 1. FRACTIONAL VALUES vs THE NOTES SHEET ===")
print(f"Months Nov 2010 - May 2015 (Notes say estimated): {int(in_notes_block.sum())}")
print(f"  of those, with fractional values: {int((frac_any & in_notes_block).sum())}")
print(f"  of those, whole numbers:          {int((~frac_any & in_notes_block).sum())}")
outside = df.loc[frac_any & ~in_notes_block, "period"].dt.strftime("%b %Y").tolist()
print(f"Fractional months OUTSIDE that block: {outside}")
for p in df.loc[frac_any & ~in_notes_block, "period"]:
    row = df.loc[df["period"] == p].iloc[0]
    cols = [c for c in base_cols if abs(row[c] % 1) > 1e-9]
    print(f"\n{p:%b %Y}: fractional columns and values")
    for c in cols:
        print(f"   {c[:60]:<60} {row[c]:,.2f}")

# ------------------------------------------------------------ 2-4. flags
df["estimated_from_weekly"] = in_notes_block
df["scoreable"] = df["period"] >= "2015-06-01"  # real monthly collection only

df["phase"] = np.select(
    [df["period"] < "2020-03-01", df["period"] <= "2021-05-01"],
    ["pre_covid", "covid"],
    default="post_covid",
)

# ------------------------------------------------------------ 5. calendar features
def month_counts(period):
    days = pd.date_range(period, period + pd.offsets.MonthEnd(0))
    dow = days.dayofweek  # Monday = 0 ... Sunday = 6
    return pd.Series(
        {
            "n_weekdays": int((dow < 5).sum()),
            "n_mondays": int((dow == 0).sum()),
            "n_weekend_days": int((dow >= 5).sum()),
        }
    )


cal = df["period"].apply(month_counts)
df = pd.concat([df, cal], axis=1)
df["month"] = df["period"].dt.month

# ------------------------------------------------------------ 6. save + summary
keep = [
    "period", "type1", "type2", "type3", "total",
    "estimated_from_weekly", "scoreable", "phase",
    "booked_included", "cyber_attack_window",
    "month", "days_in_month", "n_weekdays", "n_mondays", "n_weekend_days",
]
out = df[keep]
out.to_csv(OUT, index=False)

print("\n=== 2. WHAT WE KEPT ===")
print(f"Rows: {len(out)}  |  columns: {len(out.columns)}")
print(f"Estimated (Nov 2010 - May 2015, training only): {int(out['estimated_from_weekly'].sum())}")
print(f"Scoreable (Jun 2015 onwards): {int(out['scoreable'].sum())}")
print("\nScoreable months by phase:")
print(out[out["scoreable"]].groupby("phase")["period"].agg(["count", "min", "max"]).to_string())
print("\nLast 3 rows:")
print(out.tail(3).to_string(index=False))
print(f"\nSaved: {OUT}")
