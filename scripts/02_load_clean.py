"""
Step 02: load the 'Activity' sheet into a tidy table, check it, and save it.

What this does:
  1. Finds the header row (the row whose second cell says 'Period').
  2. Keeps only the real data columns and drops the chart-helper junk on the right.
  3. Checks the months are complete, the numbers add up, and counts which
     months look estimated (fractional values).
  4. Adds three flag columns taken straight from the Notes sheet.
  5. Saves data/processed/ae_activity_clean.csv

Run from the project folder, with the virtual environment active:
    python scripts\02_load_clean.py
"""

import re
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
XLS = PROJECT / "data" / "raw" / "monthly_ae_time_series.xls"
OUT_DIR = PROJECT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "ae_activity_clean.csv"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.options.display.float_format = "{:,.1f}".format

# ---------------------------------------------------------------- 1. find header
raw = pd.read_excel(XLS, sheet_name="Activity", header=None, engine="xlrd")
hits = raw.index[raw[1].astype(str).str.strip().eq("Period")]
assert len(hits) == 1, f"Expected exactly one 'Period' header row, found {len(hits)}"
header_row = int(hits[0])
print(f"Header row found at row {header_row} (0-based). Data starts on row {header_row + 1}.")

# ---------------------------------------------------------------- 2. columns
FIRST_COL, LAST_COL = 1, 13  # columns 14-17 are chart helpers ('Operational standard', 0.95)
headers = raw.iloc[header_row, FIRST_COL : LAST_COL + 1]
assert headers.notna().all(), "A header cell is empty inside the expected data columns"


def slug(text):
    text = str(text).replace(">", " over ").replace("<", " under ")
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


names = [slug(h) for h in headers]
# Friendly names for the five columns we will model; check the text first
expected = {
    1: ("Period", "period"),
    2: ("Type 1", "type1"),
    3: ("Type 2", "type2"),
    4: ("Type 3", "type3"),
    5: ("Total Attendances", "total"),
}
for pos, (must_contain, friendly) in expected.items():
    header_text = str(headers.iloc[pos - 1])
    assert must_contain in header_text, f"Column {pos} is '{header_text}', expected '{must_contain}'"
    names[pos - 1] = friendly
assert len(set(names)) == len(names), "Duplicate column names after cleaning"

print("\nColumns kept (position -> clean name -> original header):")
for i, (n, h) in enumerate(zip(names, headers), start=FIRST_COL):
    print(f"  {i:>2}  {n:<45} {str(h)[:90]}")

# ---------------------------------------------------------------- 3. build table
df = raw.iloc[header_row + 1 :, FIRST_COL : LAST_COL + 1].copy()
df.columns = names
df["period"] = pd.to_datetime(df["period"], errors="coerce")
dropped = int(df["period"].isna().sum())
df = df.dropna(subset=["period"]).sort_values("period").reset_index(drop=True)
print(f"\nRows dropped because the period was not a date: {dropped}")

num_cols = [c for c in df.columns if c != "period"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
print("Missing values per column (should all be 0):")
print(df[num_cols].isna().sum().to_string())

# ---------------------------------------------------------------- 4. checks
print("\n=== CHECKS ===")
print(f"Rows: {len(df)}  |  first month: {df['period'].min():%b %Y}  |  last month: {df['period'].max():%b %Y}")
print(f"All dates are the 1st of a month: {bool((df['period'].dt.day == 1).all())}")
expected_months = pd.date_range(df["period"].min(), df["period"].max(), freq="MS")
missing = expected_months.difference(pd.DatetimeIndex(df["period"]))
print(f"Missing months: {len(missing)}  |  duplicate months: {int(df['period'].duplicated().sum())}")

gap = (df["total"] - (df["type1"] + df["type2"] + df["type3"])).abs()
print(f"Largest gap between Total and Type1+Type2+Type3: {gap.max():,.3f}")

count_cols = ["type1", "type2", "type3", "total"]
fractional = ((df[count_cols] % 1).abs() > 1e-9).any(axis=1)
print(f"Months with fractional counts (estimated from weekly data): {int(fractional.sum())}")
if fractional.any():
    est = df.loc[fractional, "period"]
    print(f"  first: {est.min():%b %Y}  |  last: {est.max():%b %Y}")
    in_range = ((df["period"] >= est.min()) & (df["period"] <= est.max())).sum()
    print(f"  months between first and last: {int(in_range)} (equals count above if it is one solid block)")

# ---------------------------------------------------------------- 5. flags from the Notes sheet
df["estimated_from_weekly"] = fractional
df["booked_included"] = df["period"] >= "2020-08-01"
df["cyber_attack_window"] = (df["period"] >= "2022-08-01") & (df["period"] <= "2023-02-01")
df["days_in_month"] = df["period"].dt.days_in_month

print(f"\nFlag counts -> estimated: {int(df['estimated_from_weekly'].sum())}, "
      f"booked included: {int(df['booked_included'].sum())}, "
      f"cyber window: {int(df['cyber_attack_window'].sum())}")

# ---------------------------------------------------------------- 6. eyeball
show = ["period", "type1", "type2", "type3", "total"]
print("\nFirst 4 months:")
print(df[show].head(4).to_string(index=False))
print("\nLast 4 months:")
print(df[show].tail(4).to_string(index=False))

yearly = df.assign(year=df["period"].dt.year).groupby("year")["total"].agg(["count", "sum"])
print("\nTotal attendances per calendar year (months counted in 'count'):")
print(yearly.to_string(float_format=lambda x: f"{x:,.0f}"))

df.to_csv(OUT, index=False)
print(f"\nSaved: {OUT}")
