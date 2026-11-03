"""
Step 03: explore the cleaned A&E series before we model anything.

What this does:
  1. Looks closely at the months around the 'estimated' period (Notes say to May 2015,
     but fractional counts run to Jan 2016).
  2. Shows how far attendances fell in early 2020 compared with 2019.
  3. Measures the seasonal pattern (which months are high/low) for Type 1,
     both raw and per day (so we can see how much is just month length).
  4. Saves two charts to outputs/figures.

Run from the project folder, with the virtual environment active:
    python scripts\03_explore.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
CSV = PROJECT / "data" / "processed" / "ae_activity_clean.csv"
FIG_DIR = PROJECT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 200)
pd.options.display.float_format = "{:,.3f}".format

df = pd.read_csv(CSV, parse_dates=["period"])
count_cols = ["type1", "type2", "type3", "total"]
df["fractional"] = ((df[count_cols] % 1).abs() > 1e-9).any(axis=1)

# ------------------------------------------------------------ 1. the 'estimated' period
first = df.loc[df["fractional"], "period"].min()
last = df.loc[df["fractional"], "period"].max()
block = df[(df["period"] >= first) & (df["period"] <= last)]
whole = block.loc[~block["fractional"], "period"].dt.strftime("%b %Y").tolist()

print("=== 1. FRACTIONAL (ESTIMATED?) MONTHS ===")
print(f"First fractional month: {first:%b %Y}  |  last: {last:%b %Y}")
print(f"Whole-number months INSIDE that span ({len(whole)}): {', '.join(whole)}")
print("\nMonths around the end of the block (Notes say estimates stop after May 2015):")
win = df[(df["period"] >= "2015-03-01") & (df["period"] <= "2016-03-01")]
print(win[["period", "type1", "total", "fractional"]].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

# ------------------------------------------------------------ 2. COVID drop
d = df.set_index("period")
cov = d.loc["2020-01":"2020-08", ["type1", "total"]].copy()
prev = d.loc["2019-01":"2019-08", ["type1", "total"]]
cov["type1_vs_2019_pct"] = (cov["type1"].values / prev["type1"].values - 1) * 100
cov["total_vs_2019_pct"] = (cov["total"].values / prev["total"].values - 1) * 100
print("\n=== 2. EARLY 2020 vs SAME MONTH IN 2019 ===")
print(cov.to_string(float_format=lambda x: f"{x:,.1f}"))

# ------------------------------------------------------------ 3. seasonal pattern
pre = d.loc["2012":"2019"].copy()  # full years before COVID
pre["year"] = pre.index.year
pre["month"] = pre.index.month
table = pd.DataFrame(index=range(1, 13))
for label, series in {
    "type1_raw": pre["type1"],
    "type1_per_day": pre["type1"] / pre["days_in_month"],
}.items():
    year_mean = series.groupby(pre["year"]).transform("mean")
    table[label] = (series / year_mean).groupby(pre["month"]).mean()
print("\n=== 3. SEASONAL INDEX, TYPE 1, 2012-2019 (1.000 = average month of its year) ===")
print(table.to_string())

# ------------------------------------------------------------ 4. charts
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
for ax, col, title in [
    (axes[0], "type1", "Type 1 (major A&E) attendances per month"),
    (axes[1], "total", "Total attendances per month"),
]:
    ax.plot(df["period"], df[col], color="#1f4e79", linewidth=1.4)
    ax.axvspan(first, last, color="orange", alpha=0.15, label="fractional counts (estimated?)")
    ax.axvspan(pd.Timestamp("2022-08-01"), pd.Timestamp("2023-02-28"), color="red", alpha=0.15,
               label="cyber-attack window (Total / Type 3 understated)")
    ax.axvline(pd.Timestamp("2020-08-01"), color="green", linestyle="--", linewidth=1,
               label="booked appointments counted from here")
    ax.set_title(title)
    ax.set_ylabel("attendances")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}m"))
    ax.grid(alpha=0.3)
axes[0].legend(loc="lower right", fontsize=8)
fig.tight_layout()
f1 = FIG_DIR / "01_series_overview.png"
fig.savefig(f1, dpi=130)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5.5))
cmap = plt.get_cmap("viridis")
years = [y for y in range(2012, 2026) if y not in (2020, 2021)]
for i, y in enumerate(years):
    sub = d[d.index.year == y]
    ax.plot(sub.index.month, sub["type1"], color=cmap(i / (len(years) - 1)), label=str(y), linewidth=1.3)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
ax.set_title("Type 1 attendances by month, one line per year (2020-21 left out)")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e6:.2f}m"))
ax.grid(alpha=0.3)
ax.legend(ncol=4, fontsize=8)
fig.tight_layout()
f2 = FIG_DIR / "02_seasonality_type1.png"
fig.savefig(f2, dpi=130)
plt.close(fig)

print(f"\nSaved charts:\n  {f1}\n  {f2}")
