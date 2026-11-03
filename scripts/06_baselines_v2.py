"""
Step 06: re-run the baselines with the reusable backtest module, and split the
post-COVID period into 'recovery' (Jun 2021 - May 2022) and 'settled' (Jun 2022 on).

Why: in the recovery months, 'same month last year' was itself COVID-affected, so
baseline errors there say more about COVID than about the method.

Run from the project folder, with the virtual environment active:
    python scripts\06_baselines_v2.py
"""

import ae_backtest as bt

TARGET = "type1"

tbl = bt.load_table()
y = tbl[TARGET]

res = bt.walk_forward(y, bt.BASELINES, tbl["scoreable"])
bt.OUT_DIR.mkdir(parents=True, exist_ok=True)
res.to_csv(bt.OUT_DIR / "backtest_baselines_v2_type1.csv", index=False)

print("=== BACKTEST SETUP ===")
print(f"Target: {TARGET}  |  forecast rows: {len(res):,}  (step 05 had 4,698)")
print(f"Targets scored: {res['target'].min():%b %Y} to {res['target'].max():%b %Y}")

months = res[(res["horizon"] == 1) & (res["method"] == "naive")].groupby("phase")["target"].agg(["count", "min", "max"])
print("\nScored months per period:")
print(months.to_string())

summary = bt.summarise(res)
bt.print_tables(summary, bt.BASELINES)
print("\nCheck: the pre_covid rows should match step 05 (e.g. horizon 1: 4.5 / 2.9 / 2.6).")
