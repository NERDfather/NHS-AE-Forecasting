"""
Step 01: look inside the NHS England Monthly A&E Time Series file.

This script only READS the file and prints what it finds. It changes nothing.
Goal: see how messy the layout is before we write any cleaning code.

Run from the project folder, with the virtual environment active:
    python scripts\01_inspect_excel.py
"""

from pathlib import Path

import pandas as pd

# Project folder = the folder above "scripts"
PROJECT = Path(__file__).resolve().parent.parent
XLS = PROJECT / "data" / "raw" / "monthly_ae_time_series.xls"

# Make printed tables easier to read in Command Prompt
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 12)
pd.set_option("display.max_colwidth", 28)

print(f"File: {XLS}")
print(f"Exists: {XLS.exists()}  |  Size: {XLS.stat().st_size:,} bytes\n")

# Read every sheet with NO header guessing, so we see the raw layout exactly
sheets = pd.read_excel(XLS, sheet_name=None, header=None, engine="xlrd")

print(f"Number of sheets: {len(sheets)}")
for name, df in sheets.items():
    print(f"  - '{name}': {df.shape[0]} rows x {df.shape[1]} columns")

for name, df in sheets.items():
    print("\n" + "=" * 100)
    print(f"SHEET: '{name}'  ({df.shape[0]} rows x {df.shape[1]} columns)")
    print("=" * 100)
    print("--- first 15 rows ---")
    print(df.head(15).to_string(max_cols=12))
    print("--- last 5 rows ---")
    print(df.tail(5).to_string(max_cols=12))
