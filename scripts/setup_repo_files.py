r"""
Creates the supporting files for the GitHub repository:
  .gitignore, LICENSE, README.md, requirements.txt
It NEVER overwrites a file that already exists. It does not touch Git.

Run from the project folder with the virtual environment active:
    python scripts\setup_repo_files.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLDER = "Franklyn Epidi"

GITIGNORE = """\
# virtual environment and Python caches
.venv/
__pycache__/
*.pyc

# scratch output from the first look at the Excel file
inspect_output.txt

# editor and operating-system files
.vscode/
.idea/
.DS_Store
Thumbs.db
"""

LICENSE = f"""\
MIT License

Copyright (c) 2026 {HOLDER}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This licence covers the code only. The NHS England data in data/raw is
published under the Open Government Licence v3.0 (see README.md).
"""

README = """\
# Forecasting NHS England A&E attendances, with honest backtesting

Monthly attendances at major (Type 1) A&E departments in England, forecast 12
months ahead. The point of the project is not a clever model. It is an
evaluation that does not flatter itself: every forecast in the backtest uses
only the data that existed at the time, and the final forecast is frozen in
this repository before the next data release so it can be scored live.

![Forecast to Sep 2027](outputs/figures/03_forecast_type1.png)

## The frozen forecast

`outputs/forecast_type1_2026-09.csv` and `.json` hold the forecast made from
data up to September 2026 (published by NHS England on 8 Oct 2026). They were
generated on 1 Nov 2026 and are committed before the October 2026 figures are
released (expected 12 Nov 2026). **They must not be edited or regenerated.**
Scoring against the real figures will be added as each month is published,
whether the forecast looks good or bad.

## What was done

- Cleaned the messy NHS Excel file (header on row 14, estimated months,
  changing definitions) into a tidy table.
- Walk-forward backtest from June 2015, horizons 1 to 12 months. Results are
  split into before COVID, COVID, recovery and settled periods, because the
  COVID years dominate any single average.
- Baselines: naive and seasonal naive ("same month last year").
- Models: Holt-Winters (ETS), SARIMA, ridge regression and gradient boosting on
  lag features. COVID-aware versions leave the COVID months out of training.
- Prediction intervals built from each model's own past errors (known at the
  time of the forecast), with a coverage check.
- Final forecast: equal-weight average of ETS, SARIMA (COVID months masked) and
  COVID-aware ridge, chosen by a decision rule written down before looking at
  the combination results.

## Caveats

- The COVID window and the "stable" error history for the intervals were
  chosen after seeing results (hindsight). Both are flagged in the scripts.
- Many models and variants were compared, so the best-looking result is
  partly luck. The write-up will list every variant tried.
- About a fifth of SARIMA fits did not report convergence.
- Coverage was only checked at horizons 1, 3, 6 and 12. Interval widths at
  neighbouring horizons are not smooth.
- The 80% intervals may be slightly too narrow in the settled period.
- Months from Nov 2010 to May 2015 were estimated from weekly data and are
  never scored.

## Reproduce

Python 3 on Windows, from the project folder with a virtual environment:

    pip install -r requirements.txt
    python scripts\\02_load_clean.py
    python scripts\\04_prepare_dataset.py
    python scripts\\07_models_ets_sarima.py
    python scripts\\08_models_ml.py
    python scripts\\10_models_ml_covidaware.py
    python scripts\\11_prediction_intervals.py
    python scripts\\12_combinations.py

Scripts 01, 03, 05, 06 and 09 are inspection, exploration and diagnostics.
Script 13 produces the final forecast and includes a proof step: it re-runs a
past forecast with the shipped code and stops unless it matches the backtest.
Do not run 13 again and commit the result.

## Data and licence

Data: NHS England, Monthly A&E Time Series (Activity sheet), in `data/raw`.
Contains public sector information licensed under the
[Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

Code: MIT, see `LICENSE`.
"""


def write_new(name, text):
    path = ROOT / name
    if path.exists():
        print(f"  skipped {name} (already exists, left untouched)")
        return
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"  created {name}")


print(f"Project folder: {ROOT}")
write_new(".gitignore", GITIGNORE)
write_new("LICENSE", LICENSE)
write_new("README.md", README)

req = ROOT / "requirements.txt"
if req.exists():
    print("  skipped requirements.txt (already exists, left untouched)")
else:
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        print("  could not create requirements.txt:", result.stderr.strip()[:300])
    else:
        req.write_text(result.stdout, encoding="utf-8", newline="\n")
        print(f"  created requirements.txt ({len(result.stdout.splitlines())} packages)")
print("Done.")
