"""
Step 13: the final forecast, to be FROZEN before the next data release.

What this does:
  1. PROOF STEP. Re-runs one past forecast (origin Sep 2024) with the code in ae_models.py
     and compares it with the forecasts saved from the backtest. If they differ, the code we
     ship is not the code we tested, and the script stops.
  2. Fits the three models on ALL the data and forecasts the next 12 months:
        ets_mul        Holt-Winters, damped trend, multiplicative seasonality
        sarima_masked  SARIMA on log attendances, COVID months treated as missing
        ridge_cx       ridge regression on year-on-year changes, COVID rows left out of training
  3. Averages them with equal weights (combo3), as chosen by the rule declared in step 12.
  4. Builds 80% and 95% intervals from combo3's own past errors at each horizon, using the
     'stable' error history (errors for targets Mar 2020 - May 2022 left out), as in step 11.
  5. Saves the frozen forecast (CSV + a small JSON record), and one chart.

Needs the saved backtest files from steps 07, 08, 10, 11 and 12.

Run from the project folder, with the virtual environment active:
    python scripts\13_final_forecast.py
"""

import datetime
import importlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

import ae_backtest as bt
import ae_models as am

m11 = importlib.import_module("11_prediction_intervals")

TARGET = "type1"
MAX_H = 12
LEVELS = (0.80, 0.95)
REPRO_ORIGIN = pd.Timestamp("2024-09-01")
REPRO_TOL = 1e-4  # relative difference allowed in the proof step
HISTORY_MONTHS = 48  # months of history shown in the chart
FIG_DIR = bt.OUT_DIR / "figures"

SAVED = [
    bt.OUT_DIR / "backtest_models_v2_type1.csv",
    bt.OUT_DIR / "backtest_ml_cx_type1.csv",
    bt.OUT_DIR / "backtest_combos_type1.csv",
]

# chart tokens (the reference palette's light mode)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BLUE = "#e1e0d9", "#c3c2b7", "#2a78d6"

for f in SAVED:
    if not f.exists():
        raise SystemExit(f"Cannot find {f}. Run steps 07, 08, 10, 11 and 12 first.")
FIG_DIR.mkdir(parents=True, exist_ok=True)

tbl = bt.load_table()
y = tbl[TARGET]
origin = y.index[-1]
generated_on = datetime.date.today()
print(f"Data: {len(y)} months, {y.index[0]:%b %Y} to {origin:%b %Y}  |  latest value {y.iloc[-1]:,.0f}")
print(f"Forecast origin: {origin:%b %Y}  |  forecasting {(origin + pd.DateOffset(months=1)):%b %Y} to "
      f"{(origin + pd.DateOffset(months=MAX_H)):%b %Y}")

models = am.shipped_models()
saved = pd.concat([pd.read_csv(f, parse_dates=["origin", "target"]) for f in SAVED], ignore_index=True)

# ------------------------------------------------------------ 1. proof step
print(f"\n=== 1. PROOF: does this code reproduce the backtest forecast made in {REPRO_ORIGIN:%b %Y}? ===")
hist = y.loc[:REPRO_ORIGIN]
fresh = {}
for name, fn in models.items():
    fresh[name] = fn(hist, MAX_H)
fresh["combo3"] = pd.concat(list(fresh.values()), axis=1).mean(axis=1)
failed = False
for name, fc in fresh.items():
    ref = (saved[(saved["method"] == name) & (saved["origin"] == REPRO_ORIGIN)]
           .sort_values("horizon").set_index("horizon")["forecast"])
    if len(ref) != MAX_H:
        raise SystemExit(f"Saved backtest has {len(ref)} forecasts for {name} at {REPRO_ORIGIN:%b %Y}, expected {MAX_H}.")
    rel = np.abs(fc.to_numpy() - ref.to_numpy()) / ref.to_numpy()
    ok = rel.max() <= REPRO_TOL
    failed |= not ok
    print(f"  {name:<14} max relative difference {rel.max():.2e}  -> {'OK' if ok else 'MISMATCH'}")
if failed:
    raise SystemExit("\nSTOP: the shipped code does not reproduce the tested forecasts. Paste this output to Claude.")
am.nonconverged.clear()

# ------------------------------------------------------------ 2-3. final fit and average
print("\n=== 2. FINAL FIT ON ALL DATA ===")
parts = {}
for name, fn in models.items():
    parts[name] = fn(y, MAX_H)
    print(f"  {name:<14} next month {parts[name].iloc[0]:>12,.0f}   in 12 months {parts[name].iloc[-1]:>12,.0f}")
sarima_ok = not any(n == "sarima_masked" for n, _ in am.nonconverged)
print(f"  SARIMA optimiser converged: {'yes' if sarima_ok else 'NO - treat the SARIMA forecast with extra care'}")

frame = pd.DataFrame(parts)
frame["combo3"] = frame[list(models)].mean(axis=1)
frame["snaive"] = bt.snaive(y, MAX_H)
last = float(y.iloc[-1])
assert np.isfinite(frame.to_numpy()).all(), "a forecast is not finite"
assert frame.gt(0.7 * last).all().all() and frame.lt(1.4 * last).all().all(), "a forecast looks implausible"

# ------------------------------------------------------------ 4. intervals
c3 = saved[(saved["method"] == "combo3") & (saved["target"] <= origin)]
rows = []
for h in range(1, MAX_H + 1):
    s = c3[c3["horizon"] == h]
    keep = (((s["target"] < m11.COVID_START) | (s["target"] > m11.STABLE_END))
            & (s["forecast"] > 0) & (s["actual"] > 0))
    errs = np.abs(np.log(s.loc[keep, "actual"] / s.loc[keep, "forecast"])).to_numpy()
    rec = {"horizon": h, "n_errors": len(errs)}
    for level in LEVELS:
        q = m11.conformal_q(errs, level)
        if np.isnan(q):
            raise SystemExit(f"Not enough past errors at horizon {h} to make a {level:.0%} interval.")
        rec[f"q{int(level * 100)}"] = q
    rows.append(rec)
q = pd.DataFrame(rows).set_index("horizon")

out = pd.DataFrame({"origin": origin, "target": frame.index, "horizon": range(1, MAX_H + 1)})
out["forecast"] = frame["combo3"].to_numpy()
for level in LEVELS:
    k = int(level * 100)
    out[f"lo{k}"] = out["forecast"] * np.exp(-q[f"q{k}"].to_numpy())
    out[f"hi{k}"] = out["forecast"] * np.exp(q[f"q{k}"].to_numpy())
for name in list(models) + ["snaive"]:
    out[name] = frame[name].to_numpy()
last_year = [y.loc[t - pd.DateOffset(months=12)] for t in out["target"]]
out["same_month_last_year"] = last_year
out["change_vs_last_year_pct"] = (out["forecast"] / out["same_month_last_year"] - 1) * 100
out["n_errors_used"] = q["n_errors"].to_numpy()

print("\n=== 3. THE FORECAST: average of three models, with intervals from its own past errors ===")
show = pd.DataFrame({
    "month": out["target"].dt.strftime("%b %Y"),
    "forecast": out["forecast"].map("{:,.0f}".format),
    "80% interval": [f"{a:,.0f} - {b:,.0f}" for a, b in zip(out["lo80"], out["hi80"])],
    "95% interval": [f"{a:,.0f} - {b:,.0f}" for a, b in zip(out["lo95"], out["hi95"])],
    "vs same month last year": out["change_vs_last_year_pct"].map("{:+.1f}%".format),
})
print(show.to_string(index=False))

spread = (frame[list(models)].max(axis=1) - frame[list(models)].min(axis=1)) / frame["combo3"] * 100
print(f"\nDisagreement between the three models (highest minus lowest, % of the average): "
      f"{spread.min():.1f}% to {spread.max():.1f}%  (Sep 2027: {spread.iloc[-1]:.1f}%)")
print(f"Past errors behind each interval: {int(q['n_errors'].min())} to {int(q['n_errors'].max())} forecasts per horizon.")
print("Intervals for horizons 2, 4, 5, 7-11 use the same method but were not separately coverage-checked "
      "(step 11 checked 1, 3, 6 and 12).")

# ------------------------------------------------------------ 5. save
tag = f"{origin:%Y-%m}"
csv_path = bt.OUT_DIR / f"forecast_type1_{tag}.csv"
out.to_csv(csv_path, index=False, float_format="%.2f")


def version(mod):
    try:
        return __import__(mod).__version__
    except Exception:  # noqa: BLE001
        return "unavailable"


record = {
    "target_series": "England A&E attendances, Type 1 departments, monthly",
    "data_source": "NHS England Monthly A&E Time Series (published 8 Oct 2026); Open Government Licence v3.0",
    "data_last_month": f"{origin:%Y-%m}",
    "latest_value": last,
    "generated_on": generated_on.isoformat(),
    "forecast_months": [f"{t:%Y-%m}" for t in out["target"]],
    "method": "equal-weight average of ets_mul, sarima_masked, ridge_cx (rule declared in step 12)",
    "intervals": "80% and 95%; empirical, from combo3's own past errors per horizon; "
                 "error history excludes targets Mar 2020 - May 2022 (chosen after seeing step 11 results)",
    "proof_step": f"reproduced the backtest forecast for origin {REPRO_ORIGIN:%Y-%m} within {REPRO_TOL:g}",
    "sarima_converged": bool(sarima_ok),
    "library_versions": {k: version(k) for k in ["pandas", "numpy", "statsmodels", "sklearn"]},
    "scoring_note": "Score against actuals as NHS England publishes them (Oct 2026 data due 12 Nov 2026). "
                    "Do not change this file after generation.",
}
json_path = bt.OUT_DIR / f"forecast_type1_{tag}.json"
json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
print(f"\nSaved: {csv_path}\nSaved: {json_path}")


# ------------------------------------------------------------ chart
def millions(v, _):
    return f"{v / 1e6:.2f}m"


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]

hist_start = origin - pd.DateOffset(months=HISTORY_MONTHS - 1)
h = y.loc[hist_start:origin]
fx = out.set_index("target")
idx = pd.DatetimeIndex([origin] + list(fx.index))


def from_origin(values):
    return np.r_[last, np.asarray(values, dtype=float)]


fc_line = from_origin(fx["forecast"])
lo80, hi80 = from_origin(fx["lo80"]), from_origin(fx["hi80"])
lo95, hi95 = from_origin(fx["lo95"]), from_origin(fx["hi95"])

fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=130, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)

ax.fill_between(idx, lo95, hi95, color=BLUE, alpha=0.10, linewidth=0)
ax.fill_between(idx, lo80, hi80, color=BLUE, alpha=0.20, linewidth=0)
for name in models:
    ax.plot(idx, from_origin(fx[name]), color=MUTED, linewidth=0.9, zorder=2)
ax.plot(h.index, h.to_numpy(), color=BLUE, linewidth=1.8, solid_capstyle="round", zorder=3)
ax.plot(idx, fc_line, color=BLUE, linewidth=1.8, linestyle=(0, (4, 2)), zorder=3)
ax.axvline(origin, color=AXIS, linewidth=0.8, zorder=1)
for x, v in [(origin, last), (idx[-1], fc_line[-1])]:
    ax.plot([x], [v], marker="o", markersize=6, color=BLUE, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=4)

ax.annotate(f"{idx[-1]:%b %Y}\n{fc_line[-1] / 1e6:.2f}m", (idx[-1], fc_line[-1]), xytext=(9, 0),
            textcoords="offset points", ha="left", va="center", fontsize=9, color=INK)
ax.text(origin + pd.DateOffset(days=12), 1.0, "forecast", transform=ax.get_xaxis_transform(), fontsize=8.5,
        color=MUTED, va="top")

lo_y = min(h.min(), lo95.min())
hi_y = max(h.max(), hi95.max())
pad = (hi_y - lo_y) * 0.06
ax.set_ylim(lo_y - pad, hi_y + pad)
ax.set_xlim(hist_start - pd.DateOffset(days=20), idx[-1] + pd.DateOffset(months=3))
ax.yaxis.set_major_formatter(FuncFormatter(millions))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(axis="both", labelsize=9, labelcolor=INK2, color=AXIS, length=3)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(AXIS)

handles = [
    Line2D([0], [0], color=BLUE, linewidth=1.8, label="Actual"),
    Line2D([0], [0], color=BLUE, linewidth=1.8, linestyle=(0, (4, 2)), label="Forecast (average of three models)"),
    Patch(facecolor=BLUE, alpha=0.20, linewidth=0, label="80% interval"),
    Patch(facecolor=BLUE, alpha=0.10, linewidth=0, label="95% interval"),
    Line2D([0], [0], color=MUTED, linewidth=0.9, label="The three models separately"),
]
leg = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, labelcolor=INK2, handlelength=2.2)

fig.text(0.065, 0.955, "England A&E attendances at major departments (Type 1): forecast to "
         f"{idx[-1]:%b %Y}", fontsize=14, fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.065, 0.905, "Average of three models. Bands show where outcomes have landed 80% and 95% of the time "
         "in past forecasts.", fontsize=10, color=INK2, ha="left", va="top")
fig.text(0.065, 0.025, f"Source: NHS England, Monthly A&E Time Series (published 8 Oct 2026). Contains public sector "
         f"information licensed under the Open Government Licence v3.0.\nHistory shown from "
         f"{hist_start:%b %Y}; earlier months (including COVID) omitted for scale. Forecast generated "
         f"{generated_on:%d %b %Y}.", fontsize=7.5, color=MUTED, ha="left", va="bottom")
fig.subplots_adjust(left=0.085, right=0.95, top=0.84, bottom=0.14)

fig_path = FIG_DIR / "03_forecast_type1.png"
fig.savefig(fig_path, facecolor=SURFACE)
plt.close(fig)
print(f"Saved chart: {fig_path}")
