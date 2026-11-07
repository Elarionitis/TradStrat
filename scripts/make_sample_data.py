"""Generate the bundled synthetic dataset.

The series is a seeded Markov regime-switching random walk, so demos and tests get bull markets, bear
markets and sideways chop without any network access or third-party data. It is simulated data: results
on it say nothing about real markets.

Usage: python scripts/make_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT = Path(__file__).resolve().parents[1] / "tradstrat" / "sample_data" / "synthetic_ohlcv.csv"
SEED = 20240101
START, END = "2016-01-04", "2025-12-31"

# Each regime is (annual drift, annual volatility, expected length in trading days).
REGIMES = (
    (0.22, 0.14, 300),  # steady bull
    (-0.30, 0.32, 90),  # bear
    (0.02, 0.20, 150),  # sideways
)


def simulate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, END)
    regime = 0
    log_returns = np.empty(len(dates))
    for day in range(len(dates)):
        drift, volatility, expected_length = REGIMES[regime]
        if rng.random() < 1 / expected_length:
            regime = int(rng.choice([other for other in range(len(REGIMES)) if other != regime]))
        # Student-t innovations give the fat tails real returns have; dividing by sqrt(df / (df - 2))
        # restores unit variance.
        shock = rng.standard_t(df=5) / np.sqrt(5 / 3)
        log_returns[day] = drift / 252 + volatility / np.sqrt(252) * shock

    close = 100 * np.exp(np.cumsum(log_returns))
    previous_close = np.concatenate([[100.0], close[:-1]])
    # About a third of the daily variance arrives overnight, as a gap between yesterday's close and the open.
    open_ = previous_close * np.exp(0.6 * log_returns + rng.normal(0, 0.002, len(dates)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, len(dates))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, len(dates))))
    volume = rng.lognormal(mean=15, sigma=0.35, size=len(dates)).astype(int)

    frame = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.Index(dates, name="Date"),
    )
    return frame.round({"Open": 4, "High": 4, "Low": 4, "Close": 4})


if __name__ == "__main__":
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    simulate().to_csv(OUTPUT)
    print(f"Wrote {OUTPUT}")
