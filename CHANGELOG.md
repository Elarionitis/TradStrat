# Changelog

All notable changes to TradStrat. The project follows [semantic versioning](https://semver.org).

## 2.1.0

### Added

- **Interest on idle cash.** Uninvested cash can earn a fixed rate or the real 13-week US Treasury bill
  yield, accrued over calendar days at the rate known the evening before.
- **Sharpe ratios on excess returns.** Sharpe, Sortino, alpha, the bootstrap and the deflated Sharpe ratio are
  all measured on returns above cash, so holding Treasury bills is never mistaken for skill.
- **Probability of backtest overfitting.** Parameter searches now report how often the in-sample winner falls
  to the bottom half out of sample, using combinatorially symmetric cross-validation over 12,870 splits.
- **Trailing stops** that follow the highest high of completed bars.
- **A plain-English verdict** on every backtest, shared by the dashboard, the command line and the report.
- **HTML reports.** `tradstrat report` and a dashboard button write a self-contained report with the verdict,
  interactive charts, robustness statistics and the full trade log.
- **Strategy comparison.** `tradstrat compare` and a dashboard tab run every strategy at its defaults against
  buy and hold.
- **Real-world CSV files.** Column names are matched without regard to case, dates may be Unix timestamps,
  and volume is optional. The dashboard accepts uploads.
- New charts: calendar-year returns, rolling Sharpe ratio, the overfitting distribution, and each strategy's
  own indicator lines on the price chart.

### Changed

- The dashboard was redesigned: verdict first, KPI cards with sparklines, quick ticker and period pickers,
  and charts that follow the light or dark theme.
- Search and walk-forward results are cleared when any input changes, so they can never describe other
  settings.
- The research study in `docs/RESULTS.md` now credits idle cash with Treasury bill yields and reports the
  probability of overfitting for every search.

## 2.0.0

- Rebuilt as an installable package with an event-driven engine, walk-forward validation, the deflated Sharpe
  ratio, a block bootstrap, a command-line interface and a test suite with property-based and mutation checks.
