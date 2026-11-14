# Changelog

All notable changes to TradStrat. The project follows [semantic versioning](https://semver.org).

## 2.1.0

### Added

- **Custom rules without code.** A small rule language, such as `close crosses above sma(200)`, in the
  dashboard and through `--buy` and `--sell` on the command line. Every number in a rule can be tuned by the
  optimizer and walk-forward validation, and custom rules appear in the strategy comparison.
- **Two more classic strategies:** MACD crossover, and the 200-day trend filter made famous by Mebane
  Faber's study of tactical asset allocation.
- **The latest signal.** The dashboard shows whether the rule would own the asset or hold cash at the latest
  close, and since when.
- **Shareable links.** The address bar holds the market, history and strategy, so a link reopens the same test.
- **Interest on idle cash.** Uninvested cash can earn a fixed rate or the real 13-week US Treasury bill
  yield, accrued over calendar days at the rate known the evening before.
- **Sharpe ratios on excess returns.** Sharpe, Sortino, alpha, the bootstrap and the deflated Sharpe ratio are
  all measured on returns above cash, so holding Treasury bills is never mistaken for skill.
- **Probability of backtest overfitting.** Parameter searches now report how often the in-sample winner falls
  to the bottom half out of sample, using combinatorially symmetric cross-validation over 12,870 splits.
- **Trailing stops** that follow the highest high of completed bars.
- **A plain-English verdict** on every backtest, shared by the dashboard, the command line and the report.
- **A landing page built around questions.** The dashboard opens by explaining what it does in one sentence,
  beside an animated example, and offers six ready-made questions such as *"Does the famous golden cross beat
  the S&P 500?"*. One click runs the test on real prices.
- **A stress test of every result.** A new tab answers six questions worth asking of any backtest: was the
  market picked with hindsight, are the fills realistic, are all costs included (rerun with costs doubled),
  does the result depend on the start date (equal windows starting in up to 120 different months), does it
  still work without its 5 best trades, and did it survive a real bear market (every fall of 20% or more,
  named when it is a well-known crash). The Results tab sums it up in one line, the HTML report includes it,
  and `tradstrat stress` prints it.
- **A 40-second tour.** A short video with an original soundtrack, on the landing page and in the README,
  shows the problem, a real test, a custom rule and the checks that decide whether to believe the result.
- **Help for newcomers.** The dashboard states the rule being tested in one sentence, explains each built-in strategy
  in plain English, and defines jargon such as the Sharpe ratio on hover.
- **A verdict on parameter searches** that combines the deflated Sharpe ratio and the probability of
  overfitting into one plain answer.
- **HTML reports.** `tradstrat report` and a dashboard button write a self-contained report with the verdict,
  interactive charts, robustness statistics and the full trade log.
- **Strategy comparison.** `tradstrat compare` and a dashboard tab run every strategy at its defaults against
  buy and hold.
- **Real-world CSV files.** Column names are matched without regard to case, dates may be Unix timestamps,
  and volume is optional. The dashboard accepts uploads.
- New charts: calendar-year returns, rolling Sharpe ratio, the overfitting distribution, and each strategy's
  own indicator lines on the price chart.

### Changed

- A new look for the dashboard and the HTML report, in the style of a research note: a serif for headlines,
  a plain sans for reading, mono for figures, warm paper and ink colors, and hairline rules in place of
  floating cards. The landing illustration is drawn from a real trend rule on simulated prices, and the
  questions read as a numbered index. "All questions" leads back from any result.

- Ready to run as a public website: friendly messages instead of raw errors when prices can't be loaded,
  with a one-click switch to demo data; size limits on every cache, rule and upload; no tracebacks shown to
  visitors; and a Dockerfile with a health check.
- Plain-language wording on the landing page and in the README: what TradStrat is, why it exists, and who
  it is for.

- The dashboard was redesigned around five tabs: Results, Trades, Tune, Compare and Guide. It leads with the
  verdict, keeps advanced settings in one place, uses plain labels such as "Chance the edge is real", and its
  charts follow the light or dark theme.
- Results read as a head-to-head between the rule and simply holding: growth, return for the risk, the worst
  crash and a luck check, each with a winner, in the chart colors rather than alarm colors. Each verdict
  explains what the result means, and the money line says what $10,000 became each way. Chart lines use the
  same two names.
- Verdicts name the asset, as in "Just holding SPY did better", and the dashboard says what buy and hold
  means next to the rule being tested.
- Search and walk-forward results are cleared when any input changes, so they can never describe other
  settings.
- The research study in `docs/RESULTS.md` now credits idle cash with Treasury bill yields and reports the
  probability of overfitting for every search.

### Fixed

- Choosing all available history returned only the last month for assets that started trading after 1970,
  such as Bitcoin, because yfinance reads a start date of exactly the Unix epoch as no start date.

## 2.0.0

- Rebuilt as an installable package with an event-driven engine, walk-forward validation, the deflated Sharpe
  ratio, a block bootstrap, a command-line interface and a test suite with property-based and mutation checks.
