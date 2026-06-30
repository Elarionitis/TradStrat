# TradStrat

TradStrat is a beginner-friendly Streamlit app for exploring simple trading strategies on historical daily price data. It is designed for learning how backtests work, not for predicting live markets or giving investment advice.

## What it does

- Lets you choose a ticker, date range, strategy, and basic risk settings.
- Runs an event-driven backtest day by day, with next-day-open execution, commission, and fixed position sizing.
- Shows an equity curve, buy-and-hold benchmark, trade markers, drawdown chart, and summary metrics.
- Includes an offline sample dataset so you can try the app even when Yahoo Finance is unavailable.

## Project Structure

- `app.py` - Streamlit UI and plotting.
- `data_loader.py` - Daily OHLCV fetch + validation + CSV cache.
- `strategies.py` - Strategy classes for moving-average crossover and mean reversion.
- `engine.py` - Event-driven backtesting loop and trade log generation.
- `metrics.py` - Return, risk, and trade-quality metrics.
- `glossary.py` - Plain-English definitions used by the UI help text.
- `sample_data/aapl_sample.csv` - Offline demo dataset.
- `tests/test_metrics.py` - Unit tests for the metrics module.

## How To Run

Install dependencies:

```bash
/bin/python3 -m pip install --break-system-packages -r requirements.txt
```

Start the app:

```bash
cd "/media/elarion/6715dcac-8957-4b06-8f07-70d7f123b1b4/home/Projects/Trading Strategy Backtester/TradStrat"
/bin/python3 -m streamlit run app.py
```

If live Yahoo Finance data is not available in your environment, enable `Use sample dataset (offline demo)` in the sidebar.

Run the tests:

```bash
cd "/media/elarion/6715dcac-8957-4b06-8f07-70d7f123b1b4/home/Projects/Trading Strategy Backtester/TradStrat"
/bin/python3 -m unittest tests.test_metrics
```

## Strategies

### Moving Average Crossover

Buys when a short moving average rises above a long moving average and exits when the signal turns off.

### Mean Reversion

Uses a rolling z-score. It enters when price is stretched below its recent average and exits when price moves back toward normal.

## Metrics

- Total Return: percentage gain or loss over the full backtest.
- CAGR: annualized growth rate.
- Sharpe Ratio: average return relative to total volatility.
- Sortino Ratio: average return relative to downside volatility only.
- Max Drawdown: largest peak-to-trough decline.
- Max Drawdown Duration: longest time spent below a previous peak.
- Win Rate: percentage of profitable trades.
- Average Win / Average Loss: average profit and average loss per closed trade.
- Number of Trades: total closed trades.

## Important Limitations

- This app uses daily data only, not intraday tick data.
- Execution is simplified: signals are generated from historical closes and trades are filled at the next day’s open.
- Slippage and commission are simplified and do not model real order-book behavior.
- It does not model latency, partial fills, spread dynamics, borrow costs, or market impact.
- The bundled sample dataset is synthetic and meant for demonstration only.
- Results are educational and should not be treated as trading advice.

## Notes For Beginners

- A backtest is a simulation of how a strategy would have behaved in the past.
- An equity curve is the line showing how portfolio value changes over time.
- The buy-and-hold benchmark is the simplest comparison: buy once and hold through the whole period.
- Look-ahead bias is avoided by using only information available up to the current day and executing on the next open.