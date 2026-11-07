# Methodology

This document records how TradStrat computes its results and why it makes the choices it does.

## Execution Model

A signal formed at the close of bar `t` is filled at the open of bar `t + 1`. Because the decision and the
fill happen on different bars, a strategy cannot trade at a price it used to make the decision.

**Fills.** A buy fills at `open * (1 + slippage)` and a sale at `open * (1 - slippage)`. Commission is a
percentage of the traded value, charged on both sides. Only whole shares are traded.

**Stops.** A stop rests at `entry * (1 - stop_loss)`. If the bar's low reaches it, the fill is the lower of
the stop price and the open. Without the open, a gap through the stop would fill at a price that never
traded. After a stop the strategy stays flat until its signal resets, or the next bar would buy straight
back in.

**Volatility targeting.** Position size at entry is `min(position_fraction, target / realized_volatility)`,
with volatility estimated up to the previous bar. No position opens until an estimate exists. Sizing
happens at entry only, and positions are not rebalanced while open.

**Liquidation.** Any open position is sold at the final close, so all P&L is realized and the trade log
reconciles with the change in equity.

**Benchmark.** Buy and hold runs through the same simulator, with the same costs, but without the strategy's
position limit, stop or volatility target.

## Data

Yahoo's `Close` is adjusted for splits but not dividends. Using it directly understates the return of
holding a stock: for KO over 2020 to 2024 it gives 13.2% where the total return is 32.2%. TradStrat scales
open, high, low and close by `Adj Close / Close`, so that returns include dividends.

Timestamps are normalized to dates, and high and low are widened if needed to contain the open and close.

## Metrics

Returns are simple daily returns, and annualization uses 252 trading days.

| Metric | Definition |
| --- | --- |
| Sharpe | `mean(r - rf) / std(r - rf) * sqrt(252)`, with sample standard deviation |
| Sortino | `mean(r - rf) / sqrt(mean(min(r - rf, 0)^2)) * sqrt(252)` |
| CAGR | `(final / initial) ^ (1 / years) - 1`, using calendar years |
| Calmar | `CAGR / abs(max drawdown)` |
| Profit factor | Gross profit divided by gross loss |

The Sortino denominator averages over all observations, counting gains as zero. Taking the standard
deviation of the negative returns alone is a common mistake: a strategy that loses the same amount every
time would then appear to have no downside risk.

Undefined ratios are `NaN`, not zero. A flat equity curve has no Sharpe ratio, and scoring it 0.0 would rank
a strategy that never traded above one that lost money.

## Walk-Forward Validation

1. Score every parameter combination on the training window.
2. Take the best combination and trade it on the next window.
3. Roll forward by the length of the test window and repeat.

The bars before each test window warm up the indicators but are not traded, and bars after it are not
passed in at all. Capital compounds across folds. Each fold starts flat and is liquidated at its end, which
costs one extra round trip per fold and so errs slightly against the strategy.

**Efficiency** is the mean out-of-sample score divided by the mean in-sample score. Near 1 suggests that the
parameters generalized, and near 0 suggests they fitted noise.

## Deflated Sharpe Ratio

If you try enough configurations, the best will look good by luck alone. For `N` trials with Sharpe variance
`V`, the expected best Sharpe under no skill is approximately:

```text
SR* = sqrt(V) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N * e)))
```

Here `Z` is the inverse normal CDF and `g` is the Euler-Mascheroni constant. The deflated Sharpe ratio is the
probability that the true Sharpe exceeds `SR*`, given the sample length `T`, skewness `s` and kurtosis `k`:

```text
DSR = Phi((SR - SR*) * sqrt(T - 1) / sqrt(1 - s * SR + (k - 1) / 4 * SR^2))
```

All Sharpe ratios in these formulas are per-period, and `k` is 3 for a normal distribution.

**Against zero and against the benchmark.** The standard test asks whether the true Sharpe exceeds what
selection alone explains. A long-only strategy can pass that by owning a rising market. TradStrat therefore
also applies the test to active returns (strategy minus benchmark), and it bases its verdict on that result.

**Known conservatism.** Neighbouring parameters give correlated results, so `N` overstates the number of
independent trials. In simulations of pure noise, the test's false-positive rate is also well below its
nominal 5%, because the winner must clear the expected maximum by 1.645 standard errors of a single Sharpe
estimate, and that is wider than the spread of the maximum itself. A low score is therefore weaker evidence
than a high score.

## Block Bootstrap

Daily returns are not independent: volatility clusters, and a long/flat strategy's returns depend on its
position. A plain bootstrap would destroy that structure, so TradStrat resamples circular blocks of 20 days.
Strategy and benchmark share the same resampled dates, so their comparison uses returns that occurred
together. The default is 2,000 resamples.

## Synthetic Dataset

The bundled data is a seeded Markov regime-switching random walk, with bull, bear and sideways regimes and
fat-tailed shocks. It exists so that the demo and tests run offline. It has no real-world meaning.
