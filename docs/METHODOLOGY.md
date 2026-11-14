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

**Trailing stops.** A trailing stop rests at `peak * (1 - stop_loss)`, where `peak` starts at the entry fill
and rises to the highest high of each completed bar. The current bar's high is added only after its low has
been checked. A daily bar does not say whether its high came before its low, and assuming it did would let the
stop ratchet up and fill on the same bar, which is a quiet form of look-ahead.

**Cash interest.** Uninvested cash earns interest before each open, at the rate known at the previous close,
over the calendar days since that close: `cash * rate * days / 365.25`. A Monday therefore collects the
weekend's interest. The rate is either fixed or the 13-week US Treasury bill yield (Yahoo's `^IRX`), carried
forward onto each trading day. Interest is recorded per bar, so the trade log plus interest reconciles exactly
with the change in equity.

**Volatility targeting.** Position size at entry is `min(position_fraction, target / realized_volatility)`,
with volatility estimated up to the previous bar. No position opens until an estimate exists. Sizing
happens at entry only, and positions are not rebalanced while open.

**Liquidation.** Any open position is sold at the final close, so all P&L is realized and the trade log
and interest reconcile with the change in equity.

**Benchmark.** Buy and hold runs through the same simulator, with the same costs, but without the strategy's
position limit, stop or volatility target.

## Data

Yahoo's `Close` is adjusted for splits but not dividends. Using it directly understates the return of
holding a stock: for KO over 2020 to 2024 it gives 13.2% where the total return is 32.2%. TradStrat scales
open, high, low and close by `Adj Close / Close`, so that returns include dividends.

Timestamps are normalized to dates, and high and low are widened if needed to contain the open and close.

CSV files are matched by column name without regard to case, so exports that use `close`, `adj_close` or
`timestamp` load as they are. Dates may be text or Unix timestamps in seconds or milliseconds. Volume is
optional, because index and currency data often have none.

## Custom Rules

A custom rule is parsed into comparisons between indicators and numbers, and compiled into a strategy class
whose parameters are the rule's numbers. Repeated indicators, such as `sma(200)` in both the buy and the sell
rule, share one parameter, so tuning cannot give the two sides different windows.

Every indicator on a bar uses data up to and including that bar's close. `high(n)` and `low(n)` cover the
previous `n` bars and exclude the current one, so a close can break above them. A cross requires both sides to
be defined on the previous bar, so a series that starts above a line has not crossed it. Indicators are
undefined during their warm-up, and a comparison with an undefined side is false, so no rule can trade before
it has enough history.

Without a sell rule, the position is held only while the buy rule is true. With one, the position is entered
when the buy rule fires and held until the sell rule fires, as the built-in reversal and breakout strategies
do. The tests check that the built-in moving average, momentum, Donchian and RSI strategies, written as rules,
produce identical positions.

## Metrics

Returns are simple daily returns, and annualization uses 252 trading days. The risk-free return `rf` is
the cash return for each bar, which is zero unless a cash rate is set.

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

Measuring Sharpe on returns above cash matters once cash earns interest. A strategy that sits in Treasury
bills earns their yield with almost no volatility, and a Sharpe ratio measured against zero would read that
as exceptional skill. Against cash its excess return is zero and its Sharpe is undefined, which is correct.
Alpha and beta are estimated on excess returns for the same reason.

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

## Probability of Backtest Overfitting

The deflated Sharpe ratio corrects one number. The probability of backtest overfitting (Bailey, Borwein,
Lopez de Prado and Zhu, 2017) tests the search procedure itself, using combinatorially symmetric
cross-validation:

1. Record the daily excess returns of every configuration in the search, as a matrix with one column each.
2. Cut the rows into `S` contiguous blocks, 16 by default.
3. For each of the `C(S, S/2)` ways to pick half the blocks (12,870 for 16), call those blocks in-sample and
   the rest out-of-sample. Find the configuration with the best in-sample Sharpe ratio and its rank `r` among
   all `N` configurations out of sample.
4. Convert the rank to a logit: `w = r / (N + 1)`, `logit = ln(w / (1 - w))`.

The probability is the share of splits with a logit at or below zero, which means the in-sample winner ranked
at or below the median out of sample. Choosing at random gives about 50%, and a real, persistent edge gives
close to 0%. Every block is used both in-sample and out-of-sample, so there is no single train/test boundary
to choose, and the result does not depend on one lucky split.

Ties share the average rank, and a configuration that never traded has an undefined Sharpe ratio and ranks
last. The calculation is vectorized from per-block sums, so all 12,870 splits take well under a second.

## Stress Tests

A backtest is one path through one stretch of history, with one set of costs. The stress test reruns a
finished backtest after changing one of those, and asks whether the result survives. It answers six questions
that are often asked of backtests shared online, in a fixed order. Two are answered by how TradStrat works,
and four by computation.

1. **Was the market picked with hindsight?** TradStrat tests one asset at a time, so there is no
   constituent list to suffer from survivorship bias. Choosing an asset because of how it later performed is
   still hindsight, and no computation can detect it, so the answer says so.
2. **Are the fills realistic?** Fills happen at the next bar's open with slippage, and a stop that the market
   gaps through fills at the open (see Execution Model). Daily bars cannot model intraday quotes, which matters
   for day trading and options, so the answer says so.
3. **Are all the costs included?** The backtest is rerun with commission and slippage doubled, for both the
   rule and holding. The answer warns when doubling costs costs the rule its lead over holding, or more than one
   percentage point of growth a year. Zero costs are flagged as unrealistic.
4. **Does it depend on the start date?** The rule and holding run over windows of equal length (half the
   history, at most five years), starting on the first trading day of each month. Indicators are computed once
   on the full history, which uses no future data because every strategy is causal, so each window starts with
   its indicators warmed up, as a trader starting that month would have had them. Long histories are sampled
   down to 120 evenly spaced windows. The answer reports the share of windows in which the rule grew faster
   than holding (steady from 75%, rarely beats holding below 40%), and the worst start for the rule.
5. **Does it still work without its best trades?** The 5 trades with the highest return are removed by
   replacing the strategy's return on every day from each one's entry to its exit with the cash rate, as if
   the money had stayed in cash. Every other day is left exactly as it was. The answer compares the resulting
   growth with holding, and reports the share of the rule's profit those trades brought in. A rule with 5
   trades or fewer is flagged, since its whole result rests on a handful of decisions.
6. **Did it survive a real bear market?** The deepest non-overlapping falls of at least 20% in the
   buy-and-hold equity curve are found, up to four, each measured from its peak to its bottom. A fall is
   blocked from its peak until the price regains that peak, so the next one found is a separate episode. Falls
   whose bottom lies in a well-known crisis are named after it. The answer counts the falls in which the rule
   lost less than holding, and warns when the history holds no fall of 20%.

The summary counts the four computed checks that pass, leaving out any that the history is too short to run.

## Verdict

The dashboard, command line and HTML report share one rule for their headline:

| Headline | Condition |
| --- | --- |
| Better for the risk, but buy and hold made more money | Higher Sharpe, but a lower CAGR |
| Beat buy and hold, and probably not by luck | Higher Sharpe, and higher in at least 95% of bootstrap resamples |
| Beat buy and hold, but it could be luck | Higher Sharpe, but below that 95% bar |
| A smoother ride, but buy and hold did better | Lower Sharpe, but a shallower maximum drawdown |
| Buy and hold did better | Lower Sharpe and a drawdown at least as deep |

The comparison uses Sharpe ratios on excess returns, so a strategy cannot win by holding cash or by taking
more risk. A higher Sharpe with lower growth is reported as exactly that, because a rule that sits in cash often
earns more per unit of risk while making less money, and calling it a win would contradict the dollar figures
shown beside it. The verdict describes one backtest. Walk-forward validation is still the test of whether a tuned
strategy works on unseen data.

## Block Bootstrap

Daily returns are not independent: volatility clusters, and a long/flat strategy's returns depend on its
position. A plain bootstrap would destroy that structure, so TradStrat resamples circular blocks of 20 days.
Strategy and benchmark share the same resampled dates, so their comparison uses returns that occurred
together. The default is 2,000 resamples, drawn from excess returns.

## Synthetic Dataset

The bundled data is a seeded Markov regime-switching random walk, with bull, bear and sideways regimes and
fat-tailed shocks. It exists so that the demo and tests run offline. It has no real-world meaning.
