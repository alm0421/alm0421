# Plain-English backtester

Describe a trading strategy in a sentence; get a full backtest on daily data for the
Nasdaq-100 constituents, QQQ and SPY, back as far as each ticker's history goes
(AAPL/INTC/AMD from 1980, MSFT from 1986, SPY from 1993, QQQ from 1999).

```bash
pip install -r requirements.txt
python -m backtester "buy at the close Microsoft when it trades down 5 days in a row, hold for 1 day, and sell at the close"
```

The tool first prints **how it understood the sentence**, so you can check the rules before
trusting the numbers:

```
Buy MSFT at the close of the signal day when: (down_days >= 5)
Exit: after 1 bar(s) at the close
Sizing: $10,000 start, up to 1 position(s) at 100% of equity each
```

and then the results, plus a report folder under `reports/<strategy-name>/`:

| File | Contents |
|---|---|
| `report.html` | Self-contained interactive report (open in any browser; light/dark) |
| `trades.csv` | Every trade: dates, fills, shares, P&L, return, bars held, exit reason, MAE/MFE |
| `equity.csv` | Daily equity, drawdown, exposure, positions, and benchmark buy-and-hold curves |
| `yearly.csv`, `monthly.csv` | Calendar returns |
| `summary.json`, `strategy.json` | All statistics; the exact rules used (re-run with `--spec strategy.json`) |

## What's in a report

- **Headline**: final equity from $10,000, total return, CAGR, Sharpe, Sortino, Calmar, volatility
- **Equity curve** (log/linear) against buy-and-hold of the traded stock, SPY and QQQ
- **Drawdowns**: underwater chart, max drawdown with peak/trough/recovery dates, the 5 deepest
  drawdowns, longest time under water, Ulcer index
- **Returns by year** (strategy vs benchmarks, with trades, win rate, exposure, max DD per year) and a
  **monthly returns heatmap**
- **Trade statistics**: win rate, average/median trade, average win vs loss, payoff ratio, profit
  factor, expectancy, streaks, bars held, MAE/MFE, t-statistic of the mean trade
- **Market exposure**: time in market, average exposure, beta / alpha / correlation vs SPY
- **Robustness**: Monte Carlo block bootstrap (5th–95th percentile CAGR and drawdown), and the same
  rules re-run at 0 / 5 / 10 / 25 bps slippage per side
- **Every trade**, sortable and filterable

## Writing strategies

A sentence has an **entry** (who, when, at what price) and one or more **exits**.

```text
buy Microsoft at the close when it is down 5 days in a row, hold 1 day and sell at the close
buy AAPL at the close when RSI(2) is below 10 and it is above its 200-day moving average, sell when it closes above its 5-day moving average
buy Nasdaq 100 stocks at the close when the 2-day RSI is below 5, exit when RSI(2) > 70 or after 10 days, max 5 positions, since 2005
short QQQ at the open when it gaps up 1%, cover at the close
buy QQQ at the open when it gaps down 1% and SPY is above its 200 day moving average, sell at the close
buy SPY at the close when it drops 2% or more in a day, hold 5 days, with a 3% stop loss
buy NVDA when it makes a new 20-day low and IBS is below 0.2, sell on the first up close
buy SPY at the close on the last trading day of the month, sell at the close 5 days later
buy MSFT when golden cross, sell with a 10% trailing stop
buy TSLA at the next open when it closes in the bottom 10% of its range, hold 2 days, 5 bps slippage
```

**Understood phrases** (numbers can be words: "five days"):

| Idea | Examples |
|---|---|
| Streaks | down/up N days in a row, N consecutive down days, down *exactly* N days in a row |
| Moves | drops 3% in a day, is up 10% over the last 5 days, falls 5% or more, gaps down 1% |
| Averages | above/below its 200-day moving average (SMA/EMA), 5% below its 20-day MA, crosses above its 50-day MA, 50-day MA above the 200-day MA, golden/death cross |
| Oscillators | RSI(2) below 10, 2-day RSI above 90, IBS below 0.2, closes in the bottom 10% of its range, below the lower Bollinger band |
| Highs/lows | new 20-day low, 52-week high, all-time high, closes above the previous day's high, inside day |
| Volume | volume is 2x its 20-day average |
| Calendar | on Mondays, in October, last/first trading day of the month |
| Market filter | …and SPY is above its 200-day moving average (any other ticker works) |
| Exits | hold N days/weeks, sell at the close/next open, sell when \<condition\>, or after N days, N% stop loss, N% trailing stop, take profit at N% |
| Portfolio | max N positions, N% of equity per position, prefer the lowest RSI / biggest losers, starting with $50,000 |
| Costs & dates | 5 bps slippage, $1 per trade commission, since 2010, from 2005 to 2020 |

**Anything else** can be written in the rule language inside backticks, mixed with English:

```bash
python -m backtester "buy AMD at the close when \`zscore(close, 20) < -2 and volume > 1.5 * sma(volume, 50)\`, take profit at 5%, stop loss 4%, max hold 10 days"
python -m backtester --help-expr      # full vocabulary
```

or skip English entirely:

```bash
python -m backtester --tickers MSFT --entry "down_days >= 5" --hold 1
python -m backtester --tickers NDX --entry "rsi(2) < 5 and close > sma(close, 200)" --exit-when "close > sma(close, 5)" --max-positions 10
```

If the parser doesn't understand part of a sentence it **stops and says which words** rather than
guessing. Use `--dry-run` to see the interpretation without running.

Useful flags: `--capital`, `--slippage-bps`, `--commission`, `--start/--end`, `--max-positions`,
`--position-size`, `--rank-by`, `--whole-shares`, `--rf 0.04` (risk-free rate for Sharpe), `--out`.

## How the simulation works

- **Daily bars**, split- *and* dividend-adjusted (total return). Trade lists also show the quoted
  (split-adjusted) price.
- **"At the close"** means a market-on-close order: the rule is evaluated with that day's closing
  data and filled at the close. **"At the open"** fills at that day's open only if the rule is knowable
  at the open (e.g. a gap); anything that needs the close is checked on the *previous* close, and the
  tool tells you so. **"Next open"** fills at the following day's open.
- **"Hold N days"** exits N trading days after entry (at the close unless you say open).
- Each bar is processed in order: exits at the open → entries at the open → intraday stops/targets
  (a stop and a target touched on the same bar counts as the stop — conservative) → exits at the
  close → entries at the close. Stops that gap through fill at the open, not the stop price.
- **Sizing**: one ticker → 100% of equity per trade. Several tickers → by default up to 10 positions
  at 10% of equity each, no leverage, no pyramiding. When more stocks signal than there are free
  slots, the most liquid (20-day dollar volume) are taken unless you set a ranking.
- **Costs** default to zero; the report always shows the cost-sensitivity table.
- Idle cash earns nothing; Sharpe uses daily returns × √252 and a 0% risk-free rate unless `--rf` is set.
- "Down N days in a row" means *at least* N (it fires again on day N+1 if the streak continues);
  say "exactly N" for only the Nth day.

## Data

`scripts/fetch_data.py` downloads full daily history from Yahoo Finance (via `yfinance`) for the
**current** Nasdaq-100 members (list scraped from Wikipedia / stockanalysis.com / slickcharts /
nasdaq.com) plus QQQ and SPY, into `data/prices/*.csv`. The **Fetch price data** GitHub Action runs it
after every US close on weekdays and commits the update, so a `git pull` gives you fresh data. Run it
by hand from the Actions tab, or locally with `python scripts/fetch_data.py`.

**Caveats you should keep in mind**

- **Survivorship bias**: the universe is *today's* Nasdaq-100. Stocks that were in the index and
  later fell out or went bust are missing, and today's members are, by construction, past winners.
  Multi-stock results are therefore optimistic. The report flags years in which few of the members
  had yet listed.
- Yahoo's history before the late 1990s is occasionally coarse (prices rounded to 1/16ths, some
  zero-range days).
- No borrow costs or short-availability limits are modelled for shorts.

## Tests

```bash
python -m pytest -q
```

Includes synthetic-data tests for fills, stops, gaps, targets, costs and position limits, parser
cases, the expression sandbox, and independent vectorised re-computations of real backtests.
