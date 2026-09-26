# Plain-English backtester

Describe a trading strategy or a portfolio in a sentence and get a full, verifiable backtest on
daily data. The data covers the Nasdaq-100 (today's members and former members, with point-in-time
index membership since 2004), QQQ, SPY, about 60 ETFs (leveraged, bond, gold, sector, VIX) and major
indexes. History goes as far back as each ticker's: AAPL from 1980, MSFT from 1986, the Nasdaq-100
index from 1985.

```bash
pip install -r requirements.txt
python -m backtester web                  # the site: http://localhost:8000
python -m backtester "buy at the close Microsoft when it trades down 5 days in a row, hold for 1 day, and sell at the close"
```

## The site (`python -m backtester web`)

| Page | What it does |
|---|---|
| **Backtest** | Type a strategy. The interpretation updates as you type, and the full report appears on the page. You can download Excel, CSV and JSON, save a PDF, or copy a share link that carries the full spec and settings, so opening it re-runs exactly the same backtest. "Today's orders" turns the strategy's current target into buy/sell orders for your account value and holdings (generic CSV, or an Interactive Brokers basket file). |
| **Build** | A block editor for portfolios (weighted groups, if/else switches and top-N filters, nested as deep as you like), like a Composer symphony: indicator pickers for conditions and rankings, eight weightings (equal, specified, inverse volatility, risk parity, min variance, max Sharpe, max diversification, market cap), drag and drop, duplicate, inline checks, leverage and expense ratio. It also has a form for every field of a signal strategy. Both convert to and from JSON files and from sentences, and both offer "Today's orders". |
| **Gallery** | Library strategies and saved runs with their headline numbers. Fork one into the editor, or export/import a strategy JSON file. |
| **Compare** | Put several strategies (from history or typed) in one report. Every column is compared over the same period. |
| **Research** | A parameter sweep (`hold {1..5} days`) with a heatmap. Walk-forward optimisation (rolling or anchored). A portfolio optimiser: max Sharpe, min variance, max Sortino, min CVaR (95%), risk parity, max diversification, target return, target volatility, inverse volatility and equal weight, with per-asset and group limits (`SPY+QQQ <= 70%`), the efficient frontier, an out-of-sample check and rolling (walk-forward) re-optimisation compared with the static weights. |
| **Monte Carlo** | Thousands of simulated futures for a portfolio (tickers and weights, a sentence or a saved run): percentile bands of the balance (nominal and after inflation), chance of success over time, safe and perpetual withdrawal rates, return and drawdown percentiles. |
| **Factors** | Regress a ticker, portfolio, sentence or saved run on CAPM, Fama-French 3, Carhart 4, Fama-French 5 or FF5 + momentum (monthly or daily): loadings with t-stats, R², annualised alpha and rolling 36-month loadings. |
| **Signals & paper** | Shows what a strategy says to do on the latest bar: new entries, open positions and target weights. You can also start a forward test ("paper trading") that only uses data arriving after you saved it. |
| **History** | Saved runs, with open, edit, share and delete. |
| **Data** | Data freshness, coverage and a ticker browser. |

## What you can write

**Signal strategies** buy and sell on rules:

```text
buy Microsoft at the close when it is down 5 days in a row, hold 1 day and sell at the close
buy AAPL at the close when RSI(2) is below 10 and it is above its 200-day moving average, sell when it closes above its 5-day moving average
buy Nasdaq 100 stocks when RSI(2) is below 5 but only if SPY is above its 200-day moving average, sell when RSI(2) > 70 or after 10 days, max 5 positions, since 2005
short QQQ at the open when it gaps up 1%, cover at the close
buy QQQ when it crosses above its 50 day moving average, sell when it crosses back below
buy AAPL when MACD crosses above its signal line, sell when MACD crosses below its signal line, 5 bps slippage
buy NVDA when it closes above its 20 day high, with a 2 ATR stop and a 3 ATR trailing stop, risk 1% per trade
buy AAPL at a limit 2% below the close when RSI(2) is below 10, hold 3 days
go long SPY when it closes above its 200 day moving average, go short SPY when it closes below it
buy SPY when VIX is above 30, hold 10 days, vs QQQ
buy Nasdaq 100 stocks when RSI(2) is below 5, sell half at +3%, take profit at 6%, stop loss 5%
```

**Portfolios** allocate by weights, regimes and rotations:

```text
hold 60% SPY and 40% TLT, rebalance quarterly           60/40 SPY/TLT     SPY 60%, TLT 30%, GLD 10%
buy and hold QQQ, add $500 every month
hold 60% SPY and 40% AGG, withdraw 4% per year adjusted for inflation, starting with $1,000,000
hold 70% QQQ and 30% TLT, rebalance when any weight drifts 5% from target
equal weight SPY, QQQ, TLT and GLD, rebalance monthly
inverse volatility weighted SPY, TLT and GLD using a 60 day lookback
if SPY is above its 200-day moving average hold QQQ, else if TLT is above its 50 day moving average hold TLT, otherwise hold cash
hold QQQ when it is above its 10-month moving average, otherwise cash, rebalance monthly
hold the top 5 Nasdaq 100 stocks by 6 month momentum, inverse volatility weighted, rebalance monthly
hold the top 2 of QQQ, SPY, TLT and GLD by 3 month return, only if their 3 month return is positive, otherwise hold BIL
rotate monthly between QQQ, SPY and TLT by 3 month return
dual momentum between SPY and EFA with AGG as the safe asset
hold the top 3 sector ETFs by 6 month momentum
if the 10 day RSI of QQQ is greater than 79 then buy UVXY else buy TQQQ
if QQQ 10 day RSI is greater than SPY 10 day RSI then hold QQQ else hold SPY
if SPY is above its 200 day moving average then (if TQQQ RSI(10) is above 79 then hold UVXY else hold TQQQ) else (if SPY RSI(10) is below 30 then hold TECL else hold BIL)
if TQQQ 6 day cumulative return is less than -12% then hold TECL else hold the top 2 of TQQQ, SOXL and TECL by 10 day cumulative return, inverse volatility weighted
hold 50% QQQ and 50% (if SPY is above its 200 day moving average then TLT else GLD)
risk parity SPY, TLT, GLD and DBC over 90 days          minimum variance weighted SPY, TLT and GLD using a 60 day lookback
hold 60% SPY and 40% TLT with 2x leverage and a 0.5% expense ratio       hold 120% SPY and -20% TLT
hold 60% SPYSIM and 40% TLTSIM, rebalance yearly, since 1972
add $1,000 a month for 20 years, then withdraw $50,000 a year, hold 60% VTI and 40% BND
hold 70% QQQ and 30% TLT, rebalance quarterly or when any weight drifts more than 5%
```

Portfolios with if-conditions are checked **every day** by default (as in Composer); pure weight and
top-N trees rebalance monthly unless you say otherwise. Filters and weightings can also rank or
weight whole groups: in a JSON spec or the Build editor, any node can sit inside a filter, and it is
measured on its own simulated value over time. **Composer symphonies** can be imported directly:
`python -m backtester import-composer symphony.json --run`, or "Import Composer symphony" on the
Build page.

**Vocabulary** (numbers can be words):

| Idea | Examples |
|---|---|
| Streaks and moves | down N days in a row, after 3 down days, down *exactly* N days, drops 2% in a day, up 10% over 5 days, gaps down 1%, 10% below / within 2% of its 52-week high |
| Averages | above/below/crosses its N-day (or N-week / N-month) SMA/EMA, "the 9 EMA crosses above the 21 EMA", "the 50 MA", "20 period EMA", EMA(9), 50-day MA above the 200-day MA, golden/death cross |
| Oscillators | RSI(2) below 10, RSI crosses above 30, RSI(2) falls back below 30 / rises back above 70 (crossings), stochastic below 20, ADX above 25, CCI, Williams %R, MFI, MACD crosses its signal / turns positive, +DI above -DI, ATR(14) above 2% of price |
| Bands and trends | Bollinger ("the upper / middle / lower band" = 20-day SMA ± 2 sd), Keltner, Supertrend, Parabolic SAR, VWAP (rolling 20-day volume-weighted typical price on daily bars; "its 10 day VWAP"), closes above its 20-day high (breakout), new N-day low, all-time high, IBS, inside day, volume twice its 20-day average |
| Other tickers | "…and SPY is above its 200-day moving average", "VIX is above 30", "sell when SPY closes below it" |
| Calendar | on Mondays, in October, last / first / third / second-to-last trading day of the month, first 3 trading days of the month |
| Ranking | "buy the 5 Nasdaq 100 stocks with the lowest RSI(2) each day, hold 3 days" (up to 5 positions; free slots go to the lowest RSI(2)) |
| Entries | at the close / at the open (same day, rules must be knowable at the open) / next open; limit or stop orders ("a limit 2% below the close"); pyramiding |
| Exits | hold N days, sell when …, "sell after 10 days or when RSI is above 70", "sell when it's over 70" (*it* = the entry's indicator; refused if the entry has several), a bare "RSI" takes the entry's period, "sell when it crosses back below", "sell at the open when …" (same open if the rule is known at the open, e.g. a gap; otherwise checked at the close and sold at the next open), "cover at the next open when …", % stop, ATR stop, trailing / chandelier stop, take profit, sell half at +X%. "buy TSLA while …" / "hold TSLA when …" with no exit: in the market while the condition holds |
| Sizing | max N positions, X% per position, risk X% per trade, target X% volatility, $X or N shares per trade, 2x leverage |
| Costs | bps or % slippage, volume-based slippage / market impact, $ per trade, $ per share, % commission, IBKR commissions (fixed or tiered), borrow fee, margin rate, short rebate X% below T-bills, 30% maintenance margin / no margin calls, cap at X% of volume |
| Portfolios | %-weights, 60/40, equal / inverse-volatility / market-cap weight, if/else-if/otherwise, top/bottom N by momentum/RSI/volatility, rebalance daily…yearly or on drift, contributions, withdrawals, inflation indexing |
| Higher timeframes | the weekly RSI is above 50, weekly RSI(14), the monthly 10 SMA, weekly 20 EMA (computed on completed weeks/months only) |
| More signals | ROC(10) above 5 / rate of change, %K crosses above %D, MACD histogram turns negative, yesterday's high, not on Fridays, except in October, buy stop 1% above the close / at yesterday's high |
| Portfolio conditions | any indicator phrase compared with a number or another ticker's indicator: "TQQQ 6 day cumulative return is less than -12%", "the 10 day max drawdown of TQQQ is above 20%", "SPY 10 day standard deviation of return is above 2%", "QQQ's 3 month return beats TLT's" (total returns) |
| Schedules and flows | semi-annually, relative bands ("drifts 25% relative to its target"), schedule + band, contributions/withdrawals for N years / starting in YEAR / from year N, growing X% a year |
| Other | starting with $X, since/from/until YEAR, vs TICKER (incl. SPYSIM), versus T-bills, cash earns nothing, using today's members only |

Anything else can be written in the **rule language** inside backticks
(`` `zscore(close, 20) < -2` ``). See `python -m backtester --help-expr` for about 60 functions and
variables. If any part of a sentence isn't understood, the tool **refuses and names the words**. It
never quietly drops them or swaps in a different ticker.

## The report

- Headline tiles, an equity curve against SPY, QQQ and the stock's own buy-and-hold, and drawdowns.
- A head-to-head table: every strategy and benchmark over the same period, with CAGR, real CAGR,
  volatility, Sharpe/Sortino (daily and monthly), Calmar, max drawdown, time underwater, Ulcer index,
  best/worst year and VaR/CVaR.
- Risk and return against T-bills. Versus the benchmark: beta, alpha, R², up/down capture, tracking
  error, information ratio and Treynor.
- Trades: win rate, payoff, profit factor, expectancy, MAE/MFE, streaks, t-stat, and a long/short
  split, all on closed trades; positions still open at the end are shown separately as "Open P&L".
  An account that loses everything shows a CAGR of -100%.
- A price chart with every entry and exit marked and the rule's indicators overlaid. Click a trade
  to jump to it. Every traded ticker can be charted: the most-traded ones are embedded in report.html and
  the rest load from `charts/<TICKER>.js` next to it (keep that folder with the report).
- Benchmarks: SPY and QQQ, the stock itself, and SPYSIM (the US market spliced into SPY) as the main
  benchmark when the run starts before SPY existed. The head-to-head shows the whole period, with
  benchmarks that start later marked "from", or the common period. The equity chart has an
  after-inflation view.
- When indicators need a warm-up (a 200-day average on the first bars of the data), the statistics start
  on the first day every rule has a value, and the notes say so.
- Allocation over time and current holdings for portfolios. A cash-flow summary with money-weighted
  IRR.
- Returns by year (partial years flagged) and a monthly heatmap.
- Rolling 12-month and 3-year return, Sharpe, beta and volatility, plus rolling-period best/worst.
- The deepest drawdowns, and how the strategy did in 12 historical crises.
- Fama-French 5-factor + momentum regression, and a correlation matrix.
- For portfolios: P&L by holding (sales − purchases − costs + dividends + value still held; with
  interest and fees it adds up to the gain after cash flows, to the cent), benchmarks that receive
  the same contributions and withdrawals, the account value in today's dollars, and with
  withdrawals the safe and perpetual withdrawal rates over the tested history.
- Benchmarks are bought at the close of the strategy's first bar, like the strategy.
- A Monte Carlo block bootstrap (with "chance the money lasts" when there are withdrawals) and a
  transaction-cost sensitivity table.
- Exports: HTML, Excel, CSV (trades, orders, equity, holdings, yearly, monthly), JSON, and PDF
  (browser or `--pdf`).

## Command line

```bash
python -m backtester "sentence" [--dry-run] [--capital 50000] [--start 2010-01-01] [--slippage-bps 5] [--benchmark QQQ] [--pdf]
python -m backtester compare "60/40 SPY/TLT rebalanced quarterly" "buy and hold QQQ" "if SPY is above its 200-day moving average hold QQQ, otherwise hold TLT"
python -m backtester sweep "buy QQQ when RSI({2..5}) is below {5..25 step 5}, hold {1,3,5} days" --objective sharpe
python -m backtester walkforward "buy QQQ when RSI(2) is below {5..25 step 5}, hold {1,3,5} days" --in-sample 5 --out-sample 1
python -m backtester optimize SPY QQQ TLT GLD --max-weight 0.6 --test-start 2018-01-01
python -m backtester optimize SPY QQQ TLT GLD --constraint "SPY+QQQ <= 70%" --constraint "GLD <= 20%" --target-vol 0.10 --rolling 12 --lookback 60
python -m backtester montecarlo --weights "SPY 60 TLT 40" --balance 1000000 --years 30 --withdrawal 40000 [--model historical|normal|t|forecast]
python -m backtester montecarlo "hold 60% SPY and 40% AGG, withdraw 4% per year adjusted for inflation, starting with $1,000,000" --model t
python -m backtester factors QQQ --model ff5 --freq monthly          # or --weights "SPY 60 TLT 40", a sentence, --run ID
python -m backtester signals "buy Nasdaq 100 stocks when RSI(2) is below 5, hold 3 days" [--webhook URL]
python -m backtester paper add "hold the top 5 Nasdaq 100 stocks by 6 month momentum" --name mom5 ; python -m backtester paper report
python -m backtester --tickers MSFT --entry "down_days >= 5" --hold 1        # explicit rules
python -m backtester --spec reports/<run>/strategy.json                      # re-run exactly
python -m backtester tickers                                                 # what data exists
```

## How the simulation works

- **Prices and dividends.** Prices are daily and split-adjusted, as quoted. Dividends are paid in
  cash on the ex-date, and short positions pay them. Idle cash earns the 3-month T-bill rate;
  borrowed cash pays it plus any margin rate.
- **Shorts and margin (signal strategies).**
  - Short sale proceeds earn the T-bill rate minus `short_rebate_spread` (default 0.25%/yr, floored
    at zero), like a broker's short rebate. "full short rebate" sets it to 0, "no short rebate" to
    nothing earned.
  - With leverage above 1x or any short, `maintenance_margin` (default 25%) is checked at every
    close: if equity / gross exposure is below it, every position is cut pro rata at that close back
    to the initial margin (1/leverage). The trades are marked "margin call" and the notes list the
    dates. Leverage above 4x needs a lower maintenance margin ("with a 15% maintenance margin");
    "no margin calls" turns the check off.
- **Broker costs (signal strategies).**
  - `commission_model: "ibkr_fixed"` ("IBKR commissions"): $0.005/share, min $1, max 1% of the
    trade value per order.
  - `"ibkr_tiered"` ("IBKR tiered"): $0.0035/share, min $0.35, max 1% of trade value, plus about
    $0.0002/share of exchange, clearing and regulatory fees (an approximation of the first tier).
  - `slippage_model: "volume"` ("volume-based slippage", "market impact"): each fill pays, on top of
    any fixed `slippage_bps`, `spread_bps / 2 + impact_bps × sqrt(order shares / ADV20)` basis points,
    where ADV20 is the average volume of the 20 bars before the order (defaults: 2 bps spread, 100
    bps impact coefficient, so 1% of ADV costs 1 + 10 = 11 bps). Tickers without volume pay only the
    half spread.
- **Holding periods.** "hold N days" exits exactly N bars after the entry bar, at the stated exit fill,
  whatever the entry fill: buy at the close + hold 1 + sell at the close is the next day's close; buy at the
  next open + hold 3 is the close 3 bars after the entry day. With an entry at the open, "sell at the close"
  and no holding period means that same day's close (hold 0).
- **Pyramiding (TradingView rules).** With `pyramiding` 1 (the default), an entry signal that fires while the
  ticker's position is already held is ignored, not queued: a next-open order from a day the position was
  still open never fills after a stop closes it at that open. A position closed at a day's close is flat, so
  that day's signal is valid.
- **Sizing is what the summary says.** "200% per position" raises leverage to 2x (with a note) instead of
  being capped at 100%; a long stop of 100% or more, and a starting capital of zero or less, are refused.
  Volume caps on fills at the open use the previous day's volume.
- **Bar ordering.** Each bar runs overnight financing and dividends → exits at the open → entries at
  the open (market, then limit/stop) → intraday stops and targets → exits at the close → entries at
  the close.
  - A stop and a target touched on the same bar count as the stop.
  - Gaps through a stop fill at the open.
- **No lookahead.**
  - "At the close" rules use that day's data and fill at the close (market-on-close).
  - "At the open" rules may only use data known at the open. Anything else is automatically checked
    on the previous close, and a spec that breaks this is rejected.
  - Negative offsets are rejected.
  - Tests truncate all data at a date and check that no earlier trade changes.
- **Survivorship.** "Nasdaq 100 stocks" means point-in-time membership from 2004 (monthly snapshots
  of the index list). A stock is only bought while it was in the index, and former members are
  included where price history exists.
  - About half of former members (mostly acquired companies) have no free price history, so some
    bias remains. The report says so.
  - Before 2004 the earliest known list is used.
- **Portfolios.** Targets are re-evaluated on the schedule (month-end close by default) and traded at
  the close or next open. Only the differences are traded, and new contributions buy the target mix.
- **Returns and costs.** Returns are time-weighted, so cash flows don't distort CAGR or Sharpe. The
  money-weighted IRR is reported separately. Costs default to zero, and the report always shows a
  cost-sensitivity table.

## Data

`scripts/fetch_data.py` runs in the **Fetch price data** GitHub Action every weekday after the US
close and commits updates, so `git pull` gets fresh data. It downloads:
- prices from Yahoo Finance for current and former Nasdaq-100 members, ETFs and indexes (Stooq as a
  fallback for delisted names)
- point-in-time membership reconstructed from the Wikipedia article's revision history, with the live
  list from stockanalysis.com, Wikipedia or Nasdaq for the current month
- the T-bill rate and CPI from FRED
- Fama-French factors from Kenneth French's data library
- share counts for market-cap weighting

**Delisted former members.** Yahoo drops companies that were acquired or went bankrupt (Celgene,
Xilinx, Activision, Yahoo, …), which is the main survivorship gap. Add a free API key as a repository
secret and the data job fills them in automatically, a batch per run:
`TIINGO_API_KEY` (tiingo.com, about 400 names a run) or `ALPHAVANTAGE_API_KEY` (alphavantage.co, 20 a
run on the free tier). Every report states the current member-month coverage.

The **Daily signals** Action then scans the paper-trading strategies (`paper/*.json`) and writes
`signals/latest.md`. It also posts to a webhook if you add a repository secret `ALERT_WEBHOOK_URL`
(for example a Slack or Discord incoming webhook).

### Long-history series (SPYSIM, TLTSIM, IEFSIM, IEISIM, SHYSIM, BILSIM, VBSIM, VBRSIM, VTVSIM, VUGSIM, EFASIM, GLDSIM)

The data job also builds simulated total-return indexes that extend funds back before they
existed, then continue with the real fund's total return:

| Series | Before the fund | Then |
|---|---|---|
| SPYSIM | US stock market (Fama-French market return, from 1926) | SPY |
| TLTSIM | 20-year Treasuries priced from FRED constant-maturity yields | TLT |
| IEFSIM | ~9-year Treasuries from the 10-year yield | IEF |
| SHYSIM | 2-year Treasuries from the 2-year yield | SHY |
| BILSIM | 1-month T-bills (Fama-French RF) | BIL |
| IEISIM | 5-year Treasuries from the 5-year yield (from 1962) | IEI |
| VBSIM | US small caps (Fama-French small portfolios, from 1926) | VB |
| VBRSIM | US small-cap value (Fama-French small / high book-to-market) | VBR |
| VTVSIM | US large-cap value (Fama-French big / high book-to-market) | VTV |
| VUGSIM | US large-cap growth (Fama-French big / low book-to-market) | VUG |
| EFASIM | Developed markets ex-US (Fama-French, from 1990) | EFA |
| GLDSIM | Gold (World Bank monthly average price, stepped daily, from 1960) | GLD |

They are total-return indexes: `close` = `adj_close`, no dividends, `volume` 0 and
open = high = low = close. Use them in the optimiser, Monte Carlo and factor pages (and in JSON
specs) for many more market regimes than the ETFs alone. Keep in mind:
- the early parts are models, not tradable funds (no fees, no bid/ask);
- rules that need intraday prices or volume (gaps, ranges, ATR, volume caps, MFI/VWAP) are
  meaningless on them;
- Fama-French portfolios are gross of costs and GLDSIM moves in monthly steps before 2004;
- sentences can name them like any ticker ("hold 60% SPYSIM and 40% TLTSIM", "vs SPYSIM"). A
  portfolio that starts before SPY existed is compared with SPYSIM by default.

## Monte Carlo

Monthly steps. Return models: **historical** (block bootstrap: whole months are drawn together for
every asset and CPI, in blocks of consecutive months, so correlations, the link with inflation
and short-term momentum are kept), **normal** and **Student-t** (historical mean and covariance;
the t's degrees of freedom are fitted by maximum likelihood), and **forecast** (your expected
return and volatility per asset with historical correlations). Inflation is bootstrapped from CPI
or fixed. Cash flows (a $ amount, inflation-indexed or not, or a % of the balance) are taken at
the start of each period, pro rata, and the portfolio is rebalanced on its schedule.
- *Success*: money left at the horizon.
- *Safe withdrawal rate*: the largest inflation-adjusted yearly withdrawal, as a share of the
  starting balance, that succeeds in at least the target share of paths (95% by default).
- *Perpetual withdrawal rate*: the largest such withdrawal that keeps the median final balance, in
  today's dollars, at the starting balance.

A sentence or saved run with fixed weights is simulated from its assets; one with rules resamples
the strategy's own monthly returns.

## Tests

```bash
python -m pytest -q
```

About 50 tests cover:
- fills, stops, gaps, limits, pyramiding, scale-outs, leverage and dividends on synthetic data
- the no-lookahead invariance tests on real data
- exact reproduction of the MSFT example, a gap-short and a 60/40 portfolio against independent
  calculations
- contribution accounting and point-in-time filtering
- the parser (including refusals)
