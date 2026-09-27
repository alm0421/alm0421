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
| **Backtest** | Type a strategy. The interpretation updates as you type, and the full report appears on the page. You can download Excel, CSV and JSON, save a PDF, or copy a share link that carries the full spec and settings, so opening it re-runs exactly the same backtest. Opening a link fills in every setting (TradingView mode, cash interest, dividends, commissions...), and changing one setting re-runs the shared spec with that change (not a re-reading of the sentence). "Today's orders" turns the strategy's current target into buy/sell orders for your account value and holdings (generic CSV, or an Interactive Brokers basket file). Below it, the **asset allocation grid** (as Portfolio Visualizer's "Backtest Portfolio"): rows of tickers or asset-class names ("US Stock Market", "Total Bond Market", with autocomplete) x up to three portfolios of weights (totals per column; each must add up to 100%, unknown tickers are flagged), start/end as a year or a month, initial amount, cash-flow phases (contribute or withdraw, $ or % of the balance per year, frequency, inflation-adjusted, start/end year: one contribution and one withdrawal phase per portfolio), rebalancing (none, monthly, quarterly, semi-annual, annual, or absolute/relative bands), benchmark, expense ratio and leverage. "Run portfolios" builds one Portfolio spec per column and runs them together in a Compare report over their common period (as Portfolio Visualizer: when one column holds an asset whose data starts later, every column starts then, with a warning naming the column and ticker, and cash-flow years and real dollars count from that start); a blended benchmark can use asset-class names ("60% US Stock Market 40% Total Bond Market"); "Copy share link" (`#backtest?g=...`) reopens the filled grid and re-runs it (API: `POST /api/grid`). |
| **Build** | A block editor for portfolios (weighted groups, if/else switches and top-N filters, nested as deep as you like), like a Composer symphony: indicator pickers for conditions and rankings (returns, drawdown, volatility, RSI, SMA/EMA, standard deviation of price, MACD and its signal, PPO and its signal, Bollinger bands), windows checked as typed (a window that is not a whole number of days, at least 1, is marked red and the run refused: never rounded or replaced), the parser's unit and range checks (an RSI threshold outside 0-100 refused, a negative max drawdown read as a fall with a note; the server applies the same checks to JSON and API specs), eight weightings (equal, specified, inverse volatility, risk parity, min variance, max Sharpe, max diversification, market cap), drag and drop, duplicate, inline checks, leverage and expense ratio. It also has a form for signal strategies: controls for the common fields (entry, exits, stops and targets, breakeven, scale-outs, stop/target levels, sizing, ranking, costs, cash interest, dividends, TradingView mode) and an "Advanced fields" JSON box that carries every other field unchanged, so a strategy opened from a sentence, a file, the gallery or a share link runs exactly as loaded (the portfolio editor has the same box). A guard compares the loaded spec with the one the editor would run and refuses a run that would change a field you did not edit (a second click runs it anyway). Both convert to and from JSON files and from sentences, and both offer "Today's orders". |
| **Gallery** | Library strategies and saved runs with their headline numbers. Fork one into the editor, or export/import a strategy JSON file. |
| **Community** | Strategies people published with "Publish to the community gallery" (Backtest and Build pages: a name, an author and a description; the strategy is backtested first). Search, sort by CAGR, Sharpe or max drawdown, **Fork** into the editor or **Run**. Kept in `data/community.json` (`BACKTESTER_COMMUNITY` points elsewhere). |
| **Compare** | Put several strategies (from history or typed) in one report. Every column is compared over the same period; a benchmark whose data starts later than that period is listed separately with its own dates. |
| **Research** | A parameter sweep (`hold {1..5} days`) with a heatmap. Walk-forward optimisation (rolling or anchored). A portfolio optimiser: max Sharpe, min variance, max Sortino, min CVaR (95%), risk parity, max diversification, max return / max drawdown, max Omega (at a threshold return), target return, target volatility, inverse volatility and equal weight (pick any subset), with per-asset and group limits (`SPY+QQQ <= 70%`), the efficient frontier, an out-of-sample check and rolling (walk-forward) re-optimisation compared with the static weights. A target that can't be reached says what can ("the minimum achievable volatility is 9.1%"). **Inputs**: historical means by default, or your expected returns (and optionally volatilities and correlations), or **Black-Litterman** (market-cap, equal or given prior weights plus absolute/relative views with confidences; the posterior feeds every objective). **Benchmark-relative**: min tracking error (optionally with a return floor) and max information ratio against a ticker or blend. **Resampled frontier** (Michaud): average the optimal weights over N simulated histories. |
| **Monte Carlo** | Thousands of simulated futures for a portfolio (tickers and weights, a sentence or a saved run): percentile bands of the balance (nominal and after inflation), chance of success over time, safe and perpetual withdrawal rates, also per percentile of the paths (10th-90th, as Portfolio Visualizer; for contribute-then-withdraw plans measured from the balance when withdrawals start) (not shown for savings plans with contributions only), return and drawdown percentiles. Stress tests (the worst historical 10-year sequence first, or a -30% first year), a horizon set by age ("until age 95") or by the SSA life table (a lifetime, for a man, a woman or a couple, with success weighted by survival), any number of cash-flow phases (contribute, then withdraw), and a glide path (e.g. 90/10 -> 40/60 over 30 years, linear or target-date shaped). |
| **Factors** | Regress a ticker, portfolio, sentence or saved run on CAPM, Fama-French 3, Carhart 4, Fama-French 5 or FF5 + momentum for the US or a region (developed, developed ex US, Europe, Japan, Asia Pacific ex Japan, North America, emerging), AQR's quality (QMJ) and betting-against-beta (BAB) factors, and the bond factors TERM and DEF, monthly (French's official monthly files) or daily: loadings with t-stats, R², annualised alpha and rolling 36-month loadings. **Style analysis** (Sharpe 1992) finds the asset-class mix that best tracks the returns, with rolling 36-month weights. |
| **Correlations** | The correlation matrix of daily or monthly total returns over a chosen period, a rolling correlation of any pair, and per-asset statistics (CAGR, volatility, Sharpe, max drawdown, best/worst year, first date of data; on the matrix's frequency: with monthly returns, volatility and Sharpe from monthly returns and the max drawdown from month-end values), like Portfolio Visualizer's asset correlations. |
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
60% VTI 40% BND                                          VTI 60% BND 40%   VTI 60, BND 40 (bare numbers must add to 100)
buy and hold QQQ, add $500 every month
hold 60% SPY and 40% AGG, withdraw 4% per year adjusted for inflation, starting with $1,000,000
hold 70% QQQ and 30% TLT, rebalance when any weight drifts 5% from target
equal weight SPY, QQQ, TLT and GLD, rebalance monthly
inverse volatility weighted SPY, TLT and GLD using a 60 day lookback
if SPY is above its 200-day moving average hold QQQ, else if TLT is above its 50 day moving average hold TLT, otherwise hold cash
hold QQQ when it is above its 10-month moving average, otherwise cash, rebalance monthly
hold the top 5 Nasdaq 100 stocks by 6 month momentum, inverse volatility weighted, rebalance monthly
hold the top 2 of QQQ, SPY, TLT and GLD by 3 month return, only if their 3 month return is positive, otherwise hold BIL
hold the top 1 of SPYSIM and EFASIM by 12 month return, only if their 12 month return is above BILSIM's 12 month return, otherwise hold IEFSIM
hold 20% each of SPY, EFA, IEF, VNQ, DBC, each only when above its 10 month moving average, otherwise BIL, rebalance monthly
hold 60% SPY and 40% TLT, vs 60/40 SPY/AGG
rotate monthly between QQQ, SPY and TLT by 3 month return
dual momentum between SPY and EFA with AGG as the safe asset
hold the top 3 sector ETFs by 6 month momentum
hold the top 2 of SPY, EFA, TLT and GLD by average of 1, 3, 6 and 12 month return, rebalance monthly
hold the top 2 of SPY, EFA, TLT and GLD by 12 month return skipping the last month      (= "12-1 momentum")
hold the top 2 of SPY, EFA, TLT and GLD by 12 month return divided by volatility         (= "risk-adjusted momentum")
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
if QQQ RSI(10) is above 79 then short TQQQ else hold TQQQ        if SPY is above its 200 day moving average then buy TQQQ else sell TQQQ
unless SPY is below its 200 day moving average hold QQQ, otherwise TLT
hold TQQQ unless both SPY and QQQ 10 day RSI are above 79, in which case hold UVXY
if any of SPY, QQQ and SMH has a 10 day RSI above 79 then UVXY else TQQQ      hold the top 2 of SPY, QQQ, TLT and GLD by 6 month return, weighted 70/30
hold the 2 of SPY, QQQ and IWM with the highest 10 day return    hold the best performing of SPY, QQQ and IWM over 10 days
hold the top 2 of SPY, QQQ and IWM by 60 day return, only if their 60 day return beats BIL's, otherwise TLT
SPY, TLT and GLD in equal thirds        TQQQ and TMF equally        TQQQ, TMF and SVIX weighted 50/30/20
hold the top 2 by 10 day RSI of SPY, QQQ, TLT                  hold the highest 10 day RSI of SPY, QQQ, TLT
if SPY 60 day max drawdown is worse than 10% then hold BIL else hold SPY, rebalance every 2 days
if not (SPY is above its 200 day moving average or QQQ 10 day RSI is above 80) then hold QQQ else hold BIL
hold 25% each of VTI, TLT, GLD and SHY, start 2008, starting balance $1,000,000      60% US stocks 40% bonds 1972-2020
hold the top 3 of SPY, EFA, TLT and GLD by 3 month return weighted 50%, 6 month weighted 30% and 12 month weighted 20%
hold 60% SPY and 40% TLT, rebalance every year in June            (or "rebalance annually in June", "each December")
hold 60% SPY and 40% TLT, withdraw 5% a year taken quarterly      (1.25% of the balance each quarter)
hold 60% SPY and 40% TLT, benchmark 60% US Stock Market and 40% Total Bond Market
```

Portfolios with if-conditions, top-N filters or dynamic weights (inverse volatility, risk parity, ...)
are re-evaluated **every day** by default (as in Composer); fixed-weight trees rebalance monthly unless
you say otherwise. Named model portfolios work as phrases: "golden butterfly since 1972, rebalance
yearly", "three fund portfolio", "all weather", "permanent", "coffeehouse", "ivy", "Bernstein
no-brainer", "60/40 portfolio", "Hedgefundie adventure", "Swensen", "larry portfolio", "Buffett 90/10",
"global market portfolio", "sandwich", "desert", "Merriman ultimate buy and hold", "weird portfolio",
"core four", "talmud", "pinwheel", "7Twelve" (Israelsen), "couch potato" (Scott Burns), "Merriman 4-fund
combo", "Frank Armstrong ideal index", "Bogleheads four-fund", "second grader's starter", "Dave Ramsey",
"Aronson family taxable", and any stock/bond split ("80/20 portfolio", "Stocks/Bonds 60/40", "US stocks
and bonds 70/30 since 1950": US total market and total bond market) (the notes list the holdings, the
source and any proxy fund, e.g. VSS for developed ex-US small caps, and long-history SIM series stand in
before the funds existed). A bare asset-class name holds 100% of it ("long-term corporate bonds since
1955", "US small cap value"). An "N month return" is 21 trading days a month by default (12 months = 252
sessions); add "using calendar months" (or "using month-end prices", or `"month_lookbacks": "calendar"` in a
JSON spec) to measure N-month returns month-end to month-end from the last completed month-end, as Portfolio
Visualizer and Antonacci do. The interpretation says which one a portfolio uses. With calendar months, "12 month
return skipping the last month" (12-1 momentum) is measured on month-end prices: the month-end price one month
ago over the month-end price 12 months ago, minus 1. A weighted mix of lookbacks ("3 month return weighted 50%,
6 month weighted 30% and 12 month weighted 20%") is the weighted average of those total returns (the weights
must add to 100%), each lookback following the same month convention. Filters and weightings
can also rank or weight whole groups ("the top 1 of (60% TECL and 40% BIL), SVIX and TQQQ by 10 day
return"): in a JSON spec or the Build editor, any node can sit inside a filter, and it is
measured on its own simulated value over time. **Composer symphonies** can be imported directly:
`python -m backtester import-composer symphony.json --run`, or "Import Composer symphony" on the
Build page. And exported: `python -m backtester composer-export spec.json` (or a sentence), "Export to
Composer" on the Build page, `api.composer_export(...)`. The export writes what Composer has (assets, equal /
specified / inverse-volatility / market-cap weights, if/else on its indicators with gt, gte, lt, lte and eq, "and"
/ "or" as Composer's all / any condition blocks and "not" by flipping comparators, top/bottom-N filters over
listed tickers, calendar or corridor rebalancing, cash in a branch of an if as Composer's empty block) and
refuses, saying why, what it lacks: shorts, leverage, cash as a holding elsewhere, other indicators or
weightings, Nasdaq-100 universes, requirements on a filter, volatility targeting, a calendar schedule together
with a drift band. The import also reads Composer's any/all conditions ("compound" and "binary-compound"), MACD,
MACD signal, PPO, PPO signal and the lower / upper Bollinger band (`macd`, `macd_signal`, `ppo`, `ppo_signal`,
`bb_lower`, `bb_upper`), the eq comparator, market-cap weights and empty blocks (cash). The corridor width is a
fraction, as Composer stores it (`"rebalance-corridor-width": 0.05` is a 5% threshold, both ways). Importing an
exported symphony gives back the same tree. The condition block matches Composer's own exports (checked against a
public symphony's JSON, `tests/fixtures/composer_frontrunner_2026.json`): beside the block, the if-child repeats its
last comparison in the single-comparison fields, as Composer's editor writes it. MACD, PPO and Bollinger conditions
import but do not export: no public export or schema names the keys of their parameters.

**Vocabulary** (numbers can be words):

| Idea | Examples |
|---|---|
| Streaks and moves | "closed down on Friday" (the previous session was a Friday and closed lower), "closes higher than the high of the previous 3 days", "3 standard deviations below its 20 day mean" (z-score), "pulls back to the 50 day moving average" (today's low reaches it after closing above it the day before), "falls 10% from its peak" (entries: from the running all-time peak; exits: a 10% trailing stop), "the 14 day momentum is above 0" (the price change over 14 days, TradingView's `ta.mom`), "the close is higher than 5 days ago" (`close > ref(close, 5)`), "yesterday's RSI(2) was below 10 and today RSI(2) is above 10" (the first part on the previous bar), "buy SPY 2 days after RSI(2) is below 10" (`ref(cond, 2)`), down N days in a row, after 3 down days, down *exactly* N days, drops 2% in a day, up 10% over 5 days, gaps down 1%, 10% below / within 2% of / less than 10% below its 52-week high, "closes more than 20% above its 52 week low", "below its 50 day moving average by more than 5%" |
| Averages | above/below/crosses its N-day (or N-week / N-month) SMA/EMA, "the 9 EMA crosses above the 21 EMA", "the 50 MA", "20 period EMA", EMA(9), 50-day MA above the 200-day MA, "SPY EMA(8) > SPY SMA(21)", "the 8 day EMA of SPY is above the 21 day SMA of SPY", golden/death cross |
| Oscillators | RSI(2) below 10, RSI crosses above 30, RSI(2) falls back below 30 / rises back above 70 (crossings), stochastic below 20 (TradingView's defaults: %K 14 with smoothing 1, %D = 3-bar SMA of %K; "slow stochastic" = smoothing 3), "%K is below 20", "%D crosses above 80", stochastic RSI ("stochastic RSI below 20", "stoch RSI %K crosses above %D": TradingView's built-in, %K 3, %D 3, RSI 14, stochastic 14 - `stoch_rsi_k` / `stoch_rsi_d`), ADX above 25, CCI, Williams %R, MFI, MACD crosses its signal / turns positive, "MACD(12,26,9) crosses above signal", +DI above -DI, ATR(14) above 2% of price, "the 14 day ATR is above its 50 day average" |
| Bands and trends | Bollinger ("the upper / middle / lower band" = 20-day SMA ± 2 sd), Keltner, Supertrend ("the supertrend direction flips to up"; exit "sell when it flips to down"), Heikin Ashi ("Heikin Ashi turns green" = the HA candle is green today and was not yesterday; "turns red", "is green"), Parabolic SAR, above / below / inside the Ichimoku cloud, VWAP (rolling 20-day volume-weighted typical price on daily bars; "its 10 day VWAP"), closes above its 20-day high (breakout), new N-day low, all-time high, IBS, inside day, volume twice its 20-day average |
| Other tickers | "…and SPY is above its 200-day moving average", "VIX is above 30", "sell when VIX crosses above 30" (the close crossing the level, `crossover(sym("^VIX").close, 30)`), "sell when SPY closes below it"; one comparison on several tickers: "both SPY and QQQ 10 day RSI are above 79", "all of SPY, QQQ and SMH have a 10 day RSI above 79", "SPY and QQQ 10 day return is below -5%" (every one), "either SPY or QQQ …", "any of SPY, QQQ and SMH has …" (at least one), "the 10 day RSI of both SPY and QQQ …"; a bare list ("SPY, QQQ 10 day RSI …") is refused with the question all or any. Every ticker a condition names must end up in its rule, or the condition is refused; a negative lookback ("SPY -10 day RSI") is refused; "neither SPY nor QQQ …" / "none of SPY, QQQ and SMH …" (no ticker may meet it); brackets group, also around conditions naming tickers ("if (SPY 10 day RSI is above 70 and QQQ 10 day RSI is above 70) or TLT …"). 'and' binds tighter than 'or': "A and B or C" is (A and B) or C, with a note saying so; "RSI is below 30 or above 70"; "is exactly 50" / "equals 50" (==, with a note: exact equality rarely holds) |
| Calendar | on Mondays, in October, last / first / third / second-to-last trading day of the month, first 3 trading days of the month |
| Ranking | "buy the 5 Nasdaq 100 stocks with the lowest RSI(2) each day, hold 3 days" (up to 5 positions; free slots go to the lowest RSI(2)); with no rule or exit, "buy the 3 Nasdaq 100 stocks with the highest 20 day rate of change each week" is a weekly rotation (an equal-weight allocation re-chosen each week) |
| Entries | at the close / at the open (same day, rules must be knowable at the open) / next open; limit or stop orders ("a limit 2% below the close"); pyramiding ("pyramid up to 3 entries": with no size given, each position's share is split across the entries); a gap rule with no timing ("buy QQQ when it gaps down 2%") fills at that day's open, when it becomes known; "on QQQ, go long when …" |
| Exits | hold N days, sell when …, "sell after 10 days or when RSI is above 70", "sell when it's over 70", "sell when it is falling", "sell when they turn down" (*it* / *they* = what the entry is about: its indicator, e.g. the 50-day MA of "buy when the 50 day moving average is rising", or the price; refused if the entry is about several things; shown as a **Warning**), a bare "RSI" takes the entry's period, "sell when it crosses back below", "sell when it crosses below it" (after "buy when close crosses above the 50 day MA": the close crossing back below that average), "sell when it crosses above the 20 day mean" (the 20-day SMA), "sell at the open when …" (same open if the rule is known at the open, e.g. a gap; otherwise checked at the close and sold at the next open), "cover at the next open when …", % stop, ATR stop, trailing / chandelier stop (both at once: whichever is closer to the price), "move the stop to breakeven after +2%", take profit, sell half at +X% ("and trail the rest with an 8% trailing stop": the trailing stop starts after the scale-out), "stop at the low of the entry bar", "stop at the 5 day low", "stop at `expr`", "target 2R" / "take profit at 2 times the risk", "target at the 20 day high", "cover when it closes above it". "buy TSLA while …" / "hold TSLA when …" with no exit: in the market while the condition holds |
| Sizing | max N positions (with a short list of k < 10 tickers and no limit: k slots at 1/k each; otherwise 10 at 10%), X% per position, risk X% per trade (to the stop; with only a trailing stop, to its starting distance), target X% volatility, $X or N shares per trade (filled in full or skipped, see below), 2x leverage |
| Pine in a sentence | `request.security(syminfo.tickerid, "W", ta.sma(close, 10))`, `ta.stoch(ta.rsi(close, 14), ta.rsi(close, 14), ta.rsi(close, 14), 14)`, `ta.rsi(close, 2)[1]`, with or without backticks |
| Costs | bps or % slippage ("slippage 0.05%"), "commission 0.1%" / "0.1% commission", "stop loss at 2 ATR below entry", volume-based slippage / market impact, $ per trade, $ per share, % commission, IBKR commissions (fixed or tiered), borrow fee, margin rate, short rebate X% below T-bills, 30% maintenance margin / no margin calls, cap at X% of volume |
| Bounds | "more than" / "greater than" / "over" / "above" / "exceeds" are strict (>), "at least" / "or more" include the number (>=); "less than" / "below" / "under" are strict (<), "at most" / "or less" / "no more than" include it (<=). A bare number includes it ("fell 5%" = at least 5%) |
| Equal weights | "in equal thirds / parts", "equally (weighted)", "equal thirds of …"; weights that are all equal and add up to 99%-99.99% ("33% / 33% / 33%", 33.3% each) are equal thirds (n-ths), with a note, not 1% cash; "A, B and C weighted 50/30/20" |
| Negation and more | "it is not the case that …", "not (… or …)", "max drawdown is worse (deeper) than 10%" = a fall of more than 10% (better / shallower = less), "the 20 day SMA of SPY crosses below its 50 day SMA" (true on the day it crosses), "the top 2 by 10 day RSI of A, B, C", "the highest / lowest 10 day RSI of A, B, C" (top / bottom 1), rebalance every N days / weeks / months |
| Portfolios | %-weights, 60/40, equal / inverse-volatility / market-cap weight, if/else-if/otherwise, "when/whenever … hold X, otherwise Y", "unless … hold X, otherwise Y", "hold X unless …, in which case Y", top/bottom N by momentum/RSI/volatility, "…, weighted 70/30" (a filter's picks by rank: the best 70%, the next 30%), "the 2 of … with the highest …", "the best/worst performing of … over 10 days", "only if their 60 day return beats BIL's", "only if their 20 day RSI is above 50" (any indicator), rebalance daily…yearly or on drift, contributions, withdrawals, inflation indexing |
| Directions in holdings | "buy/hold/go long X" = long; "short X", "go short X", "sell short X", "X short" = a short position (-100% of the slice plus the proceeds in cash); "sell X", "exit X", "cover X", "sell everything", "exit" in an if/otherwise branch = cash. "sell X" inside a list ("hold TQQQ and sell TMF") is refused as ambiguous. Two identical branches get a note; a comparison of a value with itself is refused |
| Ranking by drawdown | "by max drawdown" and "by drawdown" both rank by the size of the drawdown (a positive number, as Composer): "top 1" selects the most drawn down, "bottom 1" or "smallest drawdown" the least; the notes say which |
| Higher timeframes | the weekly RSI is above 50, weekly RSI(14), the monthly 10 SMA, weekly 20 EMA (computed on completed weeks/months only) |
| More signals | ROC(10) above 5 / rate of change, %K crosses above %D, MACD histogram turns negative, yesterday's high, not on Fridays, except in October, buy stop 1% above the close / at yesterday's high |
| Portfolio conditions | any indicator phrase compared with a number or another ticker's indicator: "TQQQ 6 day cumulative return is less than -12%", "the 10 day max drawdown of TQQQ is above 20%", "SPY 10 day standard deviation of return is above 2%", "QQQ's 3 month return beats TLT's" (total returns); "the RSI of SPY is 10 points above the RSI of QQQ" (a difference: `rsi(close, 14) - rsi(sym("QQQ").close, 14) >= 10`; points of a return are percentage points) |
| Schedules and flows | semi-annually, relative bands ("drifts 25% relative to its target"), schedule + band ("rebalance quarterly or when any weight drifts more than 5%": every quarter AND whenever a weight leaves its band in between), fortnightly (every 2nd week-end); a band with no schedule, or "never rebalance" with if/else or filters, re-evaluates the rules every close and trades only when the target allocation changes or a holding leaves its band (Composer's threshold rebalancing), contributions/withdrawals for N years / starting in YEAR / from year N, growing X% a year |
| Other | starting with $X, since/from/until YEAR, month names and months ("from March 2005 to June 2015", "since Jan 1999", "until 2020-06": the month's first / last day), weight-first lists without commas ("60% VTI 40% BND since 2010", "VTI 60 BND 40"), "rebalance when drift exceeds 5%" / "at 5% drift" / "threshold rebalance 5%" / "5% threshold rebalancing" / "a 5% rebalance corridor" / "rebalance band 5%" (a band that can never trigger, e.g. 150% on 50/50, earns a warning), "a 10% target volatility using 60 day volatility" (rescaled monthly unless you say otherwise), vs TICKER (incl. SPYSIM), a blended benchmark ("vs 60/40 SPY/AGG", "compared with 60% SPY and 40% AGG", "benchmark 60/40 SPY/AGG"), versus T-bills, cash earns nothing / no interest on cash / with interest on cash, no dividends / with dividends / price-only returns (signal strategies), using today's members only; TradingView syntax without backticks ("when close > ta.sma(close, 200)"); a cross needs a direction ("crosses 70" is refused with the two readings) |
| Relative hurdles | "only if their 12 month return is above BIL's 12 month return" (each candidate vs BIL; also beats / exceeds / greater than / higher than, "above BIL" = the same indicator), "hold SPY if its 12 month return beats BIL's, otherwise IEF". A condition that compares a value with itself is refused |

Anything else can be written in the **rule language** inside backticks
(`` `zscore(close, 20) < -2` ``). See `python -m backtester --help-expr` for about 60 functions and
variables. TradingView (Pine) spellings are translated: `close[1]` (= `ref(close, 1)`; fixed offsets of 0 or
more only), `ta.sma`, `ta.ema`, `ta.rsi`, `ta.atr`, `ta.highest`, `ta.lowest`, `ta.stdev`, `ta.crossover`,
`ta.crossunder`, `ta.change` (`diff`), `ta.barssince` (`bars_since`), `ta.valuewhen` (`valuewhen`),
`ta.macd(close, 12, 26, 9)` (the MACD line; `macd_signal` / `macd_hist` for the others), `ta.cci`, `ta.mfi`,
`ta.wpr`, `ta.vwma`, `ta.hma`, `ta.alma`, `ta.linreg`, `ta.cross`, `ta.obv`, `ta.tr`, `ta.sar(start, inc, max)`,
`ta.pivothigh` / `ta.pivotlow`, `ta.stoch(close, high, low, n)` (the raw %K, `stoch_k(n, 1)`), `ta.vwap` (on daily
bars the session VWAP is the bar's own typical price, `hlc3`; use `vwap(n)` or `avwap("2020-03-23")` for longer
ones), `na()` / `nz()`, `hl2` / `hlc3` / `ohlc4`, `math.abs`, `and` / `or` / `not`. `request.security(syminfo.tickerid,
"W", x)` becomes `weekly(x)` ("M": `monthly(x)`, "D": `x`), and `request.security("SPY", "D", ta.sma(close, 200))`
becomes `sma(sym("SPY").close, 200)` (indicators must take the other ticker's series explicitly);
`lookahead = barmerge.lookahead_on` is refused. Functions that return several values (`ta.bb`, `ta.kc`, `ta.dmi`,
`ta.supertrend`) are refused with the equivalent: `bb_upper(20, 2)`, `bb_lower(20, 2)`, `sma(close, 20)`;
`supertrend(10, 3)` (the line) and `supertrend_dir(10, 3)` (-1 up, +1 down, as TradingView returns it; the first bar
starts down, as in Pine).

More indicators, all causal and matching TradingView's formulas (tested against reference implementations): `hma`,
`vwma`, `alma(x, n, offset, sigma)`, `kama(x, n, fast, slow)`, `linreg(x, n, offset)` (the least-squares moving
average), Ichimoku `tenkan(9)`, `kijun(26)`, `senkou_a(9, 26, 26)`, `senkou_b(52, 26)` (the cloud as TradingView
draws it on the current bar, i.e. computed displacement - 1 = 25 bars earlier; nothing from the future), `aroon_up`,
`aroon_down`, `aroon_osc`, `cmf(20)`, Heikin Ashi bars `ha_open` / `ha_high` / `ha_low` / `ha_close`,
`pivothigh(x, left, right)` / `pivotlow` (the pivot's value on the bar it is confirmed, `right` bars after the pivot;
NaN otherwise, so `valuewhen(pivothigh(5, 5) > 0, pivothigh(5, 5), 0)` is the latest confirmed pivot high) and
`avwap("2020-03-23")` (VWAP anchored at a date). The report charts them (averages, Ichimoku, Heikin Ashi and pivots
on the price; Aroon, CMF and the Supertrend direction in panes). For simple state, `bars_since(cond)` counts the bars since `cond` was last true and
`valuewhen(cond, x, k)` is `x` on the k-th most recent bar where `cond` was true (both causal).

Notes that reinterpret one of your words (*it* resolved to the entry's indicator, a bare "RSI" given the
entry's period) start with **Warning:** and are listed first on the command line, as warning banners in the
report and flagged on the site. Check them. If any part of a sentence isn't understood, the tool **refuses and names the words**. It
never quietly drops them or swaps in a different ticker.

## The report

- Headline tiles, an equity curve against SPY, QQQ and the stock's own buy-and-hold, and drawdowns.
- A head-to-head table: every strategy and benchmark over the same period, with CAGR, real CAGR,
  volatility, Sharpe/Sortino (daily and monthly), Calmar, max drawdown (daily and from month-end values, as Portfolio Visualizer), time underwater, Ulcer index,
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
  benchmarks that start later marked "from", or the common period (everything under that heading covers
  exactly those dates; benchmarks that start later are in a separate table with their own dates). The equity
  chart has an after-inflation view.
- When indicators need a warm-up (a 200-day average on the first bars of the data), the statistics start
  on the first day every rule has a value, and the notes say so.
- Allocation over time and current holdings for portfolios. A cash-flow summary with money-weighted
  IRR. Trailing returns (3 months, YTD, 1, 3, 5, 10 years and the full period) for the portfolio and each
  benchmark, and per-asset statistics of the holdings. Trade distribution and excursion charts are only
  shown for signal strategies.
- The price chart also draws each trade's stop, trailing stop and target, the other tickers a rule filters on
  (e.g. SPY and its 200-day average) in their own pane, and a strip of the bars on which the entry and exit
  rules were true; it has a log scale and a date label on the crosshair.
- The chart draws each series as the rule reads it: `ref(highest(high, 20), 1)` (Pine's `ta.highest(high, 20)[1]`,
  "closes above its 20 day high") is drawn one bar back and labelled `highest(high, 20)[1]`, not today's value.
  Drawings (horizontal and trend lines) are kept per ticker in the browser, so they carry across runs and reports.
- Every series a rule uses is charted: moving averages, bands, channels, Supertrend, SAR and the VWAP proxy
  over the price; RSI, stochastic, MACD, ADX/DMI, CCI, MFI, Williams %R, returns, volatility, ATR, OBV,
  volume, IBS, streaks and other tickers' series in their own panes (as many as needed: click a pane's
  title to fold it, drag its bottom edge to resize it). Weekly and monthly indicators are drawn as steps
  holding each completed period's value, as the engine reads them. The legend shows every value on the bar
  under the crosshair. Candles or a line, horizontal and trend lines (kept in your browser), and bar
  replay (step or play through time, revealing bars and trades one at a time).
- Returns by year (partial years flagged) and a monthly heatmap.
- Rolling 12-month and 3-year return, Sharpe, beta and volatility, plus rolling-period best/worst.
- The deepest drawdowns, and how the strategy did in 12 historical crises.
- Fama-French 5-factor + momentum regression, and a correlation matrix.
- For portfolios: P&L by holding (sales − purchases − costs + dividends + value still held; with
  interest and fees it adds up to the gain after cash flows, to the cent), benchmarks that receive
  the same contributions and withdrawals, the account value in today's dollars, and with
  withdrawals the safe and perpetual withdrawal rates over the tested history (as a share of the starting
  balance; for a save-then-withdraw plan, of the balance on the first withdrawal, over the withdrawal years,
  and labelled so). They come from the portfolio re-run without the cash flows over the whole period (or
  the whole withdrawal phase), so they do not depend on the amount entered, even one that empties the account.
- With cash flows every benchmark gets the same flows: one that starts later starts with the portfolio's
  balance on its first day; one whose data starts after the portfolio ran out of money is left out, with a
  note (never shown without the flows).
- Holdings that move in **monthly steps** (EFASIM/EFVSIM/VGKSIM before 1990, EEMSIM before 2003, VNQSIM
  before 2004, DBCSIM before 2006, BNDXSIM's model, LQDSIM's monthly-yield years; detected from the data by
  `data.stepped_ranges`): when one is held with a material weight (5% on average over its stepped stretch),
  volatility, Sharpe, Sortino, skew, kurtosis, beta/alpha and the factor regression are computed from
  monthly returns for the whole run, daily figures (best/worst day, positive days, daily VaR/CVaR) are
  blank, and a warning says so. The Correlations page switches daily returns to monthly (with a note) when a
  series is stepped in the window, and the Factors page regresses monthly.
- For portfolios, risk contribution by holding: each holding's share of the portfolio's volatility
  (average weight x marginal contribution, from the covariance of daily and of monthly total returns over
  the run; plus the realised share with the actual drifting weights) and of the loss in its maximum
  drawdown. Also in the Excel export (sheet "Risk contributions").
- Benchmarks are bought when the strategy is: at the close of the session before a portfolio's first day
  (day 0, see "Equity curve"), otherwise at the close of the strategy's first bar.
- For portfolios, an income table (dividends and other distributions, cash interest, the total and its yield on
  the balance at the start of each year) and each holding's calendar-year total return next to the
  portfolio's; also in the Excel export (sheets "Income" and "Asset returns by year"). A partial first or last
  year is labelled with its dates ("2010 (from Mar 3)", "2026 (to Sep 25)").
- A Monte Carlo block bootstrap (with "chance the money lasts" when there are withdrawals) and a
  transaction-cost sensitivity table.
- Exports: HTML, Excel, CSV (trades, orders, equity, holdings, yearly, monthly), JSON, and PDF
  (browser or `--pdf`).

## Command line

```bash
python -m backtester "sentence" [--dry-run] [--capital 50000] [--start 2010-01-01] [--slippage-bps 5] [--benchmark QQQ] [--tv-compat] [--pdf]
python -m backtester compare "60/40 SPY/TLT rebalanced quarterly" "buy and hold QQQ" "if SPY is above its 200-day moving average hold QQQ, otherwise hold TLT"
python -m backtester sweep "buy QQQ when RSI({2..5}) is below {5..25 step 5}, hold {1,3,5} days" --objective sharpe
python -m backtester walkforward "buy QQQ when RSI(2) is below {5..25 step 5}, hold {1,3,5} days" --in-sample 5 --out-sample 1
python -m backtester optimize SPY QQQ TLT GLD --max-weight 0.6 --test-start 2018-01-01
python -m backtester optimize SPY QQQ TLT GLD --constraint "SPY+QQQ <= 70%" --constraint "GLD <= 20%" --target-vol 0.10 --rolling 12 --lookback 60
python -m backtester montecarlo --weights "SPY 60 TLT 40" --balance 1000000 --years 30 --withdrawal 40000 [--model historical|normal|t|forecast]
python -m backtester montecarlo "hold 60% SPY and 40% AGG, withdraw 4% per year adjusted for inflation, starting with $1,000,000" --model t
python -m backtester montecarlo --weights "SPY 60 TLT 40" --withdrawal 40000 --age 65 --until-age 95 --stress worst_sequence
python -m backtester montecarlo --weights "VTISIM 90 BNDSIM 10" --glide-to "VTISIM 40 BNDSIM 60" --years 30 --withdrawal 40000
python -m backtester factors QQQ --model ff5 --freq monthly          # or --weights "SPY 60 TLT 40", a sentence, --run ID
python -m backtester factors AGG --model ff3+bonds                   # models: capm ff3 carhart ff5 ff6 bonds ff3+bonds ...
python -m backtester factors VGK --model europe_ff5                  # regional: <region>_ff3|ff5|carhart|ff6
python -m backtester factors EFA --model auto                        # the ticker's region (EFA -> developed ex US)
python -m backtester factors QQQ --model ff5+qmj+bab                 # add-ons: mom, qmj, bab, bonds, term, def
python -m backtester style QQQ [--assets "SPY EFA EEM IEF BIL"] [--window 36]
python -m backtester montecarlo --weights "SPY 60 IEF 40" --withdrawal 50000 --horizon mortality --age 65 --sex joint --age2 63
python -m backtester correlation SPY TLT GLD EFASIM --window 36 --freq monthly [--pair SPY,TLT] [--start 2000-01-01]
python -m backtester optimize SPY TLT GLD --methods omega,max_return_over_maxdd --omega-threshold 0.03
python -m backtester optimize SPY TLT GLD --expected-return "SPY=7%,TLT=4%,GLD=3%" --expected-vol "SPY=16%" --correlation "SPY/TLT=-0.2"
python -m backtester optimize SPY QQQ TLT GLD --view "SPY = 8% @ 60%" --view "QQQ > TLT by 3% @ 40%" --prior "SPY=40%,QQQ=20%,TLT=30%,GLD=10%"
python -m backtester optimize SPY QQQ TLT GLD --benchmark "60% SPY 40% TLT" --target-active 0.01 --resample 200 [--json]
python -m backtester signals "buy Nasdaq 100 stocks when RSI(2) is below 5, hold 3 days" [--webhook URL]
python -m backtester paper add "hold the top 5 Nasdaq 100 stocks by 6 month momentum" --name mom5 ; python -m backtester paper report
python -m backtester trade "hold 60% SPY and 40% TLT, rebalance monthly" --broker alpaca --dry-run   # see "Broker trading"
python -m backtester --tickers MSFT --entry "down_days >= 5" --hold 1        # explicit rules
python -m backtester --spec reports/<run>/strategy.json                      # re-run exactly
# report folders: reports/<the start of the description>-<8-character hash of the full spec>, e.g.
# reports/if-spy-is-above-its-200-day-moving-average-then-ho-1a2b3c4d: two runs that start alike get their own
# folders, and re-running the same spec reuses its folder (the command prints the path)
python -m backtester tickers                                                 # what data exists
python -m backtester composer-export "if SPY is above its 200 day moving average then QQQ else BIL" --out s.json
```

## How the simulation works

- **Indicator prices.** Signal strategies read indicators on quoted (split-adjusted) prices, as
  TradingView does. Allocation portfolios read them on dividend-adjusted, total-return prices, as
  Composer and Portfolio Visualizer do (`price_basis: "adjusted"`; say "using quoted prices" or set
  `"quoted"` to switch). The adjusted series is built forward from the first bar, so it never changes
  when newer data arrives. Trading always uses the quoted price plus cash dividends. A price level
  compared with a fixed number ("SPY price is above 400"; an SMA, standard deviation or Bollinger band
  of price against a number, also from Composer imports and the Build page) is read on quoted prices:
  the rule becomes `quoted(close) > 400` and a note says so. So are price levels of **different tickers**
  compared with each other ("GLD price is above HYG price", "the 50 day SMA of GLD is above the 50 day SMA of
  HYG") and rankings **by** a price level ("the top 1 of GLD and HYG by current price"): each ticker's
  total-return level starts at its own first close and grows with its own dividends, so levels of two
  tickers are only comparable as quoted (`quoted(close) > quoted(sym("HYG").close)`). Relative
  comparisons within one ticker (price vs its own average), returns, ratios and RSI stay on the adjusted
  basis.
- **Warm-up.** A portfolio holds cash until every rule, ranking and weighting has its full lookback,
  and the statistics start that day with the starting capital. Nasdaq-100 universes start at the first
  point-in-time membership snapshot (March 2004); nothing counts as a member before it.
- **Target volatility and blended benchmarks.** "target 10% volatility (using 60 day volatility)"
  scales a portfolio's exposure towards that volatility (capped at its leverage). A benchmark can be a
  blend: "vs 60/40 SPY/AGG", `--benchmark "60 SPY 40 AGG"`.

- **Prices and dividends.** Prices are daily and split-adjusted, as quoted. Dividends are paid in
  cash on the ex-date, and short positions pay them ("no dividends" / `dividends: false` turns this off: price-only). Idle cash earns the 3-month T-bill rate;
  borrowed cash pays it plus any margin rate.
- **Shorts and margin (signal strategies).**
  - Short sale proceeds earn the T-bill rate minus `short_rebate_spread` (default 0.25%/yr, floored
    at zero), like a broker's short rebate. "full short rebate" sets it to 0, "no short rebate" to
    nothing earned.
  - With leverage above 1x or any short, `maintenance_margin` (default 25%) is checked at every
    close: if equity / gross exposure is below it, every position is cut pro rata at that close back
    to the initial margin (1/leverage). The trades are marked "margin call" and the notes list the
    dates. "no margin calls" turns the check off.
  - Leverage is what a broker would lend: under Regulation T (`margin_account: "reg_t"`, the default) at
    most 2x overnight on stocks. Up to 4x needs a portfolio-margin account ("with portfolio margin",
    `margin_account: "portfolio"`). The maintenance margin must be below the initial margin (1/leverage):
    4x with the default 25% maintenance is refused (every close below the entry would be a margin call);
    say e.g. "4x leverage, with portfolio margin and a 15% maintenance margin".
- **Broker costs (signal strategies and portfolios).** Both engines charge them through one module,
  `backtester/costs.py` ("IBKR commissions" and "volume-based slippage" work in portfolio sentences too).
  - `commission_model: "ibkr_fixed"` ("IBKR commissions"): $0.005/share, min $1, max 1% of the
    trade value per order.
  - Per-share fees, minimums and caps, whole-share sizing, `fixed_shares` sizing and the share counts and
    prices in the trade list use shares **as traded** that day, not split-adjusted ones: AAPL in 1995
    traded around $40, not the split-adjusted $0.38, so $10,000 bought ~240 shares, not ~26,000.
    (As-traded shares = split-adjusted shares / the product of later splits. That uses later splits only
    as a unit conversion - the as-traded price is what was quoted that day - and changes no return or
    signal.) A split while a trade is open changes its share count: `exit_shares` is the count at exit.
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
  that day's signal is valid. With `pyramiding` N and no size given, each entry gets 1/N of the position's
  default share (a note says so), so every add-on has room.
- **Today's signals and orders.** `signals` (and the daily Action, its webhook payload and `signals/latest.md`)
  lists, besides new entries: exits made at today's close (market-on-close), exits for the next open (an exit
  rule filled at the next open, a holding period ending at an open), holding periods ending on a later date,
  and for each open position the stop (stop order) and target / scale-out (limit orders) levels for the next
  session, marked one-cancels-other. They are what the backtest does: truncated at any day D, the scan's exits
  equal the engine's trades on D and D+1 (tested). The orders page turns next-open exits into SELL orders.
- **Sizing is what the summary says.** "200% per position" raises leverage to 2x (with a note) instead of
  being capped at 100%; a long stop of 100% or more, and a starting capital of zero or less, are refused.
  A fixed size ("100 shares per trade", "$5,000 per trade") is filled in full or not at all, as TradingView does:
  an order the account can't pay for (cash plus the commission, within the leverage allowed, and any volume cap)
  is skipped, never cut to what the cash allows, and a **Warning** counts the skipped orders and lists the first
  dates. Share counts are whole shares as stated.
- **Costs are never negative.** Negative slippage, commissions, borrow fees, margin rates or expense ratios are
  refused (in sentences, on the command line, in the site's options and in JSON specs).
  Volume caps on fills at the open use the previous day's volume.
- **Bar ordering.** Each bar runs overnight financing and dividends → exits at the open → entries at
  the open (market, then limit/stop) → intraday stops and targets → exits at the close → entries at
  the close.
  - A stop and a target touched on the same bar count as the stop (conservative). In TradingView-compatible
    mode the bar's path decides instead, as TradingView's broker emulator assumes: open → high → low → close
    when the open is nearer the high, otherwise open → low → high → close.
  - Gaps through a stop fill at the open.
  - The stop and the target are one-cancels-other (OCA): whichever fills closes the position and cancels the
    other.
  - Trailing, chandelier and breakeven stops move with the best price reached, using each bar's high (low for
    shorts) after that bar's own stop check: a level set by today's high applies from the next bar, because a
    daily bar does not say whether its high came before its low. A breakeven stop ("move the stop to
    breakeven after +2%", `breakeven_after: 0.02`) is a stop at the entry price, armed once the best price is
    that far in favour.
- **TradingView-compatible mode** (`--tv-compat`, `"tv_compat": true` in a spec, the "TradingView-compatible"
  setting on the site). Entries and rule exits whose timing the sentence doesn't state fill at the next bar's
  open (TradingView's default, `process_orders_on_close = false`) instead of the signal bar's close, and a stop and
  a target touched on the same bar follow TradingView's open-high-low-close path (above). Stated timing ("at the
  close") is kept. Returns are price-only, as TradingView's strategy tester reports them: idle cash earns nothing and
  dividends are not credited (TradingView's default charts are split-adjusted, not dividend-adjusted). Each is stated in
  a note and can be switched back: "with interest on cash", "with dividends" (or the site's settings, or
  `cash_rate` / `dividends` in a spec).
- **Scale-outs are resting limit orders.** "Sell half at +3%, take profit at 6%, stop loss 5%": on a bar that reaches
  several levels, the levels fill in the order the price reaches them - the nearest profit level first (a scale-out
  below the target fills before the target closes the rest), and a level the open already passed fills at the open
  (a gap). When the stop and a profit level are both touched the stop is assumed first (all of what is left exits at
  the stop); in TradingView-compatible mode the bar's path decides for every level, scale-outs included (open → high
  → low → close: the scale-out and target fill on the way up, then the stop on the way down).
- **Stops and targets at a price** (`stop_level`, `target_level`, `target_r`). A rule-language expression evaluated
  once, when the position opens, and then fixed: "stop at the low of the entry bar" (`low`), "stop at the 5 day low"
  (`lowest(low, 5)`), `entry_price - 2 * atr(14)`; a target may use `stop_price`, and "target 2R" is entry + 2 x
  (entry - initial stop). Only data known at the fill is used: a fill at the close reads that bar; a fill at the open
  (same day, next day, or a limit/stop order during the bar) reads the previous bar, unless the expression is
  open-safe - so with a next-open entry "the low of the entry bar" is the signal bar's low (a note says so). An entry
  whose level is not defined yet (indicator warm-up) is skipped.
- **Reconciling with TradingView's Strategy Tester.** In TradingView-compatible mode the backtest follows the broker
  emulator's documented conventions:
  - Orders placed at a bar's close fill at the next bar's open (`process_orders_on_close = false`); "at the close"
    is `process_orders_on_close = true`.
  - Sizing as a percentage of equity (`strategy.percent_of_equity`) or a cash amount (`strategy.cash`): the quantity
    is computed on the bar the order is placed, from that bar's equity and **close**, and rounded down to **whole
    shares** for stocks and ETFs (TradingView's minimum quantity of 1 share; the Help Center's Strategy properties:
    order sizes are "subject to constraints due to the minimum tradable quantities for the symbol"; crypto and FX
    pairs keep fractions; say "fractional shares" to allow them). An order the cash can't pay for at the fill (a gap
    up after a 100%-of-equity signal) is skipped, as TradingView does, and a note counts the skipped orders.
  - "Hold N days" is `strategy.close()` once N bars have passed since the entry bar: placed at that close, filled at
    the next open (N + 1 bars after the entry bar). "Hold N days and sell at the close" keeps the close.
  - A stop and a target (and scale-outs) touched on one bar follow the open → high → low → close path (open nearer
    the high) or open → low → high → close; price levels crossed by a gap fill at the open.
  - Cash earns nothing and dividends are not credited (price-only returns, on split-adjusted prices).

  Remaining differences to check when numbers don't match: TradingView's defaults are `initial_capital = 1,000,000`,
  an order size of 1 share (`strategy.fixed`) and **no commission** - this backtester defaults to $10,000 and 100% of
  equity, so state the capital and size; `pyramiding` defaults to 1 in both; slippage in TradingView is in ticks
  (here basis points); margin calls follow the 25% maintenance margin here, TradingView's margin model differs; the
  price history (a different data vendor, split adjustments) and the first bar used (TradingView starts at the
  chart's first bar, here at the start date, with indicator warm-up taken from earlier data) can differ;
  `use_bar_magnifier` / intrabar data is not available (daily bars only); a stop or target recomputed on every bar
  (a `strategy.exit` with a moving level) is fixed at entry here.
- **Pine scripts.** A pasted TradingView strategy (text starting with `//@version`, or containing `strategy(...)`
  and `strategy.entry`) is translated instead of read as English - on the command line (`python -m backtester
  "$(cat script.pine)" --tickers SPY` or `python -m backtester script.pine --tickers SPY`) and on the site (the
  "Ticker (Pine script)" setting), or with a `// ticker: SPY` comment in the script. Supported:
  `strategy()` settings (initial_capital, default_qty_type / value, commission_type / value, pyramiding,
  process_orders_on_close, margin_long), `input.*()` (their defaults), variables of `ta.*` / `math.*` expressions,
  tuples (`[m, s, h] = ta.macd(...)`, ta.bb, ta.supertrend, ta.dmi, ta.kc), `strategy.entry` with `when=` or inside
  `if` / `else if` / `else`, `strategy.close` / `strategy.close_all`, `strategy.exit` with `stop=` / `limit=` from
  `strategy.position_avg_price` (a percentage, or `- 2 * ta.atr(14)`), `profit=` / `loss=` / `trail_offset=` as a
  distance `/ syminfo.mintick`, `qty_percent` (scale-outs), `request.security` on the chart's symbol or another
  ("W" / "M" / "D"), date filters (`time >= timestamp(2015, 1, 1)` sets the start), bars since entry
  (`bar_index - strategy.opentrades.entry_bar_index(0)`). Plots and alerts are skipped. Anything else (var / `:=`
  state, custom functions, loops, ternaries, `strategy.order`, limit/stop entries, levels that move every bar) is
  refused with its line number. Each translated line is listed in the notes, and the result runs in
  TradingView-compatible mode.
- **Dates are checked first.** A start or end that is not a date ("garbage", 2016-13-45) is refused with a clear
  message in sentences, on the command line, on the site and in JSON specs.
- **Entry fill labels.** In trades.csv `entry_fill` is `close`, `open`, `limit` or `stop` (a limit/stop order filled at
  its level during the bar); a limit/stop order the open had already crossed is `open`.
- **Partly known at the open.** For an entry at the open whose rule mixes open-time and close-time parts
  (`` `ta.change(open) > 0 and ibs < 0.5` ``), only the close-time parts are checked on the previous bar, and a
  **Warning** names exactly which parts were lagged and which use today's values.
- **No dust trades.** Orders worth less than `min_order` (default $1) are skipped. When a pyramiding add-on
  finds no room left (the position already uses the capital or leverage available) it is skipped and a
  warning says how often.
- **Slot contention.** When more tickers signal than there are free position slots and no ranking is given,
  the most liquid are taken first: the highest 20-day average dollar volume (close × volume), as known when
  the order is placed. A note says on how many days this happened; "prefer the lowest RSI" (or `rank_by`)
  chooses differently.
- **No lookahead.**
  - "At the close" rules use that day's data and fill at the close (market-on-close).
  - "At the open" rules may only use data known at the open. Anything else is automatically checked
    on the previous close, and a spec that breaks this is rejected.
    - Calendar variables and the period-end flags (`is_week_end()`, `is_month_end()`, `is_quarter_end()`,
      `is_year_end()`, `trading_days_left_in_month`) come from the published NYSE schedule
      (`backtester/calendar.py`), so they are known at the open. The schedule is point in time: holidays on the
      rules of their year (Feb 22 / May 30 holidays, Election Day and the other pre-1971 closures) and the
      pre-announced closures (the Reagan, Ford, G. H. W. Bush and Carter days of mourning) from their
      announcement day; unscheduled closures (9/11, Hurricane Sandy) are never known in advance.
      `trading_days_left_in_month` counts the sessions AFTER today: 0 on the month's last session
      ("the last trading day of the month" is `trading_days_left_in_month == 0`).
    - The check is a whitelist: the open, `gap`, calendar variables, `sym("X").open`, anything inside
      `ref(..., n)` with n ≥ 1, and one-series indicators given an open-safe series (`sma(open, 5)`;
      `sma(20)` means `sma(close, 20)` and is refused). Keyword arguments are refused at the open.
    - Lookbacks, lengths and offsets must be numbers written in the rule: `sma(close, abs(20))` or
      `ref(close, 2-1)` is an error everywhere, so the check and the calculation can't read a rule
      differently.
  - Rules written as Python functions (the Python API: `entry=lambda df, ns: ...`, also exits, rankings and
    order levels) are causal by construction: each is called on the data up to bar i (a copy; `ns['sym']` and
    `data.load` are cut at the same bar) for every bar in order, and only the last value of each call is used,
    so a function that caches the largest frame it has seen, or computes `shift(-1)`, never sees a later row.
    Reading files or the network while it runs raises an error. A 5,000-bar series takes a few seconds; a
    note gives the timing. A function whose whole-history answer differs from its bar-by-bar one is refused
    (below). Mark one `f.vectorized_causal = True` to call it once on the whole history instead (faster, with a
    warning note): only then does the empirical probe guard it: the function is run on the data cut at
    many dates chosen adversarially - the latest 40 bars one by one, every day an entry fires and the three
    days before it, every day its answer changes, every day of a short run window, then a dense random grid
    (up to 1,200 cuts, a few seconds) - and its output up to each cut must equal its output on the full data.
    (A leak confined to a few days of a long history can still escape the random part.) `df.close.shift(-1)`, `rolling(..., center=True)` or
    `df.close.mean()` change when later rows are removed, and the run is refused ("Lookahead: the entry
    function uses future data: its result on D changes when the data after D is removed"). A function used
    at the open must be marked `f.open_safe = True` and is also run with that day's close/high/low/volume
    perturbed. Portfolio `custom` functions only ever receive the history up to each date.
    - Behind it, every run at the open replays the rule on dates across the whole history with that
      day's close/high/low/volume replaced by other valid values (tiny to large), for every ticker the
      rule reads; any change in the decision rejects the spec.
  - Negative offsets are rejected.
  - Tests truncate all data at a date and check that no earlier trade changes.
- **Survivorship.** "Nasdaq 100 stocks" means point-in-time membership from 2004 (monthly snapshots
  of the index list, with exact change days from the dated component-change table from 2007: SMCI counts from
  2024-07-22, PLTR from 2024-12-23, not from the next month). A stock is only bought while it was in the index,
  and former members are included where price history exists.
  - "using today's members only" trades the CURRENT member list (the latest snapshot plus the dated changes
    since) over the whole period with no membership filter: survivorship-biased by construction, and both
    engines add a warning saying so.
  - About 90 former members (mostly acquired companies) have no free price history, so some bias
    remains (member-month coverage about 48% in 2004, about 75% over 2004-2026). The report and the
    console summary put the coverage in the headline ("Survivorship: 75% of member-months have data (48%
    in 2004) - results are biased upward"). The figures cover the run's own period (its first to last trading
    day) and are computed once, so the note, the report headline and the command line quote the same numbers,
    with the biggest missing members by member-months and, as
    context, an equal-weight portfolio of the members with data against a fund holding the whole index
    (QQQE, else QQQ) over the same months. A free Tiingo key fills most of the gap (see Data).
  - Market-cap rankings and weights ("top 10 Nasdaq 100 stocks by market cap") start on the first day
    share counts cover at least 80% of the members (a note says so), and a note lists any rebalance where
    a top-N filter ranked fewer than N names or under 80% of its universe.
  - Before 2004 the earliest known list is used.
- **Delistings.** When a held ticker's data ends more than a week before the backtest does (acquired or
  delisted), the position is sold at its last close on its last day (trades/orders marked `delisted`, and a
  note). The signal engine keeps the proceeds in cash for new signals; a portfolio holds them in cash until
  its next rebalance, where the ticker counts as no longer trading (a fixed slice of it stays in cash, a
  filter or weighting picks among the rest). An index universe drops it from membership.
- **Spin-offs.** A "dividend" worth more than 15% of the price (the data books spun-off shares at their
  value, e.g. MDLZ on 2012-10-02) is paid in cash like a dividend but labelled a spin-off/special
  distribution in the notes and the portfolio ledger.
- **Splits missing or booked wrongly, bad ticks.** Every price file passes an integrity gate on load
  (`backtester/integrity.py`). A one-day level change by a common split ratio (2, 3, 1/4, 1/10, ...) that leaves a
  day in line with the best-correlated reference (SPY, TLT, SOXX, NVDA, ...), holds for the following days, and
  (where volume is reported) shifts the volume the same way is an unrecorded split: it is recorded and the earlier
  prices back-adjusted (PGOVX, PCRAX and PSLDX 2023-03-27, PIMCO reverse splits; McDonald's 1968 and 1969
  2-for-1s). A booked split that the prices already moved by is dropped (NVDS 2023-08-09). An isolated bar 25%+
  from both neighbours, which agree, is replaced by their average (CPER 2014-12-04; DFEN on the NYSE's bad-print
  day 2024-06-03). Real crashes stay (SVXY 2018-02-06, AAPL 2000-09-29): the ratio leaves no plausible day or the
  volume burst says otherwise. `data/inferred_splits.json` holds manual overrides (SOXS 2026-05-26: prices 15x
  too high before it, no split that day) and the data job's log of every repair with its evidence; a backtest
  over a repaired day says so ("Data repaired: ..."); `data.price_repairs(ticker)` lists them.
- **Corporate actions booked twice.** Yahoo sometimes records one event in two ways on its ex-date: DHR on
  2016-07-05 (Fortive spin-off) has both a $24.56 payout and a 1.319 "split", which together gave a phantom
  +39% day; EXPE 2011-12-21 (TripAdvisor) and TMUS 2013-05-01 (MetroPCS) pay per pre-split share on a
  post-split price basis (phantom -25% and -13%). Yahoo's adjusted close is built from the same two fields,
  so it is no independent check. When a day's split and payout disagree with the adjusted close by more than
  2%, `data.reconcile_actions` tries the other readings (payout per pre-split share; the "split" is the
  spin-off itself and is dropped; the split alone stands for it) and keeps the one whose one-day return is
  closest to the market's that day: DHR +2.6% (Danaher's own figures: $101.91 before, $78.94 + $24.56 of
  Fortive after), EXPE +1.8%, TMUS +4.1%. Both engines use the reconciled data and a note lists the days.
- **Equity curve.** The curve starts with the starting capital on the previous trading session (never a
  weekend or holiday); that row has no year or month of its own. A **portfolio** is bought at that session's
  close (day 0), as Portfolio Visualizer starts from the prior period-end: "hold 100% SPY from 2010" is bought
  at the 2009-12-31 close, so 2010 is SPY's whole calendar-year return (15.06%) and the first day's return
  counts. The initial allocation is decided on data up to that close; the benchmarks and equity.csv start on
  day 0 too. When a holding has no price that day (its first day of data, e.g. no start date and a fund that
  began then) or the rules cannot decide yet (warm-up, membership data), it is bought at the first day's close
  instead and a note says that day's return is not counted. A signal strategy trades from its first bar.
- **Portfolios.** Targets are re-evaluated on the schedule (month-end close by default) and traded at
  the close or next open. Only the differences are traded, and new contributions buy the target mix.
  - "Trade at the next open" needs real opening prices. A ticker with none in the period (a SIM series or a
    mutual fund: only a daily close) is refused, as in signal strategies ("SPYSIM has no real opening
    prices ... Trade at the close instead"). A day on which a ticker's open was not quoted (old data) fills
    that ticker at the day's close, with a note ("Opens: ...").
  - A period ends on the last *scheduled* NYSE session of the week/month/quarter as known that day: after an
    unscheduled closure (9/11) the rebalance happens on the first bar after it, not in hindsight on the bar
    before.
  - "rebalance every year in June" (or "annually in June", "each December"): once a year at the last trading
    day of that month (`"rebalance": "yearly_6"` in a spec), plus the initial purchase.
  - Cash flows are made on the first trading day of each period, the first day of the backtest included (as
    Portfolio Visualizer and the Monte Carlo simulation do): "add $1,000 a month for 20 years" is 240
    contributions, the first on day one; "withdraw 4% a year" takes the first withdrawal on day one.
  - "adjusted for inflation" (or "indexed to inflation", "in real terms") belongs to the flow it is written
    with: "add $1,000 a month for 20 years, then withdraw $50,000 a year adjusted for inflation" indexes only
    the withdrawals. Written apart from the flows ("..., adjusted for inflation") it indexes every $ flow, with
    a note. A real amount is in dollars of the backtest's first day unless it says otherwise: "in 2000 dollars"
    (that year's average CPI) or "in today's dollars" (dollars of the flow's first payment, so the first
    withdrawal is exactly the amount). The notes say which, and the first payment in dollars.
  - Flow indexing uses CPI *as published*: each month's figure from about two weeks after the month, so a
    withdrawal only grows with inflation that was known that day. The yearly table's inflation column is for
    reporting and uses calendar months: December to December (a partial year to the latest month published).
    CPI is CPI-U not seasonally adjusted (FRED CPIAUCNS, the official figure Portfolio Visualizer uses: 1967
    is 3.0%).
  - A withdrawal is capped at the balance: when one is more than the account holds, everything is sold at that
    close and what is left is paid out (cash_flow in equity.csv is the amount actually paid). The notes and the
    cash-flow table say "portfolio depleted on <date>", and the report stops on that day: statistics, the
    yearly table (the depletion year is its last row), the monthly table, the equity and allocation charts, the
    holdings' statistics and the benchmarks all end there. Benchmarks that receive the same flows are capped at
    their own balance the same way (their values are shown on the depletion day), and the Monte Carlo replays
    the flows as scheduled over the whole period.
  - "starting with $0" works when contributions fund the account from the first day: returns are
    time-weighted from the first funded day (and it is refused when nothing would ever be invested).
  - Short positions (negative weights): the proceeds earn the cash rate less `short_rebate_spread` (default
    0.25%/yr) and pay `borrow_fee` (annual, default 0) on their market value, charged daily.
  - Rules that rank or weigh assets need their look-back: the statistics start once *every* ranked asset has it
    (`warmup: "all"`, the default; `"first"` starts once enough assets can fill the slots), and the notes name
    the assets that were waited for.
  - trades.csv of a portfolio lists holding periods: `position_value` is the average market value while held,
    `entry_value` the first purchase, `bought`/`sold` every purchase/sale (a daily-rebalanced holding makes
    many), `pnl` sales − purchases − costs + dividends (+ value still held), and `return` the ticker's own
    total return over the period.
- **Returns and costs.** Returns are time-weighted, so cash flows don't distort CAGR or Sharpe. The
  money-weighted IRR is reported separately. Costs default to zero, and the report always shows a
  cost-sensitivity table.

## Broker trading (Alpaca)

`python -m backtester trade "<sentence>" --broker alpaca` sends today's orders for a strategy to
[Alpaca](https://alpaca.markets). It uses the **paper** account unless the environment has `ALPACA_LIVE=1`
**and** `--live` is given (both, so a typo can't send real orders).

```bash
export ALPACA_API_KEY_ID=...   ALPACA_API_SECRET_KEY=...        # paper keys from app.alpaca.markets
python -m backtester trade "hold 60% SPY and 40% TLT, rebalance monthly" --dry-run    # print, send nothing
python -m backtester trade "hold 60% SPY and 40% TLT, rebalance monthly"              # paper account
python -m backtester trade "buy QQQ when RSI(2) is below 10, sell when RSI(2) is above 70, 5% stop loss" --whole-shares
```

- **What is sent.** The target is the Orders page's reconciliation: what the strategy holds after the latest
  bar (a portfolio's tree evaluated today; a signal strategy's open positions plus new entries), sized to the
  account's equity, minus the positions the broker reports. Sells go first.
- **Order types.** Fills at the close are market-on-close (`time_in_force: "cls"`), at the (next) open
  market-on-open (`"opg"`). Limit/stop entries are the engine's own working orders (`signals` lists them as
  "BUY LIMIT SPY @ 763.64"): the level computed on the signal bar exactly as the backtest does (a limit 1% below
  the close is 0.99 x that close), sized at that price, a day order (good-til-cancelled, with a note, when it
  works for several sessions), with the stop loss and take profit attached as a bracket at the levels the
  engine applies after a fill at that price. Market entries at the next open carry their stop / target as a
  day market bracket (sent before the open, it fills at the open), so the entry session is protected; a
  market-on-close entry can't carry one at Alpaca, so a dry run shows the OCO exit pair the next run places
  once the shares are held (before the first session they are exposed in). Open positions of a signal strategy get the next session's exit
  orders: stop + target as one-cancels-other, or a single stop / limit, and scale-out limits, at the
  backtest's levels. Orders this tool placed earlier (client ids starting `bt-`, and the open legs of its
  filled brackets) are cancelled first, so
  yesterday's levels are replaced (`--keep-open-orders` keeps them).
- **Fractional shares.** Portfolios are rebalanced to exact weights with fractional shares, which Alpaca only
  takes as day market orders; `--whole-shares` keeps the close/open timing.
- **Idempotent.** Each order's `client_order_id` is built from the strategy, the date, the symbol and its role,
  so a second run on the same day is rejected by Alpaca instead of doubling the orders.
- **Timing.** The data is end-of-day: a strategy that fills "at the close of the signal day" is executed at the
  next close, a day after the backtest's fill (a note says so). Next-open strategies trade as tested.
- **Dry run.** `--dry-run` prints the order payloads and sends nothing; without keys it sizes a
  `--account-value` (default $10,000) account with no positions.
- **Secrets.** Keys are read from the environment only and never printed or logged (errors are scrubbed).
- **Scheduled trading.** `.github/workflows/trade-alpaca.yml` is a template: it runs only when started by hand
  (with a dry-run switch) until you uncomment its `schedule`, and does nothing unless the repository secrets
  `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` are set. The strategy sentence is the repository variable
  `ALPACA_STRATEGY`; the variable `ALPACA_LIVE=1` switches it to the live account.

## Data

`scripts/fetch_data.py` runs in the **Fetch price data** GitHub Action every weekday after the US
close and commits updates, so `git pull` gets fresh data. It downloads:
- prices from Yahoo Finance for current and former Nasdaq-100 members, ETFs and indexes (Tiingo,
  Alpha Vantage and Stooq as keyed fallbacks for delisted names, below)
- point-in-time membership reconstructed from the Wikipedia article's revision history, with the live
  list from stockanalysis.com, Wikipedia or Nasdaq for the current month, and the dated component changes
  (`data/ndx_changes.csv`: date, added, removed, reason, from the "Historical components of the Nasdaq-100"
  table, February 2007 on). Within 40 days of a dated change the table decides membership (a member from the
  effective day); elsewhere the monthly snapshots stand
- the T-bill rate and CPI from FRED
- Fama-French factors from Kenneth French's data library: US daily and official monthly files
  (3 factors, 5 factors, momentum), the same for developed, developed ex US, Europe, Japan, Asia Pacific
  ex Japan and North America, and emerging markets (monthly only)
- AQR's Quality Minus Junk and Betting Against Beta factors (monthly spreadsheets, every country and
  aggregate). Each file is parsed on its own (`backtester/sources.py`); a failure is logged in
  `data/factors/fetch_log.txt` and the rest of the job carries on
- share counts for market-cap weighting: Yahoo (from about late 2015; merged into the saved files, so
  the history grows). **Market caps therefore start around 2015-11 for most stocks** (AAPL: 2015-10-29); a
  `market_cap` rule is false before a stock's first count and a note names the date. ETFs, funds and indexes
  have no market cap: a signal rule on `market_cap` for one is refused. SEC EDGAR XBRL company facts would
  reach back to about 2009 (`data/shares_sec`), but the SEC currently blocks the download from GitHub
  Actions, so that folder is empty and nothing before Yahoo's counts is available. (Keyless, one
  count per 10-Q/10-K - the cover-page shares outstanding, else the balance-sheet or weighted-average count -
  dated by the filing date, so it is only used once public). SEC counts fill the dates before Yahoo's
  first count and any gap of more than 120 days in Yahoo's.

**Market cap** is the close as quoted that day times the shares outstanding last reported before that day
(each count is used from the next session). Yahoo's share counts are in the share units of their date, so
they are put on the same split basis as the prices, with counts that Yahoo still reports in pre-split units
for a few weeks after a split corrected, and a jump of more than 15% only used once a second report confirms
it. A market cap whose implied daily turnover (dollar volume / market cap) is outside 0.001%-100% is treated
as unknown. Share classes of one company (GOOG/GOOGL, FOX/FOXA, ...) each get the company's value divided by
the number of listed classes. Examples: AMZN about $1.7T in mid-2021, NVDA about $2.3T at the end of March
2024, AAPL about $3.0T at the end of 2023, AEP under $60B.

**Nothing is ever deleted.** A failed or partial download never replaces a saved history: a refresh that
comes back shorter than the file (Yahoo reset EA to a single bar when it was taken private in August 2026)
is spliced onto the saved rows when the two agree on their overlap, and otherwise the saved file is kept.
Symbols whose history has ended are listed in `data/delisted.json` (last date, membership months and the
reason when known), and a backtest that runs past a ticker's last date says so. What the merge did on each
run is in `data/merge_log.txt`.

**Data checks.** Opening prices get a sanity pass when loaded (using only each bar and the ones before it):
an open outside the bar's own low-high range is clipped into it, and for a company with two listed share
classes an open far from the other class's (scaled by that day's closes, where the two normally track each
other) is replaced from it - GOOG's open and high on 2014-04-02 are about 6% above anything that traded. A
repaired open is never used for a fill. `data.open_anomalies(ticker)` lists further suspect opens judged
with hindsight, for review. When you name a ticker whose file is probably not the company you mean in the
period - a recycled symbol such as CPWR (Compuware was an index member; the file is a later penny stock),
DELL before 2016 or MNST before 2012 - the report adds an "Identity:" note.

**Broad fund universe.** Besides the Nasdaq-100, the core ETFs and the indexes, the job keeps a curated
list of about 650 US-listed ETFs across asset classes (total market, style, size, Avantis, Dimensional,
factor, dividend, sector and industry, regional and country, Treasuries, corporates, high yield, munis, TIPS,
international and EM bonds, REITs, commodities, alternatives, currencies, crypto, and the leveraged / inverse
funds used in Composer symphonies) and about 200 popular mutual funds (Vanguard, Fidelity, DFA, PIMCO,
American Funds, T. Rowe Price, Dodge & Cox, PRPFX, ...), listed in `backtester/fund_lists.py`. Yahoo serves a
mutual fund as a daily NAV with its distributions, so its file is a total-return history (flat bars, no
volume), mostly from the 1980s or the fund's launch. They are refreshed in rotating batches of up to 450 a run
(missing files first, then the ones updated longest ago), so each run stays short and polite to Yahoo and a
fund's last bar may be a day older than the ETFs'. A symbol Yahoo doesn't know is retried after 30 days
(`broad_failed` in `data/universe.json`). None of them is ever read as a Nasdaq-100 member. A price file is
about 50 bytes a day (about 0.25 MB for a typical ETF, 0.35 MB for a mutual fund with 30-45 years of
history), so the broad list adds roughly 150 MB to `data/prices` (223 MB before it).

**Adding tickers.** The job also downloads every ticker in `data/extra_tickers.txt` (one or more per line,
`#` for comments). Add a symbol there and run the **Fetch price data** workflow (Actions → Fetch price data →
Run workflow; pushing a change to the file also starts it), then pull. On the site's Data page, **Add
ticker** downloads the symbol directly when the machine has internet access; otherwise (the cloud sandbox)
it appends the symbol to `data/extra_tickers.txt` for you and says to push the file. A sentence or tree
that names a ticker without data says so and points to this file.

**Your own series.** Import a daily or monthly return or price series (a CSV of `date,value` rows; returns in
% or as decimals) as a named ticker, usable anywhere a ticker is (portfolios, benchmarks, Monte Carlo, the
optimiser, factors): `python -m backtester import-series MYFUND returns.csv [--returns|--prices] [--monthly]
[--percent|--decimal]` (`--list`, `--delete`), or **Import your own series** on the Data page (POST
/api/series). It is stored in `data/custom/MYFUND.csv` (plus `MYFUND.json`, what was imported); commit the
files to keep them. Daily values sit on NYSE sessions (a value dated on a weekend counts from the next session;
gaps of more than 40 days are refused); monthly values step on the last NYSE session of each month (every
month must be present), so the reports treat the series as stepped (monthly statistics). Dates must be in
order without repeats, and implausible values (a daily return above 75%, a monthly one above 300%, a price at
or below 0) are refused. Names are 1-10 capital letters or digits and cannot be an existing ticker.

**Delisted former members.** `data/delisted.json` marks a symbol whose saved history is too short to use
(`"history": "history unavailable - needs TIINGO_API_KEY"`): EA was taken private in August 2026 and Yahoo
now serves a single bar, and its full history was never saved. Yahoo drops companies that were acquired or went bankrupt (Celgene,
Xilinx, Activision, Yahoo, …), which is the main survivorship gap: about 90 former members have no free
history, and member-month coverage is about 48% in 2004-05 and about 75% over 2004-2026. No keyless source
reachable from a GitHub Action carries them (checked in September 2026: Yahoo's chart API and Nasdaq's
historical API answer "symbol may be delisted" / "Symbol not exists"; Stooq's CSV download needs an API
key since early 2026; MarketWatch and Macrotrends block automated clients; the public Quandl WIKI mirror and
the Hugging Face price datasets need an account or cover only surviving symbols). A free API key fills the gap:

1. Get a free key at [tiingo.com](https://www.tiingo.com) (sign up, then Account → API → Token). The free
   plan allows 50 requests an hour, 1,000 a day and 500 different symbols a month, and includes delisted US
   stocks.
2. In your GitHub repository: Settings → Secrets and variables → Actions → New repository secret. Name
   `TIINGO_API_KEY`, value the token.
3. Run the **Fetch price data** workflow (Actions → Fetch price data → Run workflow), or wait for the
   nightly run. Each run fetches up to 45 missing names and keeps them, so the ~90 missing former members
   are filled in over two or three runs. Then `git pull`.

`ALPHAVANTAGE_API_KEY` (alphavantage.co, 25 requests a day on the free plan, 20 used per run) and
`STOOQ_API_KEY` (a free key from https://stooq.com/q/d/?s=aapl.us&get_apikey, after a CAPTCHA) work the same
way. A history from any of these is only used if it passes an identity check: it must trade during the
membership months like a large Nasdaq stock (a recycled symbol - a small company that later took the
ticker - fails), and where a saved file overlaps it the prices must agree. Tiingo's company name is recorded
in `data/delisted.json`. Stooq has no dividends (its histories are price-return only). Every report states
the current member-month coverage.

The **Daily signals** Action then scans the paper-trading strategies (`paper/*.json`) and writes
`signals/latest.md`. It also posts to a webhook if you add a repository secret `ALERT_WEBHOOK_URL`
(for example a Slack or Discord incoming webhook).

### Long-history series (SIMs)

The data job also builds simulated total-return indexes that extend funds back before they
existed, then continue with the real fund's total return. `data/sims_log.txt` lists any series that
failed to build and, for each one, how the model compares with the real fund where both exist
(monthly correlation, tracking error, CAGR and volatility).

| Series | Before the fund (start) | Then |
|---|---|---|
| SPYSIM | US stock market: Fama-French market return (1926, daily) | SPY |
| VTISIM | US total market: Fama-French market return (1926, daily) | the Vanguard Total Stock Market Index fund VTSMX (April 1992), then VTI (2001) |
| VBSIM | US small caps: Fama-French small-cap model (1926, daily; the candidate that tracks the fund best) | the Vanguard Small-Cap Index fund NAESX (from late 1989, when it became an index fund), then VB |
| VBRSIM | US small-cap value: the best-tracking of Fama-French small / high B/M, small high + neutral, or the 25-portfolio size quintiles 2-3 x top two B/M quintiles (1926, daily) | the Vanguard Small-Cap Value Index fund VISVX (1998), then VBR |
| VBKSIM | US small-cap growth: the same choice on the growth side (1926, daily) | the Vanguard Small-Cap Growth Index fund VISGX (1998), then VBK |
| MIDSIM | US mid caps: Fama-French portfolio of the 30th-70th NYSE size percentiles (1926, daily) | MDY (S&P 400) |
| VOESIM | US mid-cap value: Fama-French 25 size x B/M portfolios, middle size quintiles x high B/M (1926, daily) | VOE |
| VOTSIM | US mid-cap growth: the same, low B/M (1926, daily) | VOT |
| VTVSIM | US large-cap value: the best-tracking of Fama-French big / high B/M, 1/3 big-high + 2/3 big-neutral B/M, or the 25-portfolio large-cap top B/M quintiles (1926, daily) | the Vanguard Value Index fund VIVAX (1992), then VTV |
| VUGSIM | US large-cap growth: Fama-French big / low book-to-market or a 25-portfolio blend (1926, daily) | the Vanguard Growth Index fund VIGRX (1992), then VUG |
| VXUSSIM | International stocks: 80% developed ex-US (EFASIM's model) + 20% emerging (EEMSIM's, from 1989), rebalanced daily (1975) | the Vanguard Total International Stock Index fund VGTSX (1996), then VXUS (2011) |
| VWOSIM | Emerging markets: Fama-French (1989, monthly steps) | the Vanguard Emerging Markets Stock Index fund VEIEX (1994), then VWO |
| EWJSIM, EWUSIM, EWGSIM, EWCSIM, EWASIM, EWQSIM, EWLSIM, EWHSIM | Japan, UK, Germany, Canada, Australia, France, Switzerland, Hong Kong: Fama-French country indexes in USD with dividends (1975, monthly steps; Japan daily from 1990) | the iShares country ETF |
| EFASIM | Developed ex-US: Fama-French EAFE index (1975, monthly steps), Fama-French developed ex-US market (1990, daily) | EFA |
| EFVSIM | Developed ex-US value: Fama-French EAFE high book-to-market index (1975, monthly steps), big / high B/M (1990, daily) | EFV |
| SCZSIM | Developed ex-US small caps: Fama-French small portfolios (1990, daily) | SCZ |
| AVDVSIM | Developed ex-US small-cap value: Fama-French small / high B/M (1990, daily) | AVDV |
| VGKSIM | Europe: Fama-French Europe index (1975, monthly steps), Fama-French Europe market (1990, daily) | VGK |
| EEMSIM | Emerging markets: Fama-French emerging market return (1989, monthly steps) | EEM |
| VNQSIM | US REITs: FTSE Nareit All Equity REITs total return (1972, monthly steps) | the Vanguard REIT Index fund VGSIX (1996), then VNQ |
| BILSIM | 1-month T-bills: Fama-French RF (1926, daily) | BIL |
| SHYSIM | 2-year Treasuries priced from the FRED 2-year yield (1-year before 1976) (1962, daily) | SHY |
| IEISIM | 5-year Treasuries from the 5-year yield (1962, daily) | IEI |
| IEFSIM | ~9-year Treasuries from the 10-year yield (1962, daily) | IEF |
| TLTSIM | 20-year Treasuries from FRED constant-maturity yields (1962, daily) | TLT |
| LQDSIM | Investment-grade corporates: 10-year par bond at the average of Moody's Aaa and Baa yields (1953; monthly yields before 1986, daily after) | LQD |
| BNDSIM | US aggregate bonds: 70% 5-year Treasury + 30% corporate model (1962), then the Vanguard Total Bond Market Index fund VBMFX (Dec 1986) | BND |
| VCLTSIM | Long-term IG corporates: 20-year par bond at the average of Moody's Aaa and Baa yields (1953; monthly yields before 1986) | the Vanguard Long-Term Investment-Grade fund VWESX (Yahoo history from 1980), then VCLT (2009) |
| MUBSIM | US municipal bonds: the Vanguard Intermediate-Term Tax-Exempt fund VWITX (Yahoo history); no model before it (no free long muni index) | MUB (2007) |
| EMBSIM | Emerging-market USD bonds: the Fidelity New Markets Income fund FNMIX (1993); no model before it | EMB (2007) |
| TIPSIM | US TIPS: the Vanguard Inflation-Protected Securities fund VIPSX (mid-2000); no model before it (TIPS date from 1997) | TIP |
| HYGSIM | US high yield: the Vanguard High-Yield Corporate fund VWEHX (1985); no model before it | HYG |
| BNDXSIM | International government bonds hedged to USD: a 9-year par-bond model on OECD 10-year yields of up to 12 developed markets, hedged at the short-rate differential (1970, monthly steps), then the PIMCO International Bond (USD-hedged) fund PFORX (1993) | BNDX |
| GLDSIM | Gold: World Bank monthly average price (1960-1968, monthly steps), LBMA PM fixing (April 1968, daily) | GLD |
| DBCSIM | Commodity futures: AQR "Commodities for the Long Run" equal-weight index excess return + T-bills (1960, monthly steps) | the PIMCO CommodityRealReturn Strategy fund PCRIX (mid-2002) when it tracks DBC better than the model on their overlap (the log shows both), then DBC |

Where a model has candidates (VTVSIM, VUGSIM, VBRSIM, VBKSIM, VBSIM, VOESIM, VOTSIM), the data job picks the
one with the lowest tracking error against the fund (the mutual-fund twin where it exists, the longer
overlap) and logs every candidate's score in `data/sims_log.txt`. The old large-value model (Fama-French
big / high B/M alone) is too deep-value: tracking error 8.3% a year against VTV with 20% volatility against
14.5%; 1/3 big-high + 2/3 big-neutral B/M tracks VTV with 3.6% and 16.5% volatility.

**Fee and cost haircut.** The models are gross: index and Fama-French portfolio returns pay no expense
ratio, trading costs, cash drag or tax leakage, and on their overlap they beat the funds by about 1-2% a year
(VOTSIM 11.69% vs VOT 9.88%, VXUSSIM 7.36% vs VGTSX 5.95%, EFVSIM 7.69% vs EFV 6.03%). So before splicing,
the data job takes a constant annual drag off each model segment (never off the real fund after it):
`drag = max(expense ratio, min(3%, gap))`, where `gap` is how much the model's CAGR beat the fund that takes
over from it on their overlap (full months, geometric: (1 + model) / (1 + fund) - 1, 0 when the fund did
better) and the expense ratio is that fund's current one (a table in `scripts/fetch_data.py`, from the
issuers' fund pages as of 2025; older, higher fees show up in the gap). The drag is taken daily in
proportion to calendar time, so monthly-stepped models pay the same per year. With it, a model's CAGR on the
overlap equals the fund's (unless the fund did better or the gap hit the 3% cap, which flags model error
rather than costs). `data/sims_log.txt` logs each series' drag, its basis and the check; `data/sims_drag.json`
holds the figures; SIM descriptions on the site and backtests that hold a SIM during its model period state
it ("model periods are net of an estimated X%/yr fee/cost drag"). Series with no model (TIPSIM, HYGSIM,
MUBSIM, EMBSIM: real funds only) are untouched.

**Fund-exact series.** VTISIM, VXUSSIM, VWOSIM, VNQSIM, BNDSIM, VBSIM, VBRSIM, VBKSIM, VTVSIM and VUGSIM
become the named fund as soon as it or its Vanguard mutual-fund twin (same index, same manager) exists, so a
named portfolio run from 1972 holds, for example, the US market model until April 1992, VTSMX until
mid-2001 and VTI after that. Named portfolios use these first and fall back to the older series (SPYSIM,
EFASIM, ...) until the data job has built them.

**Asset-class names.** After a weight, Portfolio Visualizer's asset-class names are read as the best
long-history series, with a note: "40% US stock market, 20% international stocks, 40% total bond since 1972"
holds VTISIM, VXUSSIM and BNDSIM. Recognised: (US / total) stock market, stocks; US large / mid / small cap,
each with value or growth; international stocks, international developed, international small cap (value),
international value, emerging markets, European stocks, Japan; REITs / real estate; gold; commodities; total
bond / bonds / aggregate bonds; short, intermediate and long term Treasuries; TIPS; corporate bonds;
long-term corporate bonds; high yield; municipal bonds; international bonds; emerging market bonds;
T-bills. Name a fund (VTI, BND, ...) to use the fund alone; "cash" stays cash earning the T-bill rate. Next
to real tickers, names that always meant a fund keep it ("60% SPY and 40% gold" holds GLD), so such a mix
compares fund with fund.

They are total-return indexes: `close` = `adj_close`, no dividends, `volume` 0 and
open = high = low = close. Use them in the optimiser, Monte Carlo and factor pages (and in JSON
specs) for many more market regimes than the ETFs alone. Named portfolios ("the Ivy portfolio since
1975", "all weather since 1972") swap them in for any fund that did not exist yet, and say so in the
notes. Keep in mind:
- the early parts are models or indexes, not tradable funds (no fees, no bid/ask; Fama-French
  portfolios are gross of costs);
- **monthly steps**: a monthly source moves only on the last NYSE session of each month (a month
  that ends on a weekend or holiday is booked on its last trading day, like every real ticker's
  month-end), and is flat in between. Daily-rule strategies see nothing inside those months;
  monthly rebalancing lines up exactly; reports, correlations and factor regressions switch to monthly
  returns where they are held (see the report section);
- the data build fails a SIM (in data/sims_log.txt) that has a hole of more than 10 business days after its
  start; a run that holds one anyway gets a "Data gap" note (LQDSIM and BNDSIM missed 1983-85 until the
  corporate-yield model was fixed to use the monthly Aaa/Baa yields until both daily series exist);
- GLDSIM's 1960-68 part is monthly *average* prices (gold was pegged near $35 then); from April 1968
  it uses the daily London PM fixing, so its month-ends are real month-end prices;
- BNDXSIM's model uses OECD monthly-average yields (smoother than month-end prices: correlation with
  BNDX on the overlap is only about 0.7) and has Japan only from 1989 and Italy from 1991, so it is
  a rough guide before PFORX starts in 1993;
- DBCSIM is an equal-weight commodity index; DBC is energy-heavy, so they share direction (monthly
  correlation about 0.9) but not volatility;
- no free source gives high-yield bonds before 1985 or TIPS before 2000 without a model we could not
  defend, so HYGSIM and TIPSIM start with the oldest real funds instead;
- rules that need intraday prices or volume (gaps, ranges, ATR, volume caps, MFI/VWAP) are
  meaningless on them;
- sentences can name them like any ticker ("hold 60% SPYSIM and 40% TLTSIM", "vs SPYSIM"). A
  portfolio that starts before SPY existed is compared with SPYSIM by default.

## Optimiser inputs

- **Historical** (default): the means and covariance of the fit period's monthly total returns.
- **Forecasts**: `--expected-return "SPY=7%,TLT=4%"` replaces those assets' means; `--expected-vol` and
  `--correlation "SPY/TLT=-0.2"` replace single volatilities and correlations (the rest stay historical; an
  inconsistent correlation matrix is refused).
- **Black-Litterman**: views (`--view "SPY = 8% @ 60%"` absolute, `--view "QQQ > SPY by 2%"` relative;
  confidence 50% by default) and/or a prior (`--prior market-cap|equal|"SPY=60%,TLT=40%"`). The prior's
  implied equilibrium excess returns are pi = delta x cov x w (delta = `--risk-aversion`, 2.5); each view's
  uncertainty is (1 - c)/c x p (tau cov) p' (`--tau` 0.05; 100% confidence makes the view hold exactly); the
  posterior mean pi + tau cov P'(P tau cov P' + Omega)^-1 (Q - P pi) and covariance cov + M replace the
  historical ones. Funds have no market capitalisation here, so an unspecified prior falls back to equal
  weights with a note.
- Scenario-based objectives (Sortino, CVaR, Omega, return/drawdown) use the historical months re-shaped
  (linearly) to have exactly the forecast/posterior mean and covariance.
- **Benchmark** (`--benchmark SPY` or `"60% SPY 40% AGG"`): min tracking error (subject to `--target-active`,
  an excess return over the benchmark, or `--target-return`) and max information ratio, and every portfolio's
  tracking error, active return and information ratio. A benchmark made of the optimised tickers is
  investable: min tracking error then returns its weights exactly.
- **Resampling** (`--resample N`, Michaud): N histories of the same length are drawn from a multivariate
  normal with the inputs' means and covariance, each is optimised, and the weights are averaged
  ("Resampled max Sharpe", ...; ± shows each weight's spread across draws); the resampled frontier averages
  the frontier portfolios rank by rank.

## Monte Carlo

Monthly steps over complete months (a month still in progress at the end of the data is left out; a history
window `--start 1972-01-01` starts with January 1972, measured from the December month-end, and one starting
mid-month starts with the next whole month). Return models: **historical** (block bootstrap: whole months are drawn together for
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
- *By percentile* (as Portfolio Visualizer): each path's own safe rate (the highest it pays in full to the
  end) and perpetual rate (the highest that also keeps its real balance), at the 10th/25th/50th/75th/90th
  percentile of the paths. The 10th percentile is the cautious figure: 90% of paths sustain at least that.
  For a plan that contributes and then withdraws, the rates are a share of each path's balance at the start
  of the withdrawals (and run over the remaining years).
- The first flow is at the very start, as in the backtest, so a portfolio sentence gives the same number of
  contributions in both.
- Stress tests: `worst_sequence` puts the worst historical run of N years (default 10) at the start of every
  path; `shock` makes the first year return -30% (or your figure). A horizon can be given by age
  (`--age 65 --until-age 95`).
- Cash-flow phases: on the site, add one row per phase (e.g. contribute $20,000 a year in years 1-15, then
  withdraw $60,000 a year from year 16), each with its own frequency and inflation setting. From the command
  line, a portfolio sentence with a contribution and a withdrawal does the same.
- Withdrawals never take more than the balance: a path that cannot pay in full pays what is left and ends at
  zero (no negative balances). The results show the total actually withdrawn and the share of paths that
  fell short.
- Glide path (`--glide-to "VTISIM 40 BNDSIM 60" [--glide-years 30] [--glide linear|target_date]
  [--glide-points "0:0,10:0,20:50,30:100"]`, or "Glide path" under the weights on the site): `--weights` is the
  start mix and the target moves once a year to the end mix, the portfolio being rebalanced to it at every year
  boundary (and on its schedule in between). Linear: year 1 holds the start mix, year N the end mix, even steps
  in between (90/10 -> 40/60 over 30 years moves 1.7 points a year). `target_date`: the start mix for the first
  fifth of the glide, then de-risking that speeds up toward the end (40% of the move by 60% of the time), the
  shape of published target-date glide paths. Custom points are (years elapsed: % of the way), joined linearly.
  Returns are drawn for every asset of both mixes together, so correlations hold; the results list the target
  mix by year. A glide path needs tickers and weights (not a sentence or a saved run).
- Lifetime horizon (`--horizon mortality --age 65 --sex male|female|joint [--age2 63]`, or "a lifetime" on
  the site): the paths run until the chance of being alive falls below 0.1%, and the chance of success is
  weighted by survival - the sum over years of P(death that year) x P(money left at the end of that year),
  plus the chance of outliving the horizon times the success then. Mortality is the SSA 2023 period life
  table (Social Security area population, as used in the 2026 Trustees Report, embedded in
  `backtester/lifetable.py`); a couple is two independent lives, a man and a woman, and the money must last
  until the second death. A period table assumes no future mortality improvement, so it slightly
  understates lifetimes.

## Factor analysis

- **Monthly** regressions use Kenneth French's official monthly factor files, which French builds from
  monthly portfolio returns (they differ slightly from compounded daily factors). Until the data job has
  downloaded a monthly file, its daily counterpart is compounded instead and the report says so.
  **Daily** regressions use the daily files.
- **Regions**: `developed` (incl. the US), `developed_ex_us` (`dev_ff3`, alias `intl`; short prefix `dev`),
  `europe`, `japan`, `asia_pacific_ex_japan` (`apxj`), `north_america` (`na`), `emerging` (`em`, monthly
  only), each with `_ff3`, `_ff5`, `_carhart` and `_ff6` (e.g. `europe_ff5`, `em_carhart`). Returns are in
  US dollars and the risk-free rate is the US T-bill. `--model auto` picks the region from known tickers
  (EFA/VEA -> developed ex US, VGK -> Europe, EWJ -> Japan, EEM/VWO -> emerging, VT/ACWI -> developed), and
  a note suggests the regional model when a known ticker is run on another region's factors.
- **Add-ons**: `+mom`, `+qmj`, `+bab`, `+bonds` (`+term`, `+def`) on any model: `ff5+qmj+bab`,
  `europe_ff3+qmj`. QMJ and BAB are AQR's monthly factors (Asness, Frazzini and Pedersen) for the model's
  region: USA, Global (developed), Global Ex USA, Europe, North America or JPN. AQR publishes no emerging
  or Asia Pacific ex Japan aggregate, and they are monthly only.
- **Style analysis** (`python -m backtester style QQQ`, or "Style analysis" on the Factors page): Sharpe's
  (1992) returns-based style analysis - the non-negative weights, adding up to 100%, on asset-class returns
  that minimise the variance of the tracking difference. Default classes (the first series with data):
  US large value (VTVSIM/VTV), US large growth (VUGSIM/VUG), US small value (VBRSIM/VBR), US small growth
  (VBKSIM/VBK), developed ex US (EFASIM/EFA), emerging (EEMSIM/EEM), Treasuries (IEFSIM/IEF), corporates
  (LQDSIM/LQD) and T-bills (BILSIM/BIL); `--assets` sets your own. Reports R² (share of the variance the mix
  explains), the selection return, tracking error, and rolling 36-month weights (a stacked chart on the site).

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
