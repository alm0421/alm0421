# Operating Instructions

Practical guide to running, tuning and extending the platform. For the
architecture and safety model, see `README.md`.

---

## 1. First-time setup

### Windows

```bat
launchers\setup.bat
```

This creates `.venv`, installs `requirements.txt`, creates `logs\` and
`outputs\`, and copies `.env.example` to `.env` if it does not exist.

Then edit `.env` and add your Alpaca **paper** keys from
<https://app.alpaca.markets/paper/dashboard/overview>:

```
ALPACA_PAPER_API_KEY=PK...
ALPACA_PAPER_SECRET_KEY=...
```

Leave the `ALPACA_LIVE_*` variables blank. Filling them in does **not** enable
live trading, but there is no reason for them to be present.

### macOS / Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Verify

```bat
launchers\preflight.bat
```

Preflight exits `0` if everything passed, `1` if a check failed, `2` if the
config could not load. It places no orders and changes no configuration.

Useful flags:

| Flag | Effect |
|---|---|
| `--skip-network` | Offline validation only. Connectivity is reported as UNVERIFIED. |
| `--verify-symbols` | Ask the broker whether each universe symbol is tradable. Requires paper mode and credentials. |

---

## 2. Daily operation

```bat
launchers\run_scanner.bat                :: one cycle
launchers\run_scanner.bat --loop         :: continuous, every scanner.refresh_seconds
launchers\run_dashboard.bat              :: dashboard on http://localhost:8501
```

Each cycle writes two files to `outputs/`:

* `latest_scan.json` — overwritten each run; this is what the dashboard reads.
* `scan_YYYYMMDD_HHMMSS.json` — a timestamped history entry.

Both are written atomically (temp file, then rename), so a reader never sees a
half-written file.

### Reading the console line

```
[2026-09-16 09:35:00 EDT] cycle 3: 10/12 symbols with data, 4 scan hits,
    1 signals, 0 orders, 2 data failures, 0 errors (1.84s)
```

`data failures` is the count of symbols that returned no usable bars. It is
printed explicitly because a symbol silently dropping out of the scan is the
kind of failure that otherwise goes unnoticed for days.

### The dashboard

Six tabs: **Signals**, **Scanner**, **Data health**, **Account**, **Risk**,
**Audit**. The header shows the current mode in colour, and the status row shows
last scan time with a staleness warning, kill-switch state, the data feed and
its freshness label, credential presence, and the current market session.

Auto-refresh is off by default; enable it in the sidebar.

---

## 3. Switching to paper trading

Paper trading **does** transmit orders to Alpaca. No real money is at risk, but
everything else behaves as it would live, which is the point.

1. Confirm paper credentials are in `.env`.
2. Set `mode: paper` in `config/config.yaml`, or set `TRADING_MODE=paper` in the
   environment for a single run.
3. Run `launchers\preflight.bat`. In paper mode, a broker or data connectivity
   failure is a **FAIL**, not a warning. Do not proceed until it passes.
4. Run with execution enabled:

   ```bat
   launchers\run_scanner.bat --execute
   ```

`--execute` only *permits* order placement. The mode and every risk check still
decide whether anything is actually sent. Passing `--execute` in `signal_only`
mode transmits nothing and says so.

### What will still block an order in paper mode

- the kill switch (`logs/KILL_SWITCH`)
- account state that could not be fetched
- the connected account not being a paper account
- a quote that is missing, stale, or not labelled real-time
- the US equity market being closed (crypto is exempt — it is 24/7)
- any breached risk limit
- a missing stop loss, while `risk.require_stop_loss` is true
- an identical order within `risk.duplicate_order_window_seconds`

Every rejection is written to `logs/audit.jsonl` with the exact limit breached,
the observed value, and the limit value.

---

## 4. The kill switch

```bat
launchers\kill_switch.bat on     :: halt all order submission
launchers\kill_switch.bat off    :: resume
launchers\kill_switch.bat        :: show current state
```

It is a file (`logs/KILL_SWITCH`), not an in-process flag, so it can be engaged
from outside the process — by a person, a script or a scheduled task — and takes
effect on the next order without a restart. It is the first check the risk
engine runs.

---

## 5. Tuning

All tuning is configuration; no code change is needed.

### Universe — `config/universe.yaml`

Symbols are grouped by asset class. Crypto pairs use Alpaca's slash format
(`BTC/USD`, not `BTCUSD`) — the broker adapter normalises common forms, but the
slash form is what the API expects. Options use OCC contract symbols.

### Scanner — `config/config.yaml` under `scanner:`

The momentum scanner's own filters live in
`app/scanners/momentum_scanner.py::MomentumScanner.DEFAULTS`: price floor and
ceiling, average volume, relative volume, absolute momentum, and ATR percentage
bands. Override any of them by adding a `momentum:` block under `scanner:`.

| Symptom | Adjust |
|---|---|
| Too few candidates | lower `min_relative_volume` or `min_abs_momentum_pct` |
| Too much noise | raise `min_relative_volume`; raise `min_avg_volume` |
| Candidates too volatile to size | lower `max_atr_pct` |

### Strategy — `strategies.ema_pullback` in `config/config.yaml`

| Parameter | Effect |
|---|---|
| `fast_period` / `slow_period` / `trend_period` | EMA lengths. Must satisfy fast < slow < trend. |
| `rsi_floor` / `rsi_ceiling` | The band an entry must sit inside. Narrow it to demand cleaner pullbacks. |
| `stop_atr_multiple` | Stop distance in ATR. Wider = fewer stop-outs, smaller position size. |
| `target_atr_multiple` | Target distance. Must exceed `stop_atr_multiple`. |
| `min_confidence` | Drops low-conviction signals. |
| `allow_short` | Set false for long-only. |

Bad combinations are rejected at construction with a specific message, not at
run time.

> A note on `rsi_ceiling`: raising it admits more signals but also admits
> extended moves where the ATR stop sits far below entry, which shrinks the
> position size. That interaction is easy to miss.

### Risk — `risk:` in `config/config.yaml`

Start conservative. `max_position_pct_equity` and `max_position_notional` are
usually the binding constraints on a small account — the dashboard's sizing
detail names which one bound.

---

## 6. Extending

### Add a strategy

1. Create `app/strategies/my_strategy.py`.
2. Subclass `Strategy`, set `name`, decorate with `@register`.
3. Implement `min_bars` and `evaluate(bars) -> Signal | None`.
4. Validate parameters in `_validate_params`, raising `StrategyError`.
5. Add `my_strategy` to `strategies.enabled` **and** a `my_strategy:` parameter
   block (an enabled strategy with no parameter block is a config error).
6. Import it in `app/strategies/__init__.py` so registration runs.

Rules: no I/O, no wall-clock reads, no consulting a future bar. The same
`BarSet` must always produce the same `Signal` — that is what makes live and
simulated runs comparable.

### Add a scanner

The same pattern with `app/scanners/base.py::Scanner`, `@register`, and
`scanner.enabled`.

### Add a broker

Implement `app/execution/base.py::Broker` and add a branch to
`build_broker`. Requirements: bind the adapter to a single `TradingMode` at
construction, verify on connect that the account mode matches, and return
`PortfolioState.unknown(...)` — never an empty portfolio — when the broker
cannot be reached.

---

## 7. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Configuration error` on startup | The message names the exact key. Fix `config/config.yaml`. |
| Preflight: credentials MISSING | Add Alpaca paper keys to `.env`. Fine in signal-only mode, fatal in paper mode. |
| `no bars returned` for every symbol | No connectivity or no credentials. Equity data needs keys; crypto does not. Run `preflight.py`. |
| No signals, but the scanner has hits | Normal. The strategy wants a specific setup. Lower `min_confidence` to inspect weaker signals. |
| `order_not_sized` in the audit log | Usually unknown account equity (expected in signal-only mode — there is no account). Otherwise the account is too small for the price and stop distance. |
| Every order rejected with `stale_data` | The feed is `delayed_sip`, or quotes are older than `data.stale_quote_seconds`, or the machine clock is skewed. |
| Every order rejected with `market_closed` | Correct outside 09:30–16:00 ET. Crypto is exempt. Set `risk.allow_extended_hours` only with limit orders. |
| `ZoneInfoNotFoundError` on Windows | `pip install tzdata`. The standard library ships no tz database on Windows. |
| `.bat` file fails with a label error | The file was checked out with LF endings. `.gitattributes` forces CRLF; re-clone or run `git add --renormalize .`. |
| PowerShell refuses to run `.ps1` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| Dashboard shows a stale scan | The scanner is not running. Start `run_scanner.bat --loop`. |

### Where things are logged

| File | Contents |
|---|---|
| `logs/platform.log` | Structured JSON, one object per line; rotates at 10 MB, 5 backups. |
| `logs/audit.jsonl` | Append-only decision trail: signals, risk verdicts, order intents, broker responses. |
| `outputs/latest_scan.json` | The most recent cycle, as read by the dashboard. |
| `outputs/scan_*.json` | Timestamped history. |

---

## 8. Enabling live trading

**Live trading is disabled in this release and has never been validated against
a live account.** The paper path itself has not yet been validated end to end
against a real Alpaca paper account either — see the "Validation status" note in
the pull request.

Do not skip any step:

1. Run the platform in **paper mode** for a meaningful period. Confirm order
   construction, fills, bracket behaviour, every risk limit, and reconciliation
   after a failed submission.
2. Review `logs/audit.jsonl` and confirm every rejection was correct and every
   approval was intended.
3. Obtain **explicit written approval** from the account owner.
4. Set `LIVE_TRADING_APPROVED = True` in `app/execution/live_gate.py` in a
   reviewed commit.
5. Set `execution.live_trading_enabled: true` and `mode: live`.
6. Export `ALPACA_LIVE_TRADING_ACK` with the exact phrase from
   `app/execution/live_gate.py`.
7. Set `ALPACA_LIVE_API_KEY` and `ALPACA_LIVE_SECRET_KEY`.
8. Run `preflight.py` and confirm the broker reports a **live** account.
9. Start with the smallest possible `max_position_notional` and
   `max_daily_loss`, and keep `launchers\kill_switch.bat` within reach.

`tests/test_mode_separation.py::test_build_gate_is_closed_in_this_release` will
fail the moment step 4 is taken. That is deliberate: flipping the build gate
should require a conscious decision about that test, not slip through unnoticed.
