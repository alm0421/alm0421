# Trading Platform

Signal generation, pre-trade risk management and paper-trading execution for
**US equities** and **crypto**, built against the Alpaca API, with a Streamlit
operator dashboard and Windows-first launchers.

> **Status: signal-only by default. Live trading is disabled at build level and
> has never been validated against a live account.** See
> [Execution modes](#execution-modes).

---

## Quick start (Windows)

```bat
launchers\setup.bat              :: create .venv, install deps, create .env
launchers\preflight.bat          :: validate config, data, broker, symbols
launchers\run_scanner.bat        :: generate signals (transmits nothing)
launchers\run_dashboard.bat      :: open the operator dashboard
```

## Quick start (macOS / Linux)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env             # then add your Alpaca PAPER keys
python scripts/preflight.py
python scripts/run_scanner.py --once
python scripts/run_dashboard.py
```

The dashboard opens at <http://localhost:8501>.

---

## Execution modes

The `mode` key in `config/config.yaml` is the most important setting in the
project. It alone decides whether an order is recorded, sent to a paper
account, or sent to a live one. `TRADING_MODE` in the environment overrides it.

| Mode | What happens to an order | Credentials needed |
|---|---|---|
| **`signal_only`** *(default)* | Recorded as an intent. **Nothing is transmitted.** | None |
| `backtest` | **Refused.** No simulation engine ships in this release. | — |
| `paper` | Transmitted to the Alpaca **paper** endpoint. No real money. | Paper keys |
| `live` | **Blocked.** See below. | — |

All routing goes through one function, `app/execution/router.py::build_broker`.
Nothing else constructs a broker, so that file is the single place to audit for
mode separation.

### Live trading is blocked by three independent gates

Live trading requires **all three** of these, in three different places, so no
single edit or leaked file can enable real-money trading by accident:

1. **Build gate** — `LIVE_TRADING_APPROVED = False` in
   `app/execution/live_gate.py`. A reviewed source change.
2. **Config gate** — `execution.live_trading_enabled: true` **and** `mode: live`.
3. **Runtime gate** — `ALPACA_LIVE_TRADING_ACK` set to the exact
   acknowledgement phrase on the machine that is running.

**Gate 1 ships closed.** Opening gates 2 and 3 alone does nothing — there is a
test that asserts exactly this
(`tests/test_mode_separation.py::test_live_blocked_even_with_config_and_env_and_credentials`).

The full enablement procedure is documented at the top of
`app/execution/live_gate.py`. Do not shortcut it.

---

## Safety model

**Market data is always labelled, never assumed.** Every quote and bar carries
a `Freshness` value — `realtime`, `delayed`, `historical`, `snapshot`,
`derived` or `unknown`. Only `realtime` and `snapshot` data may size or trigger
an order; everything else is refused by the risk engine. A quote older than
`data.stale_quote_seconds`, or dated in the future (clock skew), is refused too.

**Every check is fail-closed.** An unreachable broker produces an *unknown*
account, not an empty one, and an unknown account rejects every order. Missing
data rejects. An unrecognised feed is labelled `unknown` and rejects.

**Two independent layers guard every order.**
`app/execution/validation.py` checks that the order is internally coherent and
permitted by config (no inverted brackets, no market orders outside regular
hours, correct time-in-force per asset class). `app/risk/engine.py` then checks
it against live account and market state. Both must pass.

**A kill switch halts everything without a restart.** Creating
`logs/KILL_SWITCH` makes the risk engine reject every order immediately.

```bat
launchers\kill_switch.bat on     :: halt
launchers\kill_switch.bat off    :: resume
launchers\kill_switch.bat        :: show state
```

**Nothing that is broken is invisible.** Failed data loads, disconnections,
stale data, unknown account state and config errors all surface on the
dashboard and in the console — never as a silently empty table.

**Everything is auditable.** Every signal, risk verdict, order intent and broker
response is appended to `logs/audit.jsonl` as one JSON object per line.

### Risk limits

Configured under `risk:` in `config/config.yaml` and enforced in
`app/risk/engine.py`:

position count · position notional · position % of equity · per-symbol
concentration (counting existing exposure, so repeated adds cannot creep past
the cap) · daily loss (absolute and %) · trades per day · minimum account
equity · buying power · mandatory stop loss · market-hours and extended-hours
restriction · duplicate-order suppression · broker-side trading block · kill
switch.

Position sizing is **risk-based**: the share count comes from the distance to
the stop, then gets clamped by every notional cap. A signal with no stop cannot
be sized and is refused rather than falling back to an arbitrary size.

---

## What is and is not supported

| | Data | Trading |
|---|---|---|
| US equities | ✅ | ✅ (paper) |
| Crypto | ✅ | ✅ (paper) |
| US options | ✅ | ⚠️ modelled, but not in `allowed_asset_classes` by default; needs Alpaca options approval |
| **Futures** | ❌ | ❌ |

**Alpaca does not offer futures trading or futures market data.** Futures is
modelled throughout (`AssetClass.FUTURES`) so that sizing, risk and routing are
not hardcoded to equities, but it is rejected explicitly at order validation
rather than mis-routed as an equity. Putting `futures` in
`execution.allowed_asset_classes` is a hard config error. Futures support needs
a different broker adapter (IBKR, Tradovate).

---

## Project layout

```
app/
  config.py          Typed config loading, validation, credential resolution
  clock.py           Market sessions, timezones, staleness arithmetic
  enums.py           Shared vocabulary (modes, asset classes, freshness)
  pipeline.py        End-to-end orchestration
  data/              Market data providers; everything is freshness-labelled
  scanners/          Universe filtering and ranking
  signals/           Indicators (lookahead-safe) and the signal schema
  strategies/        Strategy evaluation; pure, no I/O
  risk/              Pre-trade limits, position sizing, kill switch
  execution/         Broker adapters, order validation, MODE GATING
  portfolio/         Account and position snapshots
  dashboard/         Streamlit operator dashboard
  logging/           Structured logging and the audit trail
config/              config.yaml, universe.yaml
scripts/             preflight.py, run_scanner.py, run_dashboard.py
launchers/           Windows .bat and .ps1 launchers
tests/               180 tests
logs/  outputs/      Runtime output (gitignored)
```

## Tests

```bash
python -m pytest            # 180 tests, no network access required
```

Coverage focuses on the things that would cost money if wrong: mode separation,
the live gate, every risk limit, order validation, position sizing, and
indicator lookahead safety (proved by prefix-stability: truncating the future
must not change any past value).

---

## Secrets

Credentials live only in `.env`, which is gitignored. They are never written to
config, never logged, and `Credentials.__repr__` is overridden so a stray
traceback cannot leak them. `AppConfig.as_redacted_dict()` is the log-safe view.

**This repository is public.** Never commit `.env`, and never paste keys into
`config.yaml` or an issue.

---

## Further reading

`INSTRUCTIONS.md` — operating the platform, tuning strategies and risk, adding
a scanner or strategy, troubleshooting, and the full live-enablement procedure.
