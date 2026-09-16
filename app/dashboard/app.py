"""Operator dashboard.

Design principle: nothing that is broken may be invisible. Missing credentials,
an unreachable broker, stale data, a failed refresh, an engaged kill switch and
a config error all render as prominent status, never as a blank panel or a
silently empty table.

Run with::

    python scripts/run_dashboard.py
    launchers\\run_dashboard.bat        (Windows)

Colours come from Streamlit's own theme tokens, so the page follows whichever
light or dark theme the viewer has configured.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.clock import MarketClock, age_seconds, to_display_tz, utcnow  # noqa: E402
from app.config import AppConfig, ConfigError, load_config  # noqa: E402
from app.enums import AssetClass, TradingMode  # noqa: E402
from app.logging import AuditLog  # noqa: E402

st.set_page_config(page_title="Trading Platform", page_icon="📈", layout="wide")

MODE_STYLE = {
    TradingMode.SIGNAL_ONLY: ("🟢", "SIGNAL ONLY", "No orders are transmitted."),
    TradingMode.BACKTEST: ("🔵", "BACKTEST", "No execution route exists for this mode."),
    TradingMode.PAPER: ("🟡", "PAPER", "Orders ARE transmitted to the Alpaca paper account."),
    TradingMode.LIVE: ("🔴", "LIVE", "REAL MONEY. Orders are transmitted to a live account."),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_app_config() -> AppConfig | None:
    try:
        return load_config()
    except ConfigError as exc:
        st.error(f"**Configuration error — the platform cannot run.**\n\n```\n{exc}\n```")
        return None


@st.cache_data(ttl=5)
def read_latest_scan(path_str: str) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def freshness_badge(label: str) -> str:
    return {
        "realtime": "🟢 real-time",
        "snapshot": "🟢 snapshot",
        "delayed": "🟠 DELAYED",
        "historical": "🔵 historical",
        "derived": "🔵 derived",
        "unknown": "🔴 UNKNOWN",
    }.get(label, f"🔴 {label}")


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def render_header(config: AppConfig) -> None:
    icon, label, note = MODE_STYLE[config.mode]
    st.title("📈 Trading Platform")

    if config.mode is TradingMode.LIVE:
        st.error(f"## {icon} MODE: {label}\n\n**{note}**")
    elif config.mode is TradingMode.PAPER:
        st.warning(f"### {icon} MODE: {label} — {note}")
    else:
        st.success(f"### {icon} MODE: {label} — {note}")


def render_status_row(config: AppConfig, scan: dict | None) -> None:
    columns = st.columns(5)

    # --- last refresh + staleness ---
    with columns[0]:
        if scan and scan.get("finished_at"):
            finished = datetime.fromisoformat(scan["finished_at"])
            age = age_seconds(finished)
            local = to_display_tz(finished, config.data.display_timezone)
            stale = age > max(config.scanner.refresh_seconds * 3, 300)
            st.metric(
                "Last scan",
                local.strftime("%H:%M:%S"),
                delta=f"{age:.0f}s ago",
                delta_color="inverse" if stale else "off",
            )
            if stale:
                st.error(f"STALE: last scan was {age / 60:.1f} min ago.")
        else:
            st.metric("Last scan", "never")
            st.warning("No scan has run yet.")

    # --- kill switch ---
    with columns[1]:
        engaged = config.risk.kill_switch_file.exists()
        st.metric("Kill switch", "ENGAGED" if engaged else "off")
        if engaged:
            st.error("All orders are being REJECTED.")

    # --- data feed ---
    with columns[2]:
        st.metric("Data feed", config.data.feed)
        label = {"sip": "realtime", "iex": "realtime", "delayed_sip": "delayed"}.get(
            config.data.feed, "unknown"
        )
        st.caption(freshness_badge(label))
        if label != "realtime":
            st.warning("This feed is refused for order execution.")

    # --- credentials ---
    with columns[3]:
        present = config.credentials.is_complete
        st.metric("Broker credentials", "present" if present else "MISSING")
        if not present and config.mode.submits_real_orders:
            st.error("Credentials are required in this mode.")
        elif not present:
            st.caption("Equity/option data unavailable without keys.")

    # --- market session ---
    with columns[4]:
        clock = MarketClock(display_tz=config.data.display_timezone)
        session = clock.session_for(AssetClass.US_EQUITY, now=utcnow()).session
        st.metric("US equity session", session.value)
        st.caption(f"crypto: {clock.session_for(AssetClass.CRYPTO).session.value}")


def render_live_gates(config: AppConfig) -> None:
    from app.execution.live_gate import live_gate_status

    status = live_gate_status(config)
    with st.expander("🔒 Live-trading gates", expanded=bool(status["all_gates_open"])):
        if status["all_gates_open"]:
            st.error("**ALL LIVE GATES ARE OPEN — real money is at risk.**")
        else:
            st.success("Live trading is **blocked**. At least one gate is closed.")
        st.dataframe(
            pd.DataFrame(
                [
                    {"gate": name, "open": bool(value)}
                    for name, value in status.items()
                    if name != "all_gates_open"
                ]
            ),
            width="stretch",
            hide_index=True,
        )


def render_signals(scan: dict | None) -> None:
    st.subheader("Signals")
    if not scan:
        st.info("No scan output yet. Run `python scripts/run_scanner.py --once`.")
        return

    signals = scan.get("signals") or []
    if not signals:
        st.info("The last scan produced no signals.")
        return

    frame = pd.DataFrame(signals)
    columns = [
        c for c in [
            "symbol", "direction", "strategy", "confidence", "reference_price",
            "stop_price", "target_price", "reward_risk_ratio", "data_freshness",
            "data_timestamp",
        ] if c in frame.columns
    ]
    st.dataframe(frame[columns], width="stretch", hide_index=True)

    unsafe = frame[~frame["data_freshness"].isin(["realtime", "snapshot"])]
    if not unsafe.empty:
        st.warning(
            f"{len(unsafe)} signal(s) are derived from data that is not labelled "
            "real-time. They will be refused for order execution."
        )


def render_scan_results(scan: dict | None) -> None:
    st.subheader("Scanner results")
    if not scan:
        st.info("No scan output yet.")
        return
    results = scan.get("scan_results") or []
    if not results:
        st.info("No symbols passed the scanner filters in the last run.")
        return
    st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)


def render_data_health(scan: dict | None) -> None:
    st.subheader("Data health")
    if not scan:
        st.info("No scan output yet.")
        return

    requested = scan.get("symbols_requested", 0)
    received = scan.get("bar_sets_received", 0)
    failures = scan.get("data_failures") or {}
    errors = scan.get("errors") or []

    columns = st.columns(3)
    columns[0].metric("Symbols requested", requested)
    columns[1].metric("Symbols with data", received)
    columns[2].metric("Data failures", len(failures))

    if requested and received == 0:
        st.error(
            "**No market data was received for any symbol.** Signals cannot be "
            "generated. Run `python scripts/preflight.py` to diagnose."
        )
    if failures:
        with st.expander(f"⚠️ {len(failures)} symbol(s) returned no data", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [{"symbol": s, "reason": r} for s, r in failures.items()]
                ),
                width="stretch",
                hide_index=True,
            )
    if errors:
        st.error("Errors during the last run:")
        for error in errors[:20]:
            st.code(error)


def render_portfolio(scan: dict | None) -> None:
    st.subheader("Account & positions")
    portfolio = (scan or {}).get("portfolio")
    if not portfolio:
        st.info("No portfolio snapshot in the last scan output.")
        return

    account = portfolio.get("account", {})
    if not account.get("is_known"):
        st.warning(
            "**Account state is UNKNOWN** — "
            f"{account.get('error') or 'the broker could not be reached'}. "
            "Risk checks that depend on account state will reject every order."
        )
        return

    columns = st.columns(5)
    columns[0].metric("Equity", f"{account.get('equity', 0):,.2f}")
    columns[1].metric("Buying power", f"{account.get('buying_power', 0):,.2f}")
    columns[2].metric(
        "Daily P/L", f"{account.get('daily_pl', 0):,.2f}",
        delta=f"{account.get('daily_pl_pct', 0):.2f}%",
    )
    columns[3].metric("Open positions", portfolio.get("open_position_count", 0))
    columns[4].metric("Account mode", "paper" if account.get("is_paper") else "LIVE")

    if account.get("trading_blocked"):
        st.error("The broker has **blocked trading** on this account.")

    positions = portfolio.get("positions") or []
    if positions:
        st.dataframe(pd.DataFrame(positions), width="stretch", hide_index=True)
    else:
        st.caption("No open positions.")


def render_orders(scan: dict | None) -> None:
    st.subheader("Orders")
    orders = (scan or {}).get("order_results") or []
    if not orders:
        st.caption("No orders in the last run.")
        return

    rows = [
        {
            "symbol": o["intent"]["symbol"],
            "side": o["intent"]["side"],
            "qty": o["intent"]["quantity"],
            "type": o["intent"]["order_type"],
            "status": o["status"],
            "simulated": o["is_simulated"],
            "broker": o["broker"],
            "rejection_reason": o.get("rejection_reason"),
            "message": (o.get("message") or "")[:160],
        }
        for o in orders
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    simulated = sum(1 for o in orders if o["is_simulated"])
    if simulated:
        st.info(
            f"{simulated} of {len(orders)} order(s) were **recorded intents only** — "
            "nothing was transmitted to a broker."
        )


def render_risk(config: AppConfig) -> None:
    st.subheader("Risk limits in force")
    risk = config.risk
    # Every value is rendered as text: a column mixing numbers, percentages and
    # booleans cannot be serialised to a single Arrow type, which would blank
    # the whole table.
    rows = [
        ("Max open positions", str(risk.max_open_positions)),
        ("Max position notional", f"{risk.max_position_notional:,.2f}"),
        ("Max position % equity", f"{risk.max_position_pct_equity:.1%}"),
        ("Max symbol concentration", f"{risk.max_symbol_concentration_pct:.1%}"),
        ("Max daily loss", f"{risk.max_daily_loss:,.2f}"),
        ("Max daily loss %", f"{risk.max_daily_loss_pct:.1%}"),
        ("Max trades per day", str(risk.max_trades_per_day)),
        ("Min account equity", f"{risk.min_account_equity:,.2f}"),
        ("Stop loss required", "yes" if risk.require_stop_loss else "no"),
        ("Extended hours allowed", "yes" if risk.allow_extended_hours else "no"),
        ("Block when market closed", "yes" if risk.block_when_market_closed else "no"),
        ("Duplicate window (s)", str(risk.duplicate_order_window_seconds)),
        ("Kill switch file", str(risk.kill_switch_file)),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["limit", "value"]),
        width="stretch",
        hide_index=True,
    )


def render_audit(config: AppConfig) -> None:
    st.subheader("Audit trail")
    audit = AuditLog(config.logging.dir / "audit.jsonl")
    entries = audit.tail(200)
    if not entries:
        st.info(f"No audit entries yet ({audit.path}).")
        return

    rows = [
        {
            "ts": e.get("ts", ""),
            "event": e.get("event", ""),
            "detail": json.dumps(e.get("payload", {}), default=str)[:200],
        }
        for e in reversed(entries)
    ]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=360)
    st.caption(f"Source: {audit.path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    config = load_app_config()
    if config is None:
        st.stop()

    with st.sidebar:
        st.header("Controls")
        if st.button("🔄 Refresh now", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        auto = st.checkbox("Auto-refresh", value=False)
        if auto:
            st.caption(f"Every {config.scanner.refresh_seconds}s")
            st.markdown(
                f'<meta http-equiv="refresh" content="{config.scanner.refresh_seconds}">',
                unsafe_allow_html=True,
            )
        st.divider()
        st.caption(f"Config: `{config.source_path}`")
        st.caption(f"Logs: `{config.logging.dir}`")
        st.caption(f"Outputs: `{config.outputs_dir}`")
        st.caption(f"Scanners: {', '.join(config.scanner.enabled) or 'none'}")
        st.caption(f"Strategies: {', '.join(config.strategies.enabled) or 'none'}")

    scan = read_latest_scan(str(config.outputs_dir / "latest_scan.json"))

    render_header(config)
    render_status_row(config, scan)
    render_live_gates(config)
    st.divider()

    signals_tab, scan_tab, data_tab, portfolio_tab, risk_tab, audit_tab = st.tabs(
        ["Signals", "Scanner", "Data health", "Account", "Risk", "Audit"]
    )
    with signals_tab:
        render_signals(scan)
        st.divider()
        render_orders(scan)
    with scan_tab:
        render_scan_results(scan)
    with data_tab:
        render_data_health(scan)
    with portfolio_tab:
        render_portfolio(scan)
    with risk_tab:
        render_risk(config)
    with audit_tab:
        render_audit(config)


main()
