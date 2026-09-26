"""Send today's orders to a broker: Alpaca (paper trading by default).

    python -m backtester trade "<sentence>" --broker alpaca --dry-run     # print the orders, send nothing
    python -m backtester trade "<sentence>" --broker alpaca               # paper account
    ALPACA_LIVE=1 python -m backtester trade "<sentence>" --live          # real money: both switches needed

Keys come from the environment: ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (never from the command line,
never printed or logged). The paper endpoint (paper-api.alpaca.markets) is used unless ALPACA_LIVE=1 is set
AND --live is given.

What is sent (see `plan`):

- The target comes from orders.todays_orders, the same reconciliation the site's Orders page uses: what the
  strategy holds after the latest bar (allocations: the tree evaluated today, as a rebalance would trade;
  signals: the backtest's open positions plus new entries), sized to the broker account's equity, minus the
  broker's current positions. Sells are listed before buys.
- Timing follows the strategy: fills at the close -> market-on-close orders (time_in_force "cls"); at the open
  or the next open -> market-on-open ("opg"). Limit/stop entries -> limit/stop orders at the rule's level,
  with the stop loss / take profit attached as a bracket (one-cancels-other) when the strategy has them.
- Open positions of a signal strategy get their exit orders for the next session: the stop and the target as
  an OCO pair (or a single stop / limit order), scale-outs as limit orders - the levels the backtest uses.
- Fractional quantities (allocation portfolios rebalanced to exact weights) must be day market orders at
  Alpaca (no "cls"/"opg"): they are sent as such, with a note. Use --whole-shares to keep MOC/MOO timing.
- Every order carries a deterministic client_order_id (strategy, date, symbol, role), so running twice on the
  same day cannot double an order: Alpaca rejects the duplicate.

Data timing: the prices here are daily bars fetched after the close. A strategy that fills "at the close of the
signal day" is therefore executed at the next close (MOC orders for the next session), one day after the
backtest's fill; strategies that fill at the next open are executed as tested.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"
KEY_ENV, SECRET_ENV, LIVE_ENV = "ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY", "ALPACA_LIVE"
ORDER_PREFIX = "bt-"


class BrokerError(Exception):
    """A broker request failed or was refused. The message never contains credentials."""


# ------------------------------------------------------------------ the REST client

class AlpacaClient:
    """Minimal Alpaca Trading API v2 client (account, positions, orders) over requests.

    `session` is any object with requests.Session's `request(method, url, headers=..., params=..., json=...,
    timeout=...)`; tests pass a fake one. Credentials are kept out of repr(), errors and logs."""

    def __init__(self, key_id: str | None, secret: str | None, live: bool = False, session=None,
                 base_url: str | None = None, timeout: float = 15.0):
        self._key_id = key_id or ""
        self._secret = secret or ""
        self.live = bool(live)
        self.base_url = (base_url or (LIVE_URL if self.live else PAPER_URL)).rstrip("/")
        self.timeout = timeout
        self._session = session

    @classmethod
    def from_env(cls, live: bool = False, session=None) -> "AlpacaClient":
        return cls(os.environ.get(KEY_ENV), os.environ.get(SECRET_ENV), live=live, session=session)

    @property
    def has_keys(self) -> bool:
        return bool(self._key_id and self._secret)

    def __repr__(self) -> str:
        return f"AlpacaClient({'LIVE' if self.live else 'paper'}, {self.base_url}, keys {'set' if self.has_keys else 'missing'})"

    __str__ = __repr__

    @property
    def session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
        return self._session

    def _request(self, method: str, path: str, params: dict | None = None, body: dict | None = None):
        if not self.has_keys:
            raise BrokerError(f"Alpaca keys are not set: export {KEY_ENV} and {SECRET_ENV} (paper-trading keys from "
                              "app.alpaca.markets), or use --dry-run.")
        headers = {"APCA-API-KEY-ID": self._key_id, "APCA-API-SECRET-KEY": self._secret, "Accept": "application/json"}
        url = self.base_url + path
        try:
            r = self.session.request(method, url, headers=headers, params=params, json=body, timeout=self.timeout)
        except Exception as e:  # noqa: BLE001 - network errors; the message is rebuilt so no header can leak
            raise BrokerError(f"Alpaca {method} {path}: {type(e).__name__} (no response)") from None
        status = int(getattr(r, "status_code", 0))
        if status == 204 or not getattr(r, "content", b"x"):
            payload = None
        else:
            try:
                payload = r.json()
            except Exception:  # noqa: BLE001
                payload = None
        if status >= 400:
            msg = payload.get("message") if isinstance(payload, dict) else None
            raise BrokerError(f"Alpaca {method} {path}: HTTP {status}" + (f": {_scrub(str(msg), self)}" if msg else ""))
        return payload

    def account(self) -> dict:
        return self._request("GET", "/v2/account")

    def positions(self) -> list[dict]:
        return self._request("GET", "/v2/positions") or []

    def orders(self, status: str = "open") -> list[dict]:
        return self._request("GET", "/v2/orders", params={"status": status, "limit": 500}) or []

    def submit_order(self, payload: dict) -> dict:
        return self._request("POST", "/v2/orders", body=payload)

    def cancel_order(self, order_id: str) -> None:
        self._request("DELETE", f"/v2/orders/{order_id}")


def _scrub(text: str, client: AlpacaClient) -> str:
    for s in (client._key_id, client._secret):
        if s:
            text = text.replace(s, "***")
    return text


def client_for(live: bool, live_flag_given: bool, session=None) -> AlpacaClient:
    """The client for the CLI's switches: live trading needs ALPACA_LIVE=1 *and* --live, so neither a stray
    environment variable nor a typo sends real orders."""
    env_live = os.environ.get(LIVE_ENV, "").strip() == "1"
    if live_flag_given and not env_live:
        raise BrokerError("--live also needs ALPACA_LIVE=1 in the environment (two switches, so a typo can't send "
                          "real-money orders). Without both, the paper account is used.")
    if env_live and not live_flag_given:
        raise BrokerError("ALPACA_LIVE=1 is set but --live was not given: add --live to trade the live account, or unset "
                          "ALPACA_LIVE to use the paper account.")
    return AlpacaClient.from_env(live=live and env_live, session=session)


# ------------------------------------------------------------------ orders

def alpaca_symbol(ticker: str) -> str | None:
    """Our ticker -> Alpaca's symbol (BRK-B -> BRK.B). None for what Alpaca's stock API can't trade here:
    indexes (^GSPC), crypto pairs (BTC-USD) and simulated series (...SIM)."""
    t = ticker.upper()
    if t.startswith("^") or t.endswith("SIM") or t.endswith("-USD"):
        return None
    return t.replace("-", ".")


def our_ticker(symbol: str) -> str:
    return symbol.upper().replace(".", "-")


def _px(p: float) -> str:
    """A price as Alpaca accepts it: 2 decimals from $1, 4 below."""
    return f"{p:.2f}" if p >= 1 else f"{p:.4f}"


def _qty(q: float, fractional: bool) -> tuple[str, bool]:
    """(quantity string, is it fractional)."""
    q = abs(float(q))
    whole = math.floor(q + 1e-9)
    if not fractional or abs(q - whole) < 1e-6:
        return str(int(whole)), False
    return f"{math.floor(q * 1e6) / 1e6:.6f}".rstrip("0").rstrip("."), True


def _coid(tag: str, as_of: str, symbol: str, role: str) -> str:
    return f"{ORDER_PREFIX}{tag}-{as_of}-{symbol}-{role}"[:128]


def strategy_tag(spec) -> str:
    """A short, stable id of the strategy for client_order_ids (a hash of its interpretation)."""
    try:
        text = spec.summary()
    except Exception:  # noqa: BLE001
        text = repr(spec)
    return hashlib.sha1(text.encode()).hexdigest()[:8]


@dataclass
class Plan:
    as_of: str
    account_value: float
    orders: list[dict] = field(default_factory=list)      # Alpaca order payloads, in submission order
    rows: list[dict] = field(default_factory=list)        # the reconciliation (orders.todays_orders rows)
    notes: list[str] = field(default_factory=list)


def build_orders(rows: list[dict], *, kind: str, as_of: str, tag: str, fill: str = "close",
                 entry_order: str = "market", entry_levels: dict | None = None, valid_bars: int = 1,
                 stop_loss: float | None = None, take_profit: float | None = None,
                 exit_when: dict | None = None, standing: list[dict] | None = None,
                 fractional: bool = True, notes: list[str] | None = None) -> list[dict]:
    """Alpaca order payloads for the reconciliation rows of orders.todays_orders.

    kind: "allocation" or "signal". fill: when the strategy trades ("close", "open", "next_open", "next_close").
    Signal strategies: entry_order / entry_levels {ticker: level} for limit/stop entries, stop_loss / take_profit
    (fractions) for a bracket on them, exit_when {ticker: "at the next open" | ...} to time exits, and standing
    [{ticker, action, order, price, shares, reason, oca}] exit orders (signals.exit_instructions) for positions kept.
    Pure: no network, no data."""
    notes = notes if notes is not None else []
    entry_levels = entry_levels or {}
    exit_when = exit_when or {}
    tif_fill = "opg" if fill in ("open", "next_open") else "cls"
    sells, buys, keep = [], [], {}
    frac_used, skipped = [], []
    legs, used_ids = [], set()
    for r in rows:
        cur, tgt = float(r.get("current_shares") or 0.0), float(r.get("target_shares") or 0.0)
        if r.get("side") in ("BUY", "SELL") and cur * tgt < 0:
            # long -> short (or back): Alpaca takes no order that crosses zero, so close first, then open
            legs.append({**r, "shares": abs(cur), "target_shares": 0.0})
            legs.append({**r, "shares": abs(tgt), "current_shares": 0.0})
        else:
            legs.append(r)
    for r in legs:
        t, side = r["ticker"], r["side"]
        if side not in ("BUY", "SELL") or not r.get("shares"):
            if side == "hold":
                keep[t] = float(r.get("current_shares") or 0.0)
            continue
        sym = alpaca_symbol(t)
        if sym is None:
            skipped.append(t)
            continue
        cur, tgt = float(r.get("current_shares") or 0.0), float(r.get("target_shares") or 0.0)
        # shorts can't be fractional at Alpaca: whole shares whenever a short is opened or held
        frac_ok = fractional and cur >= 0 and tgt >= 0
        qty, is_frac = _qty(r["shares"], frac_ok)
        if qty in ("0", ""):
            continue
        entering = (side == "BUY" and cur >= 0 and tgt > cur) or (side == "SELL" and cur <= 0 and tgt < cur)
        role = "entry" if (kind == "signal" and entering) else ("exit" if kind == "signal" else "rebalance")
        o: dict = {"symbol": sym, "qty": qty, "side": side.lower(), "type": "market",
                   "client_order_id": _coid(tag, as_of, sym, role)}
        if kind == "signal" and role == "exit":
            when = exit_when.get(t, "")
            o["time_in_force"] = "opg" if "open" in when else "cls"
        elif kind == "signal" and entry_order != "market" and t in entry_levels:
            lvl = float(entry_levels[t])
            o["type"] = entry_order
            o["limit_price" if entry_order == "limit" else "stop_price"] = _px(lvl)
            o["time_in_force"] = "day" if int(valid_bars or 1) <= 1 else "gtc"
            if int(valid_bars or 1) > 1:
                notes.append(f"{sym}: the entry order is good-til-cancelled; the backtest keeps it for {valid_bars} bars, "
                             "so cancel it after that.")
            sgn = 1 if side == "BUY" else -1
            if stop_loss or take_profit:
                if stop_loss and take_profit:
                    o["order_class"] = "bracket"
                    o["take_profit"] = {"limit_price": _px(lvl * (1 + sgn * take_profit))}
                    o["stop_loss"] = {"stop_price": _px(lvl * (1 - sgn * stop_loss))}
                else:
                    o["order_class"] = "oto"   # one-triggers-other: the single exit order after the fill
                    if take_profit:
                        o["take_profit"] = {"limit_price": _px(lvl * (1 + sgn * take_profit))}
                    else:
                        o["stop_loss"] = {"stop_price": _px(lvl * (1 - sgn * stop_loss))}
        else:
            o["time_in_force"] = tif_fill
        if is_frac:
            if o["type"] != "market" or o.get("order_class"):
                qty, is_frac = _qty(r["shares"], False)
                o["qty"] = qty
            else:
                o["time_in_force"] = "day"
                frac_used.append(sym)
        n_ = 2
        while o["client_order_id"] in used_ids:      # two legs of one flip: distinct ids
            o["client_order_id"] = _coid(tag, as_of, sym, f"{role}{n_}")
            n_ += 1
        used_ids.add(o["client_order_id"])
        (sells if side == "SELL" else buys).append(o)
        if side == "SELL" and tgt > 0:
            keep[t] = tgt
        elif side == "BUY" and cur > 0:
            keep[t] = cur        # only what is held now can be protected by exit orders today
    out = sells + buys
    # exit orders for the positions kept: only for shares the account holds now (a stop on shares still to be
    # bought would open a short if it triggered first)
    by_ticker: dict[str, list[dict]] = {}
    for s in standing or []:
        if s.get("price") is None:
            notes.append(f"{s['ticker']}: {s.get('reason', 'exit rule')} is checked at the open; no order can express it.")
            continue
        by_ticker.setdefault(s["ticker"], []).append(s)
    for t, lst in by_ticker.items():
        held = keep.get(t, 0.0)
        sym = alpaca_symbol(t)
        if not held or sym is None:
            continue
        side = "sell" if held > 0 else "buy"
        full = [s for s in lst if not str(s.get("reason", "")).startswith("scale out")]
        stop = next((s for s in full if s["order"] == "STOP"), None)
        tgt = next((s for s in full if s["order"] == "LIMIT"), None)
        qty, _ = _qty(min(abs(held), float((stop or tgt or {"shares": abs(held)})["shares"])), False)
        if qty != "0":
            base = {"symbol": sym, "qty": qty, "side": side, "time_in_force": "day"}
            if stop and tgt:
                out.append({**base, "type": "limit", "order_class": "oco", "take_profit": {"limit_price": _px(tgt["price"])},
                            "stop_loss": {"stop_price": _px(stop["price"])}, "client_order_id": _coid(tag, as_of, sym, "oco")})
            elif stop:
                out.append({**base, "type": "stop", "stop_price": _px(stop["price"]), "client_order_id": _coid(tag, as_of, sym, "stop")})
            elif tgt:
                out.append({**base, "type": "limit", "limit_price": _px(tgt["price"]), "client_order_id": _coid(tag, as_of, sym, "target")})
        for n, s in enumerate(x for x in lst if str(x.get("reason", "")).startswith("scale out")):
            q, _ = _qty(min(abs(held), float(s["shares"])), False)
            if q != "0":
                out.append({"symbol": sym, "qty": q, "side": side, "type": "limit", "limit_price": _px(s["price"]),
                            "time_in_force": "day", "client_order_id": _coid(tag, as_of, sym, f"scale{n + 1}")})
    if frac_used:
        notes.append(f"Fractional quantities ({', '.join(frac_used[:8])}{' and more' if len(frac_used) > 8 else ''}) are sent "
                     "as day market orders: Alpaca does not take fractional market-on-close/-open orders. Use --whole-shares "
                     "to keep the close/open timing.")
    if skipped:
        notes.append(f"Not sent (Alpaca's stock API can't trade them here): {', '.join(skipped)}.")
    return out


def plan(spec, account_value: float, positions: dict[str, float] | None = None, fractional: bool = True) -> Plan:
    """The orders that move the account from `positions` ({our ticker: shares}) to the strategy's target today."""
    from . import data, expr, orders, signals
    from .portfolio import Portfolio
    positions = positions or {}
    holdings = "\n".join(f"{t},{q}" for t, q in positions.items())
    rec = orders.todays_orders(spec, account_value, holdings, whole_shares=not fractional)
    notes = list(rec["notes"])
    tag = strategy_tag(spec)
    if isinstance(spec, Portfolio):
        payloads = build_orders(rec["orders"], kind="allocation", as_of=rec["as_of"], tag=tag,
                                fill=spec.fill, fractional=fractional, notes=notes)
    else:
        sc = signals.scan(spec)
        exit_when = {e["ticker"]: e["when"] for e in sc.get("exit_signals", []) if not e.get("done")}
        levels = {}
        if spec.entry_order != "market" and spec.entry_level:
            for r in rec["orders"]:
                try:
                    df = data.load(r["ticker"])
                    levels[r["ticker"]] = float(expr.evaluate_value(spec.entry_level, expr.Namespace(df, ticker=r["ticker"])).iloc[-1])
                except Exception:  # noqa: BLE001 - no level: the entry goes as a market order
                    pass
        if spec.entry_fill == "close":
            notes.append("The strategy fills at the close of the signal day; with end-of-day data these orders are for the "
                         "next close (market-on-close), one day after the backtest's fill.")
        payloads = build_orders(rec["orders"], kind="signal", as_of=rec["as_of"], tag=tag, fill=spec.entry_fill,
                                entry_order=spec.entry_order, entry_levels=levels, valid_bars=spec.order_valid_bars,
                                stop_loss=spec.stop_loss, take_profit=spec.take_profit, exit_when=exit_when,
                                standing=sc.get("exit_orders", []), fractional=fractional, notes=notes)
    return Plan(as_of=rec["as_of"], account_value=account_value, orders=payloads, rows=rec["orders"],
                notes=list(dict.fromkeys(notes)))


def broker_positions(client: AlpacaClient) -> dict[str, float]:
    """{our ticker: signed shares} held at the broker."""
    out: dict[str, float] = {}
    for p in client.positions():
        q = float(p.get("qty") or 0.0)
        if str(p.get("side", "long")).lower() == "short" and q > 0:
            q = -q
        out[our_ticker(p["symbol"])] = out.get(our_ticker(p["symbol"]), 0.0) + q
    return out


def account_equity(client: AlpacaClient) -> float:
    a = client.account() or {}
    for k in ("equity", "portfolio_value"):
        try:
            v = float(a.get(k))
        except (TypeError, ValueError):
            continue
        if v > 0:
            return v
    raise BrokerError("The Alpaca account reports no equity; fund the (paper) account first.")


def submit(client: AlpacaClient, p: Plan, dry_run: bool = True, cancel_stale: bool = True, log=print) -> list[dict]:
    """Send the plan's orders (or, with dry_run, only print them). Before sending, open orders this tool placed
    earlier (client_order_id starting with 'bt-') are cancelled: yesterday's stop/target levels are replaced by
    today's. Returns one result per order: {"client_order_id", "status", "id" | "error"}."""
    results = []
    if dry_run:
        for o in p.orders:
            log("DRY RUN (not sent): " + json.dumps(o, sort_keys=True))
            results.append({"client_order_id": o.get("client_order_id"), "status": "dry-run"})
        return results
    if cancel_stale:
        for o in client.orders("open"):
            if str(o.get("client_order_id", "")).startswith(ORDER_PREFIX):
                try:
                    client.cancel_order(o["id"])
                    log(f"cancelled earlier order {o.get('client_order_id')}")
                except BrokerError as e:
                    log(f"could not cancel {o.get('client_order_id')}: {e}")
    for o in p.orders:
        try:
            r = client.submit_order(o) or {}
            results.append({"client_order_id": o.get("client_order_id"), "status": r.get("status", "accepted"), "id": r.get("id")})
            log(f"sent {o['side']} {o['qty']} {o['symbol']} {o['type']} {o.get('time_in_force')}: {r.get('status', 'accepted')}")
        except BrokerError as e:
            results.append({"client_order_id": o.get("client_order_id"), "status": "error", "error": str(e)})
            log(f"FAILED {o['side']} {o['qty']} {o['symbol']}: {e}")
    return results
