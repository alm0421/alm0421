"""Round 7 (QuantConnect review): lookahead probe for Python-function rules, next-open fills of series without
opens in portfolios, broker leverage limits, parser phrases, portfolio broker costs, the Alpaca bridge, and the
signal engine's speed-ups (which must not change any number)."""
import json

import numpy as np
import pandas as pd
import pytest

from backtester import broker, costs, data, engine, expr, parser, portfolio as pf
from backtester.strategy import Strategy

HAVE = {"QQQ", "SPY", "TLT", "SPYSIM", "TLTSIM"} <= set(data.available_tickers())
needs_data = pytest.mark.skipif(not HAVE, reason="price data not downloaded")


def frame(closes, start="2015-01-01", volume=1e6, open_ok=None):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    opens = np.r_[closes[0], closes[:-1]] * 1.001
    df = pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * 1.004,
                       "low": np.minimum(opens, closes) * 0.996, "close": closes}, index=idx)
    df["volume"] = volume
    df["quote_close"] = df["close"]
    df["adj_close"] = df["close"]
    df["dividend"] = 0.0
    df["split"] = 0.0
    if open_ok is not None:
        df["open_ok"] = open_ok
    return df


def walk(seed, n=700, drift=0.0003, vol=0.015):
    r = np.random.default_rng(seed).normal(drift, vol, n)
    r[0] = 0.0
    return 100 * np.cumprod(1 + r)


@pytest.fixture
def fake(monkeypatch):
    frames = {}

    def load(t):
        return frames[data.canonical(t)]

    monkeypatch.setattr(data, "load", load)
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): load(t) for t in ts})
    return frames


# ------------------------------------------------------------ 1. Python-function rules: lookahead probe

def future(df, ns):
    return df.close.shift(-1) > df.close


def centred(df, ns):
    return df.close > df.close.rolling(5, center=True).mean()


def causal(df, ns):
    return (df.close < df.close.rolling(10).min().shift(1)) & (ns["rsi"](2) < 30)


@pytest.mark.parametrize("fn", [future, centred, lambda df, ns: df.close < df.close.mean()])
def test_function_entry_that_reads_later_rows_is_refused(fake, fn):
    fake["X"] = frame(walk(1))
    with pytest.raises(ValueError, match="uses future data: its result on .* changes when the data after"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=fn, hold_bars=1))


def test_causal_function_rules_run(fake):
    fake["X"] = frame(walk(2))
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry=causal, hold_bars=2,
                            exit_when=lambda df, ns: ns["rsi"](2) > 80, rank_by=lambda df, ns: df.close.pct_change(5)))
    assert len(r.trades) > 0
    assert r.equity.iloc[-1] == pytest.approx(10_000 + r.trades.pnl.sum() + r.interest)


def test_function_exit_ranking_and_level_are_probed_too(fake):
    fake["X"] = frame(walk(3))
    fake["Y"] = frame(walk(4))
    with pytest.raises(ValueError, match="exit function"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=causal, exit_when=centred))
    with pytest.raises(ValueError, match="ranking function"):
        engine.run(Strategy(cash_rate=None, universe=["X", "Y"], entry=causal, hold_bars=1,
                            rank_by=lambda df, ns: df.close.shift(-3) / df.close))
    with pytest.raises(ValueError, match="order level function"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=causal, hold_bars=1, entry_order="limit",
                            entry_level=lambda df, ns: df.low.shift(-1)))


def test_open_safe_function_is_also_perturbed_at_the_open(fake):
    fake["X"] = frame(walk(5))

    def peeks(df, ns):          # claims to be known at the open but reads the close
        return df.close > df.open
    peeks.open_safe = True

    def gap_down(df, ns):
        return df.open > df.close.shift(1) * 1.0005   # the synthetic opens gap up 0.1%
    gap_down.open_safe = True
    with pytest.raises(ValueError, match="Lookahead"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=peeks, entry_fill="open", hold_bars=0))
    r = engine.run(Strategy(cash_rate=None, universe=["X"], entry=gap_down, entry_fill="open", hold_bars=0))
    assert len(r.trades)


def test_probe_is_cached_per_function_and_data(fake):
    fake["X"] = frame(walk(6))
    calls = []

    def counted(df, ns):
        calls.append(len(df))
        return df.close > df.close.shift(1)
    engine.run(Strategy(cash_rate=None, universe=["X"], entry=counted, hold_bars=1))
    first = len(calls)
    assert first > 5 and min(calls) < len(fake["X"])     # probed on cut data
    engine.run(Strategy(cash_rate=None, universe=["X"], entry=counted, hold_bars=1))
    assert len(calls) == first + 1                        # second run: only the real evaluation


@needs_data
def test_reviewer_repro_shift_minus_one_on_qqq_is_refused():
    from backtester import api
    with pytest.raises(ValueError, match="uses future data"):
        api.backtest(Strategy(universe=["QQQ"], entry=lambda df, ns: df.close.shift(-1) > df.close, hold_bars=1))


# ------------------------------------------------------------ 2. portfolio next-open fills need real opens

@needs_data
def test_next_open_portfolio_of_sim_series_is_refused():
    p = parser.parse("hold 60% SPYSIM and 40% TLTSIM, rebalance monthly, trade at the next open, since 1980")
    with pytest.raises(ValueError, match="no real opening prices"):
        pf.run(p)
    p2 = parser.parse("hold 60% SPYSIM and 40% TLTSIM, rebalance monthly, since 1980")
    pf.run(p2)                     # at the close: fine


def test_next_open_day_without_a_quoted_open_fills_at_that_close(fake):
    n = 80
    ok = np.ones(n, bool)
    ok[20:30] = False              # old data: no quoted opens for two weeks
    fake["A"] = frame(walk(7, n), open_ok=ok)
    fake["B"] = frame(walk(8, n))
    p = pf.Portfolio(tree={"weights": "specified", "w": [0.5, 0.5], "children": [{"asset": "A"}, {"asset": "B"}]},
                     rebalance="daily", fill="next_open", cash_rate=None)
    r = pf.run(p)
    od = r.orders
    d = fake["A"].index[25]
    a_fill = od[(pd.to_datetime(od.date) == d) & (od.ticker == "A")]
    b_fill = od[(pd.to_datetime(od.date) == d) & (od.ticker == "B")]
    assert len(a_fill) and a_fill.price.iloc[0] == pytest.approx(fake["A"].close[d])
    assert len(b_fill) and b_fill.price.iloc[0] == pytest.approx(fake["B"].open[d])
    assert any(n_.startswith("Opens: A") for n_ in p.notes)
    fake["C"] = frame(walk(9, n), open_ok=np.zeros(n, bool))
    p3 = pf.Portfolio(tree={"asset": "C"}, rebalance="monthly", fill="next_open", cash_rate=None)
    with pytest.raises(ValueError, match="C has no real opening prices"):
        pf.run(p3)


# ------------------------------------------------------------ 3. leverage a broker would allow (signals)

def test_signal_leverage_follows_reg_t_and_portfolio_margin():
    base = dict(universe=["X"], entry="True", hold_bars=1)
    Strategy(**base, leverage=2).validate()
    with pytest.raises(ValueError, match="Regulation T"):
        Strategy(**base, leverage=3).validate()
    with pytest.raises(ValueError, match="Regulation T"):
        Strategy(**base, position_size=3.0).validate()          # raised to 3x by the size: still refused
    with pytest.raises(ValueError, match="not below the initial margin of 4x"):
        Strategy(**base, leverage=4, margin_account="portfolio").validate()   # 25% = 25%: the reviewer's case
    Strategy(**base, leverage=4, margin_account="portfolio", maintenance_margin=0.15).validate()
    with pytest.raises(ValueError, match="portfolio-margin account allows"):
        Strategy(**base, leverage=5, margin_account="portfolio", maintenance_margin=0.1).validate()
    with pytest.raises(ValueError, match="margin_account"):
        Strategy(**base, margin_account="cash").validate()
    with pytest.raises(ValueError, match="not below the initial margin"):
        Strategy(**base, leverage=2, maintenance_margin=0.5).validate()


def test_parser_portfolio_margin_phrase():
    s = parser.parse("buy QQQ when RSI(2) is below 10, hold 3 days, 4x leverage, with portfolio margin and a 15% maintenance margin")
    assert (s.leverage, s.margin_account, s.maintenance_margin) == (4, "portfolio", 0.15)
    with pytest.raises((ValueError, parser.ParseError), match="Regulation T"):
        s2 = parser.parse("buy QQQ when RSI(2) is below 10, hold 3 days, 3x leverage")
        s2.validate()


# ------------------------------------------------------------ 4. parser phrases

@needs_data
def test_margin_rate_phrases():
    p = parser.parse("hold QQQ with 2x leverage and a 1% margin rate")
    assert (p.leverage, p.margin_rate) == (2, 0.01)
    s = parser.parse("buy QQQ when RSI(2) is below 10, hold 3 days, 2x leverage, margin rate 1%")
    assert (s.leverage, s.margin_rate) == (2, 0.01)
    s = parser.parse("buy QQQ when RSI(2) is below 10, hold 3 days, 2x leverage and a 1.5% margin rate")
    assert s.margin_rate == 0.015
    assert "T-bill rate + 1.50%" in s.summary()


@needs_data
@pytest.mark.parametrize("text,asc", [
    ("buy Nasdaq 100 stocks when RSI(2) is below 10, hold 3 days, max 5 positions, rank by market cap", False),
    ("buy Nasdaq 100 stocks when RSI(2) is below 10, hold 3 days, max 5 positions, prefer the largest", False),
    ("buy Nasdaq 100 stocks when RSI(2) is below 10, hold 3 days, max 5 positions, prefer the smallest market cap", True),
])
def test_rank_by_market_cap(text, asc):
    s = parser.parse(text)
    assert s.rank_by == "market_cap" and s.rank_ascending is asc


@needs_data
def test_largest_declines_is_still_a_return_ranking():
    s = parser.parse("buy Nasdaq 100 stocks when RSI(2) is below 10, hold 3 days, max 5 positions, prefer the largest declines")
    assert s.rank_by == "change" and s.rank_ascending


@needs_data
def test_portfolio_sentences_take_broker_costs():
    p = parser.parse("hold 60% SPY and 40% TLT, rebalance monthly, IBKR commissions, volume-based slippage")
    assert (p.commission_model, p.slippage_model) == ("ibkr_fixed", "volume")
    p = parser.parse("hold 60% SPY and 40% TLT, rebalance monthly, with IBKR tiered commissions")
    assert p.commission_model == "ibkr_tiered"
    assert "IBKR tiered" in p.summary()


# ------------------------------------------------------------ 5. portfolio broker costs

def _identity(r):
    a = r.extras["attribution"]
    lhs = a["pnl"].sum() + r.interest - r.extras["fees"]
    rhs = r.equity.iloc[-1] - r.equity.iloc[0] - r.extras["flows"].sum()
    return lhs == pytest.approx(rhs, abs=1e-6)


def test_portfolio_ibkr_commissions_and_volume_slippage(fake):
    fake["A"] = frame(walk(10, 300), volume=5_000)
    fake["B"] = frame(walk(11, 300), volume=2e6)
    tree = {"weights": "specified", "w": [0.6, 0.4], "children": [{"asset": "A"}, {"asset": "B"}]}
    plain = pf.run(pf.Portfolio(tree=tree, rebalance="monthly", cash_rate=None, capital=1e6))
    p = pf.Portfolio(tree=tree, rebalance="monthly", cash_rate=None, capital=1e6, commission_model="ibkr_fixed",
                     slippage_model="volume", contribution=10_000)
    r = pf.run(p)
    od = r.orders
    for _, o in od.iterrows():
        assert o.commission == pytest.approx(costs.broker_commission("ibkr_fixed", o.shares, o.shares * o.price))
    first_a = od[(od.ticker == "A")].iloc[0]
    px = fake["A"].close[pd.Timestamp(first_a.date)]
    adv = fake["A"].volume.rolling(20, min_periods=1).mean().shift(1)[pd.Timestamp(first_a.date)]
    if np.isnan(adv):
        assert first_a.price == pytest.approx(px * (1 + 1e-4))       # no volume history yet: half the spread
    else:
        assert first_a.price > px * (1 + 1e-4)
    assert r.equity.iloc[-1] < plain.equity.iloc[-1] + r.extras["flows"].sum()
    assert _identity(r)
    assert "IBKR fixed" in p.summary() and "volume slippage" in p.summary()
    with pytest.raises(ValueError, match="commission_model"):
        pf.Portfolio(tree=tree, commission_model="robinhood").validate()


def test_order_commission_matches_the_engines_formula():
    for sh, val in ((10, 1_000), (1_000, 50_000), (3, 12.0)):
        for model in (None, "ibkr_fixed", "ibkr_tiered"):
            want = 1.0 + 0.002 * sh + 0.001 * abs(val) + costs.broker_commission(model, sh, val)
            got = costs.order_commission(sh, val, per_order=1.0, per_share=0.002, pct=0.001, model=model)
            assert got == pytest.approx(want, rel=1e-15)
    assert costs.volume_slippage(0.0005, 10_000, 1e6, 2, 100) == pytest.approx(0.0005 + (1 + 10) / 1e4)
    assert costs.volume_slippage(0.0, 10, float("nan"), 2, 100) == pytest.approx(1e-4)


# ------------------------------------------------------------ 6. Alpaca bridge (mocked HTTP, never the network)

class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload
        self.content = b"" if payload is None else json.dumps(payload).encode()

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, routes=None):
        self.calls, self.routes = [], routes or {}

    def request(self, method, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "headers": headers, "params": params, "json": json})
        for (m, suffix), resp in self.routes.items():
            if m == method and url.endswith(suffix):
                return resp(json) if callable(resp) else resp
        return FakeResponse(200, {"id": "ord-1", "status": "accepted"})


def test_paper_by_default_and_live_needs_both_switches(monkeypatch):
    monkeypatch.setenv(broker.KEY_ENV, "KEYID123")
    monkeypatch.setenv(broker.SECRET_ENV, "SECRETXYZ")
    monkeypatch.delenv(broker.LIVE_ENV, raising=False)
    c = broker.client_for(False, False, session=FakeSession())
    assert c.base_url == broker.PAPER_URL and not c.live
    with pytest.raises(broker.BrokerError, match="ALPACA_LIVE=1"):
        broker.client_for(True, True)
    monkeypatch.setenv(broker.LIVE_ENV, "1")
    with pytest.raises(broker.BrokerError, match="--live"):
        broker.client_for(False, False)
    c = broker.client_for(True, True, session=FakeSession())
    assert c.base_url == broker.LIVE_URL and c.live
    assert "KEYID123" not in repr(c) and "SECRETXYZ" not in repr(c)


def test_requests_carry_the_keys_in_headers_and_errors_never_show_them():
    s = FakeSession({("GET", "/v2/account"): FakeResponse(403, {"message": "forbidden for key KEYID123"})})
    c = broker.AlpacaClient("KEYID123", "SECRETXYZ", session=s)
    with pytest.raises(broker.BrokerError) as e:
        c.account()
    assert "KEYID123" not in str(e.value) and "SECRETXYZ" not in str(e.value) and "403" in str(e.value)
    assert s.calls[0]["headers"]["APCA-API-KEY-ID"] == "KEYID123"
    assert s.calls[0]["url"] == broker.PAPER_URL + "/v2/account"

    class Boom:
        def request(self, *a, **k):
            raise ConnectionError("proxy said: APCA-API-SECRET-KEY: SECRETXYZ")
    with pytest.raises(broker.BrokerError) as e:
        broker.AlpacaClient("KEYID123", "SECRETXYZ", session=Boom()).positions()
    assert "SECRETXYZ" not in str(e.value)
    with pytest.raises(broker.BrokerError, match="keys are not set"):
        broker.AlpacaClient(None, None, session=FakeSession()).account()


def test_positions_and_equity_are_read_from_the_account():
    s = FakeSession({("GET", "/v2/positions"): FakeResponse(200, [{"symbol": "BRK.B", "qty": "3", "side": "long"},
                                                                   {"symbol": "TSLA", "qty": "2", "side": "short"}]),
                     ("GET", "/v2/account"): FakeResponse(200, {"equity": "25000.5"})})
    c = broker.AlpacaClient("k", "s", session=s)
    assert broker.broker_positions(c) == {"BRK-B": 3.0, "TSLA": -2.0}
    assert broker.account_equity(c) == 25000.5


def _rows(*rs):
    return [dict(zip(("ticker", "side", "shares", "current_shares", "target_shares"), r)) for r in rs]


def test_allocation_orders_rebalance_to_targets_sells_first():
    notes = []
    rows = _rows(("SPY", "BUY", 12.3456, 10.0, 22.3456), ("TLT", "SELL", 5.0, 20.0, 15.0), ("GLD", "SELL", 3.0, 3.0, 0.0),
                 ("^GSPC", "BUY", 1.0, 0.0, 1.0), ("IEF", "hold", 0, 7.0, 7.0))
    o = broker.build_orders(rows, kind="allocation", as_of="2026-09-25", tag="abc", fill="close", notes=notes)
    assert [x["symbol"] for x in o] == ["TLT", "GLD", "SPY"]
    assert o[0]["time_in_force"] == "cls" and o[0]["qty"] == "5"                   # whole: market-on-close
    assert o[2]["qty"] == "12.3456" and o[2]["time_in_force"] == "day"              # fractional: day market
    assert any("Fractional" in n for n in notes) and any("^GSPC" in n for n in notes)
    o2 = broker.build_orders(rows, kind="allocation", as_of="2026-09-25", tag="abc", fill="next_open", fractional=False)
    assert all(x["time_in_force"] == "opg" for x in o2) and o2[-1]["qty"] == "12"
    assert o2[0]["client_order_id"] == "bt-abc-2026-09-25-TLT-rebalance"
    again = broker.build_orders(rows, kind="allocation", as_of="2026-09-25", tag="abc", fill="next_open", fractional=False)
    assert [x["client_order_id"] for x in again] == [x["client_order_id"] for x in o2]   # idempotent ids


def test_signal_orders_brackets_exits_and_standing_oco():
    rows = _rows(("AAPL", "BUY", 10, 0, 10), ("MSFT", "SELL", 4, 4, 0), ("NVDA", "hold", 0, 6, 6), ("AMD", "hold", 0, 5, 5))
    standing = [{"ticker": "NVDA", "order": "STOP", "price": 95.123, "shares": 6, "reason": "stop loss", "oca": "NVDA-exit"},
                {"ticker": "NVDA", "order": "LIMIT", "price": 120.0, "shares": 6, "reason": "take profit", "oca": "NVDA-exit"},
                {"ticker": "NVDA", "order": "LIMIT", "price": 110.0, "shares": 3, "reason": "scale out 50% at +10.0%"},
                {"ticker": "AMD", "order": "STOP", "price": 0.5, "shares": 9, "reason": "stop loss", "oca": None},
                {"ticker": "AAPL", "order": "STOP", "price": 90.0, "shares": 10, "reason": "stop loss", "oca": None}]
    o = broker.build_orders(rows, kind="signal", as_of="2026-09-25", tag="t", fill="next_open", entry_order="limit",
                            entry_levels={"AAPL": 100.0}, stop_loss=0.05, take_profit=0.08,
                            exit_when={"MSFT": "at the next open"}, standing=standing, fractional=False)
    by = {(x["symbol"], x["client_order_id"].rsplit("-", 1)[-1]): x for x in o}
    assert by[("MSFT", "exit")]["time_in_force"] == "opg" and by[("MSFT", "exit")]["side"] == "sell"
    e = by[("AAPL", "entry")]
    assert (e["type"], e["limit_price"], e["order_class"], e["time_in_force"]) == ("limit", "100.00", "bracket", "day")
    assert e["take_profit"] == {"limit_price": "108.00"} and e["stop_loss"] == {"stop_price": "95.00"}
    oco = by[("NVDA", "oco")]
    assert oco["order_class"] == "oco" and oco["stop_loss"] == {"stop_price": "95.12"} and oco["qty"] == "6"
    assert by[("NVDA", "scale1")]["limit_price"] == "110.00" and by[("NVDA", "scale1")]["qty"] == "3"
    amd = by[("AMD", "stop")]
    assert amd["stop_price"] == "0.5000" and amd["qty"] == "5"          # never more than the account holds
    assert not any(x["symbol"] == "AAPL" and x["client_order_id"].endswith("stop") for x in o)   # not held yet
    o2 = broker.build_orders(_rows(("AAPL", "BUY", 10, 0, 10)), kind="signal", as_of="d", tag="t", fill="close")
    assert o2[0]["time_in_force"] == "cls" and o2[0]["type"] == "market"


def test_long_to_short_flip_is_two_orders():
    o = broker.build_orders(_rows(("TSLA", "SELL", 15, 5, -10)), kind="allocation", as_of="d", tag="t", fill="close")
    assert [(x["side"], x["qty"]) for x in o] == [("sell", "5"), ("sell", "10")]
    assert len({x["client_order_id"] for x in o}) == 2


def test_dry_run_sends_nothing_and_submit_cancels_our_stale_orders():
    p = broker.Plan(as_of="2026-09-25", account_value=1e4,
                    orders=[{"symbol": "SPY", "qty": "1", "side": "buy", "type": "market", "time_in_force": "cls",
                             "client_order_id": "bt-x-2026-09-25-SPY-rebalance"}])
    s = FakeSession()
    c = broker.AlpacaClient("k", "s", session=s)
    out = []
    res = broker.submit(c, p, dry_run=True, log=out.append)
    assert s.calls == [] and res[0]["status"] == "dry-run" and "DRY RUN" in out[0]
    s = FakeSession({("GET", "/v2/orders"): FakeResponse(200, [{"id": "a1", "client_order_id": "bt-old"},
                                                               {"id": "m1", "client_order_id": "manual"}]),
                     ("DELETE", "/v2/orders/a1"): FakeResponse(204, None),
                     ("POST", "/v2/orders"): lambda body: FakeResponse(200, {"id": "n1", "status": "new"})})
    c = broker.AlpacaClient("k", "s", session=s)
    res = broker.submit(c, p, dry_run=False, log=lambda *_: None)
    assert [(x["method"], x["url"].rsplit("/v2", 1)[1]) for x in s.calls] == [
        ("GET", "/orders"), ("DELETE", "/orders/a1"), ("POST", "/orders")]
    assert s.calls[-1]["json"]["client_order_id"] == "bt-x-2026-09-25-SPY-rebalance"
    assert res == [{"client_order_id": "bt-x-2026-09-25-SPY-rebalance", "status": "new", "id": "n1"}]
    s = FakeSession({("GET", "/v2/orders"): FakeResponse(200, []),
                     ("POST", "/v2/orders"): FakeResponse(422, {"message": "client_order_id must be unique"})})
    res = broker.submit(broker.AlpacaClient("k", "s", session=s), p, dry_run=False, log=lambda *_: None)
    assert res[0]["status"] == "error" and "unique" in res[0]["error"]


@needs_data
def test_plan_for_a_portfolio_reconciles_broker_positions():
    p = parser.parse("hold 60% SPY and 40% TLT, rebalance monthly")
    pl = broker.plan(p, 100_000, {"SPY": 10.0, "GLD": 5.0}, fractional=False)
    sym = {o["symbol"]: o for o in pl.orders}
    assert sym["GLD"]["side"] == "sell" and sym["GLD"]["qty"] == "5"
    spy = float(data.load("SPY").close.iloc[-1])
    assert int(sym["SPY"]["qty"]) == int(60_000 / spy) - 10 and sym["SPY"]["time_in_force"] == "cls"
    assert [o["side"] for o in pl.orders].index("sell") < [o["side"] for o in pl.orders].index("buy")


def test_trade_cli_dry_run_without_keys_prints_payloads(monkeypatch, capsys, fake):
    from backtester import __main__ as cli
    monkeypatch.delenv(broker.KEY_ENV, raising=False)
    monkeypatch.delenv(broker.SECRET_ENV, raising=False)
    monkeypatch.delenv(broker.LIVE_ENV, raising=False)
    fake_plan = broker.Plan(as_of="2026-09-25", account_value=5e3,
                            orders=[{"symbol": "X", "qty": "2", "side": "buy", "type": "market", "time_in_force": "cls",
                                     "client_order_id": "bt-1"}])
    monkeypatch.setattr(broker, "plan", lambda spec, value, positions, fractional=True: fake_plan)
    fake["X"] = frame(walk(12, 300))
    spec = pf.Portfolio(tree={"asset": "X"})
    monkeypatch.setattr(cli.parser, "parse", lambda text: spec)
    assert cli.main(["trade", "hold X", "--dry-run", "--account-value", "5000"]) == 0
    out = capsys.readouterr().out
    assert "paper" in out and "DRY RUN (not sent)" in out and '"symbol": "X"' in out
    assert cli.main(["trade", "hold X"]) == 2          # no keys, not a dry run: refused


# ------------------------------------------------------------ 7. speed-ups change nothing

def test_fast_rsi_and_true_range_equal_the_pandas_formulas():
    rng = np.random.default_rng(0)
    c = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 500)), index=pd.bdate_range("2020-01-01", periods=500))
    c.iloc[[5, 6, 200]] = np.nan
    c.iloc[300:303] = c.iloc[299]                     # flat: RSI 100 / NaN handling
    d = c.diff()
    up, dn = expr.wilder(d.clip(lower=0), 2), expr.wilder(-d.clip(upper=0), 2)
    old = (100 - 100 / (1 + up / dn)).where(dn != 0, 100.0).where(d.notna())
    pd.testing.assert_series_equal(expr.rsi_wilder(c, 2), old, check_names=False)
    df = pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1e6})
    ns = expr.Namespace(df)
    hi, lo = df["high"], df["low"]
    tr_old = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    pd.testing.assert_series_equal(ns["atr"](14), expr.wilder(tr_old, 14), check_names=False)
    a = ns["rsi"](2)
    a.iloc[:] = 0                                     # the memo hands out copies
    pd.testing.assert_series_equal(ns["rsi"](2), old, check_names=False)
    pd.testing.assert_series_equal(ns["rsi"](ns["close"], 2), old, check_names=False)


def test_union_index_and_namespace_cache():
    a = pd.DatetimeIndex(["2020-01-02", "2020-01-06"], name="date")
    b = pd.DatetimeIndex(["2020-01-03", "2020-01-06", "2020-01-07"], name="date")
    assert engine._union_index([a, b]).equals(a.union(b)) and engine._union_index([a, b]).name == "date"
    df = frame(walk(13, 50))
    assert engine._namespace(df, "Z") is engine._namespace(df, "Z")
    assert engine._namespace(df.copy(), "Z") is not engine._namespace(df, "Z")


def test_coverage_fast_path_equals_the_month_by_month_check():
    if data.membership() is None:
        pytest.skip("no membership data")
    data.coverage.cache_clear()
    fast = data.coverage("2010-01-01", "2012-12-31")
    slow_rows = []
    m = data.membership()
    m = m[(m.index >= "2010-01-01") & (m.index <= "2012-12-31")]
    have = set(data.available_tickers())
    for y, g in m.groupby(m.index.year):
        cols = [c for c in g.columns if c in have]
        ok = 0
        for d, row in g[cols].iterrows():
            for c in cols:
                if row[c]:
                    q = data.quality(c)
                    sl = q[(q.index >= d) & (q.index < d + pd.offsets.MonthBegin(1))] if not q.empty else q
                    ok += bool(len(sl)) and sl.mean() >= 0.5
        slow_rows.append(round(ok / len(g), 1))
    assert list(fast["with_data"]) == slow_rows
