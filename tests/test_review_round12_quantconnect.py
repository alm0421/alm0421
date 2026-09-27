"""Round 12 (QuantConnect reviewer): what a sealed Python rule / portfolio function carries in is measured as a whole.

Round 11 refused only dicts / Series / arrays and long runs of numbers, so later outcomes still reached a sealed rule
through a plain list, a tuple, a string, bytes, a big integer used as a bitmask, a default argument, a function
attribute, a closure cell, a user module's global, os.environ or a portfolio function's global string (209% and 167%
CAGR). Now every value a function carries by value counts against one small budget (sandbox.MAX_ITEMS values,
MAX_TEXT-character strings, MAX_INT_BITS-bit integers, code size), the sealed process starts with an empty
environment, and a module of the user's own is measured when the sealed process imports it.
"""
from __future__ import annotations

import functools
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr, sandbox
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy


def walk(seed, n=400, start="2012-01-02"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    o = c * np.exp(rng.normal(0, 0.003, n))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.004, "low": np.minimum(o, c) * 0.996, "close": c,
                         "adj_close": c, "volume": 1e6, "dividend": 0.0, "split": 1.0, "open_ok": True,
                         "quote_close": c}, index=idx)


@pytest.fixture
def fake(monkeypatch):
    frames = {}
    monkeypatch.setattr(data, "load", lambda t: frames[data.canonical(t)])
    monkeypatch.setattr(data, "load_many", lambda ts: {data.canonical(t): frames[data.canonical(t)] for t in ts})
    expr._STREAMED.clear()
    expr._CALLABLE_PROBES.clear()
    return frames


DF = walk(12)
UP = (DF.close.shift(-1) > DF.close).fillna(False).to_numpy()           # tomorrow closes higher: the leak
FLAGS = UP.tolist()
BITS = sum(1 << i for i, v in enumerate(FLAGS) if v)
TEXT = "".join("1" if v else "0" for v in FLAGS)
PAIRS = "|".join(d.strftime("%Y%m%d") + ("1" if v else "0") for d, v in zip(DF.index, FLAGS))


def _s(d, vals):
    return pd.Series([bool(v) for v in vals[: len(d)]], index=d.index)


def f_list(d, ns):
    return _s(d, FLAGS)


def f_tuple(d, ns):
    return _s(d, tuple(FLAGS_T))


FLAGS_T = tuple(FLAGS)


def f_str(d, ns):
    return _s(d, [c == "1" for c in TEXT])


def f_bytes(d, ns):
    return _s(d, list(TEXT_B))


TEXT_B = bytes(FLAGS)


def f_bigint(d, ns):
    return _s(d, [(BITS >> i) & 1 for i in range(len(d))])


def f_default(d, ns, _t=TEXT):
    return _s(d, [c == "1" for c in _t])


def f_kwdefault(d, ns, *, _t=TEXT):
    return _s(d, [c == "1" for c in _t])


def f_attr(d, ns):
    return _s(d, [c == "1" for c in f_attr.t])


f_attr.t = TEXT


def _closure():
    flags = list(FLAGS)

    def f_cell(d, ns):
        return _s(d, flags)
    return f_cell


GET = FLAGS.__getitem__               # a bound builtin method carries its list


def f_bound(d, ns):
    return _s(d, [GET(i) for i in range(len(d))])


PART = functools.partial(lambda flags, n: flags[:n], FLAGS)


def f_partial(d, ns):
    return _s(d, PART(len(d)))


class Holder:
    flags = TEXT                      # a class attribute of the user's own class


def f_class(d, ns):
    return _s(d, [c == "1" for c in Holder.flags])


class Box:
    def __init__(self, v):
        self.v = v


BOX = Box(FLAGS)


def f_object(d, ns):
    return _s(d, BOX.v)


def f_nested(d, ns):
    return f_list(d, ns)              # the leak is in a function it calls


def f_deep(d, ns):
    return _s(d, DEEP[0][0][0][0][0][0][0][0][0][0])


DEEP = [[[[[[[[[[FLAGS]]]]]]]]]]      # nested deeper than round 11's walk looked


def f_many(d, ns):
    return _s(d, [c == "1" for s in CHUNKS for c in s[8::10]])


CHUNKS = [PAIRS[i:i + 200] for i in range(0, len(PAIRS), 200)]   # each string short, many of them


def f_literal(d, ns):
    return _s(d, LITERAL(d))


# a list literal written into generated code: a constant of 400 values
LITERAL = eval("lambda d: [" + ",".join("1" if v else "0" for v in FLAGS) + "]")

CHANNELS = [f_list, f_tuple, f_str, f_bytes, f_bigint, f_default, f_kwdefault, f_attr, _closure(), f_bound,
            f_partial, f_class, f_object, f_nested, f_deep, f_many, f_literal]


def _fresh(fn):
    """A copy of fn (so the vectorized flag set on it does not stick)."""
    g = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)
    g.__kwdefaults__ = fn.__kwdefaults__
    g.__dict__.update(fn.__dict__)
    return g


@pytest.mark.parametrize("vectorized", [False, True], ids=["streamed", "vectorized"])
@pytest.mark.parametrize("fn", CHANNELS, ids=[f.__name__ + ("_" + str(i)) for i, f in enumerate(CHANNELS)])
def test_every_capture_channel_is_refused(fake, fn, vectorized):
    fake["X"] = DF
    if vectorized:
        fn = _fresh(fn)
        fn.vectorized_causal = True
    with pytest.raises(sandbox.LeakError, match="future data") as e:
        expr.evaluate(fn, expr.Namespace(DF, ticker="X"))
    assert "captured outside the run" in str(e.value)
    with pytest.raises(ValueError, match="future data"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=fn, hold_bars=1))


def test_the_message_names_the_variable():
    cases = {f_list: "global 'FLAGS' is a list of 400 values", f_bigint: "global 'BITS' is an integer of",
             f_default: "a default argument of f_default() is a str of 400 characters",
             f_kwdefault: "default argument '_t' of f_kwdefault()", f_attr: "attribute 't' of f_attr()",
             _closure(): "variable 'flags' captured by f_cell()", f_class: "class attribute Holder.flags",
             f_literal: "a literal in the code of"}
    for fn, text in cases.items():
        with pytest.raises(sandbox.LeakError) as e:
            sandbox.pack(fn, DF.index[0])
        assert text in str(e.value), (fn.__name__, str(e.value))
        assert "at most 64 values in one" in str(e.value)


def test_environment_does_not_reach_the_sealed_process(fake, monkeypatch):
    fake["X"] = DF
    monkeypatch.setenv("XLEAK_R12", TEXT)

    def env(d, ns):
        s = os.environ.get("XLEAK_R12", "")
        return _s(d, [c == "1" for c in s] if s else [False] * len(d))
    for vec in (False, True):
        f = _fresh(env)
        if vec:
            f.vectorized_causal = True
        assert not expr.evaluate(f, expr.Namespace(DF, ticker="X")).any()

    def env_size(d, ns):
        return pd.Series(float(len(os.environ)), index=d.index)
    assert expr.evaluate_value(env_size, expr.Namespace(DF, ticker="X")).max() <= 6
    assert set(sandbox._child_env()) <= set(sandbox.ENV_KEEP) | {"PYTHONPATH", "BACKTESTER_SANDBOX_SYSPATH"}


def test_user_modules_are_measured_in_both_processes(fake, tmp_path, monkeypatch):
    fake["X"] = DF
    # a module only in sys.modules (made in the script) and imported inside the function
    m = types.ModuleType("r12_leakmod")
    m.flags_text = TEXT
    monkeypatch.setitem(sys.modules, "r12_leakmod", m)

    def via_module(d, ns):
        import r12_leakmod
        return _s(d, [c == "1" for c in r12_leakmod.flags_text])
    with pytest.raises(sandbox.LeakError, match="attribute 'flags_text' of module r12_leakmod"):
        expr.evaluate(via_module, expr.Namespace(DF, ticker="X"))
    # a module file the backtester never imported: measured when the sealed process imports it
    (tmp_path / "r12_filemod.py").write_text("OUTCOMES = " + repr(FLAGS) + "\n")
    (tmp_path / "r12_okmod.py").write_text("FAST, SLOW = 5, 20\nNAMES = ['SPY', 'TLT']\n\n"
                                           "def cross(c):\n    return c.rolling(FAST).mean() > c.rolling(SLOW).mean()\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    def via_file(d, ns):
        import r12_filemod
        return _s(d, r12_filemod.OUTCOMES)
    assert "r12_filemod" not in sys.modules
    with pytest.raises(sandbox.LeakError, match="module r12_filemod"):
        expr.evaluate(via_file, expr.Namespace(DF, ticker="X"))

    def via_ok_module(d, ns):
        import r12_okmod
        return r12_okmod.cross(d.close)
    s = expr.evaluate(via_ok_module, expr.Namespace(DF, ticker="X"))
    ref = DF.close.rolling(5).mean() > DF.close.rolling(20).mean()
    assert (s == ref).all()


def test_sealed_process_reads_no_source_or_data_file_as_text(fake, tmp_path):
    fake["X"] = DF
    src = tmp_path / "r12_hidden.py"
    src.write_text("# " + TEXT + "\n")
    prices = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(sandbox.__file__))), "data", "prices",
                          "SPY.csv")

    def open_code(d, ns):
        import io
        io.open_code(str(src)).read()
        return d.close > 0

    def raw_open(d, ns):
        import _io
        _io.open(str(src)).read()
        return d.close > 0

    def price_file(d, ns):
        import io
        io.open_code(prices).read()
        return d.close > 0
    for fn in (open_code, raw_open, price_file):
        with pytest.raises(expr.CallableIOError):
            expr.evaluate(fn, expr.Namespace(DF, ticker="X"))


def test_portfolio_function_capture_channels_are_refused(fake):
    fake["A"] = walk(7, 700, start="2010-01-04")
    fake["B"] = walk(8, 700, start="2010-01-04")
    from backtester import runner
    idx = fake["A"].index
    pairs = "|".join(d.strftime("%Y%m%d") + "1" for d in idx)

    def pick(date, h):
        k = pd.Timestamp(date).strftime("%Y%m%d")
        i = pairs.find(k)
        return {"A": 1.0} if i >= 0 and pairs[i + 8] == "1" else {"B": 1.0}

    def by_list(date, h, _days=list(range(300))):
        return {"A": 1.0} if len(h["A"]) in _days else {"B": 1.0}
    for fn in (pick, by_list):
        with pytest.raises(sandbox.LeakError, match="portfolio function"):
            runner.run(Portfolio(tree={"custom": fn, "tickers": ["A", "B"]}, rebalance="monthly", cash_rate=None,
                                 start="2010-06-01"))


TABLE = {f"T{i}": i / 100 for i in range(60)}          # a lookup table of 60 entries: fine
PARAMS = {"fast": 5, "slow": 20, "band": 0.01}


def test_legitimate_callables_still_run(fake):
    fake["X"] = DF
    lookback = [5, 10, 20]

    def params(d, ns, n=10, *, thr=0.0):
        """A docstring longer than the text limit is documentation, not data: it is not shipped to the sealed process.

        It goes on for a while to be over the limit: lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do
        eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation
        ullamco laboris nisi ut aliquip ex ea commodo consequat.
        """
        m = d.close.rolling(PARAMS["slow"]).mean()
        return (d.close > m * (1 + PARAMS["band"] * TABLE["T3"])) & (d.close.pct_change(lookback[0]) > thr)
    params.note = "my rule"
    s = expr.evaluate(params, expr.Namespace(DF, ticker="X"))
    m = DF.close.rolling(20).mean()
    ref = (DF.close > m * (1 + 0.01 * 0.03)) & (DF.close.pct_change(5) > 0)
    assert (s == ref).all() and s.any()
    v = _fresh(params)
    v.vectorized_causal = True
    assert (expr.evaluate(v, expr.Namespace(DF, ticker="X")) == ref).all()
    rng_seeded = functools.partial(lambda d, ns, k: d.close > d.close.shift(k), k=1)
    rng_seeded.__name__ = "partial_rule"
    assert expr.evaluate(rng_seeded, expr.Namespace(DF, ticker="X")).any()


# ------------------------------------------------------------ 2. deciding on today's open and filling at it

def _gap_frame():
    df = walk(21, n=600)
    o = df.open.to_numpy().copy()
    o[::17] = df.close.shift(1).to_numpy()[::17] * 0.97        # a 3% gap down every 17th bar
    o[0] = df.close.iloc[0]
    df["open"] = o
    df["high"] = np.maximum(df.high, o)
    df["low"] = np.minimum(df.low, o)
    return df


def test_open_reaction_fill_for_rules_that_read_todays_open(fake):
    fake["X"] = _gap_frame()
    base = dict(cash_rate=None, universe=["X"], entry="gap <= -0.02", entry_fill="open", hold_bars=0)
    r5 = engine.run(Strategy(**base))
    r0 = engine.run(Strategy(**base, open_reaction_bps=0))
    t5, t0 = r5.trades, r0.trades
    assert len(t5) == len(t0) > 20
    opens = fake["X"].open.reindex(pd.to_datetime(t0.entry_date)).to_numpy()
    assert np.allclose(t0.entry_price, opens)                          # 0: at the print itself
    assert np.allclose(t5.entry_price, opens * 1.0005)                 # default: 0.05% worse
    assert r5.equity.iloc[-1] < r0.equity.iloc[-1]
    note = next(n for n in r5.strategy.notes if n.startswith("Warning: Decided on today's open"))
    assert "0.05% worse than the open print" in note and "next open" in note
    assert any("optimistic" in n for n in r0.strategy.notes if n.startswith("Warning: Decided on today's open"))
    # a short entry reacts the other way (sells a little below the print)
    rs = engine.run(Strategy(**{**base, "side": "short"}))
    assert np.allclose(rs.trades.entry_price, fake["X"].open.reindex(pd.to_datetime(rs.trades.entry_date)) * 0.9995)


def test_open_fills_that_do_not_read_the_open_are_unchanged(fake):
    fake["X"] = _gap_frame()
    s = Strategy(cash_rate=None, universe=["X"], entry="dow == 1", entry_fill="open", hold_bars=0)
    r = engine.run(s)
    assert np.allclose(r.trades.entry_price, fake["X"].open.reindex(pd.to_datetime(r.trades.entry_date)))
    assert not any(n.startswith("Warning: Decided on today's open") for n in r.strategy.notes)
    assert not s.open_reaction_rules()
    nxt = Strategy(cash_rate=None, universe=["X"], entry="gap <= -0.02", entry_fill="next_open", hold_bars=1)
    assert not nxt.open_reaction_rules()                   # the next open: yesterday's gap is known before it


def test_open_reaction_in_the_interpretation():
    from backtester import parser
    s = parser.parse("buy AAPL when it gaps down 2%, sell at the close")
    assert s.entry_fill == "open" and s.open_reaction_bps == 5
    assert "0.05% worse than the open print" in s.summary()
    assert any(n.startswith("Warning: Decided on today's open") for n in s.notes)
    assert expr.reads_todays_open("gap <= -0.02") and expr.reads_todays_open('sym("SPY").open > 1')
    assert not expr.reads_todays_open("ref(open, 1) > ref(close, 2)") and not expr.reads_todays_open("dow == 1")
    with pytest.raises(ValueError, match="open_reaction_bps"):
        Strategy(universe=["X"], entry="gap < 0", entry_fill="open", hold_bars=0, open_reaction_bps=-1).validate()


# ------------------------------------------------------------ 3. phrasings

@pytest.mark.parametrize("sentence", [
    "each month buy the top 10 Nasdaq 100 stocks by 12 month return, equal weight",
    "buy the top 10 Nasdaq 100 stocks by 12 month return every month",
    "each month hold the 10 Nasdaq 100 stocks with the highest 12 month return",
])
def test_top_n_momentum_phrasings(sentence):
    from backtester import parser
    ref = parser.parse("hold the top 10 Nasdaq 100 stocks by 12 month return, rebalance monthly")
    p = parser.parse(sentence)
    assert p.tree == ref.tree and p.rebalance == ref.rebalance == "monthly"
    assert any(n.startswith("Read as:") for n in p.notes) and p.description == sentence


def test_rotation_long_short_and_dip_phrasings():
    from backtester import parser
    p = parser.parse("rotate monthly between SPY, EFA and TLT into whichever had the best 3 month return")
    ref = parser.parse("hold the top 1 of SPY, EFA and TLT by 3 month return, rebalance monthly")
    assert p.tree == ref.tree and p.rebalance == "monthly"
    q = parser.parse("hold 100% SPY and short 50% SQQQ, rebalance monthly")
    ref = parser.parse("hold 100% SPY and -50% SQQQ, rebalance monthly")
    assert q.tree == ref.tree and q.rebalance == "monthly"
    s = parser.parse("buy the dip in NVDA: when it drops 10% from its 52 week high, sell after 20 days")
    assert s.universe == ["NVDA"] and s.entry == "(drawdown(close, 252) <= -0.1)" and s.hold_bars == 20
    # a signal sentence that merely contains the words is untouched
    t = parser.parse("short SPY when rsi(2) is above 90, cover after 3 days")
    assert not any(n.startswith("Read as:") for n in t.notes)


# ------------------------------------------------------------ 4. weekly values at the open

def test_weekly_close_at_the_open_is_the_last_completed_week():
    from backtester import parser
    s = parser.parse("buy SPY at the open when `open > weekly_close()`, sell after 5 days")
    assert s.entry == "open > ref(weekly_close(), 1)" and s.entry_fill == "open"
    assert any("last period completed before today" in n for n in s.notes)
    assert expr.open_time_periodic("open > weekly_close()") == "open > ref(weekly_close(), 1)"
    assert expr.open_time_periodic("open > weekly(sma(close, 4))") == "open > ref(weekly(sma(close, 4)), 1)"
    assert expr.open_time_periodic("close > weekly_close()") is None      # today's close: not about the week
    with pytest.raises(ValueError, match=r"ref\(weekly_close\(\), 1\)"):
        Strategy(universe=["SPY"], entry="open > weekly_close()", entry_fill="open", hold_bars=5).validate()
    # causal: on a week's last session weekly_close() is that day's close; ref(..., 1) is the week before
    df = walk(5, n=300)
    ns = expr.Namespace(df, ticker="X")
    wk = expr.evaluate_value("weekly_close()", ns)
    lag = expr.evaluate_value("ref(weekly_close(), 1)", ns)
    fri = np.asarray(df.index.dayofweek == 4)
    assert (wk[fri] == df.close[fri]).all() and not (lag[fri] == df.close[fri]).any()
    assert expr.open_safe("open > ref(weekly_close(), 1)") and not expr.open_safe("open > weekly_close()")


# ------------------------------------------------------------ 5. imports and dunder names in rules

@pytest.mark.parametrize("rule", ['__import__("os").system("ls") > 0', "close.__class__ > 0", "__builtins__ > 1",
                                  'sma(__import__("os"), 5) > 1'])
def test_imports_and_dunder_names_have_a_clear_message(rule):
    with pytest.raises(ValueError, match="imports and names starting with an underscore"):
        expr.compile_expr(rule)
