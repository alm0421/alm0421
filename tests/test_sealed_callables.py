"""Python rules and portfolio functions run in a sealed process (backtester/sandbox.py) fed one day at a time.

The five leaks a QuantConnect reviewer found (round 11), each refused or neutralised:
  1. the streamed namespace's link to the engine's namespace (ns._parent / ns._prefix_src),
  2. walking inspect.stack() / gc for the engine's full DataFrame,
  3. a closure / global over data loaded before the run,
  4. data.load() inside a vectorized_causal rule,
  5. the same through a portfolio custom function.
"""
from __future__ import annotations

import gc
import inspect
import subprocess
import time

import numpy as np
import pandas as pd
import pytest

from backtester import data, engine, expr, sandbox
from backtester.portfolio import Portfolio
from backtester.strategy import Strategy


def walk(seed, n=800, start="2012-01-02"):
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


def _tomorrow_up(found, d):
    """What a leak does with a frame reaching past today: 'tomorrow closes higher'."""
    return (found.close.shift(-1) > found.close).reindex(d.index).fillna(False).astype(bool)


def _largest_frame_on_stack(d):
    best = None
    for fr in inspect.stack():
        for v in list(fr.frame.f_locals.values()):
            if isinstance(v, pd.DataFrame) and "close" in v and (best is None or len(v) > len(best)):
                best = v
    return best if best is not None else d


def _largest_frame_in_memory(d):
    best = d
    for o in gc.get_objects():
        if isinstance(o, pd.DataFrame) and "close" in o.columns and len(o) > len(best):
            best = o
    return best


# ------------------------------------------------------------ 1. no link back to the engine's namespace

def test_namespace_has_no_link_to_the_engine(fake):
    df = walk(1)
    fake["X"] = df

    def parent(d, ns):
        return ns._parent["close"].shift(-1).reindex(d.index) > d.close

    def parent_src(d, ns):
        return ns._parent._prefix_src[0].close.shift(-1).reindex(d.index) > d.close
    for fn in (parent, parent_src):
        with pytest.raises(AttributeError):
            expr.evaluate(fn, expr.Namespace(df, ticker="X"))
        fn.vectorized_causal = True
        with pytest.raises(AttributeError):
            expr.evaluate(fn, expr.Namespace(df, ticker="X"))


# ------------------------------------------------------------ 2. the stack and the heap hold nothing later

@pytest.mark.parametrize("finder", [_largest_frame_on_stack, _largest_frame_in_memory])
def test_stack_and_gc_walks_find_nothing_later(fake, finder):
    df = walk(2)
    fake["X"] = df

    def leak(d, ns):
        return _tomorrow_up(finder(d), d)
    s = expr.evaluate(leak, expr.Namespace(df, ticker="X"))
    assert not s.any()          # at every bar the largest frame anywhere in the process ends at that bar
    with pytest.raises(ValueError, match="uses future data"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=leak, hold_bars=1))
    leak.vectorized_causal = True     # given the whole history on purpose, then caught by the probe
    expr._STREAMED.clear()
    with pytest.raises(ValueError, match="uses future data"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=leak, hold_bars=1))


# ------------------------------------------------------------ 3. data captured before the run is refused

def test_closures_and_globals_over_full_data_are_refused(fake):
    df = walk(3)
    fake["X"] = df
    full = data.load("X")

    def closure(d, ns):
        return _tomorrow_up(full, d)

    def default_arg(d, ns, _full=full.close.to_numpy()):
        return pd.Series(np.r_[_full[1:len(d)] > _full[:len(d) - 1], False], index=d.index)

    def nested(d, ns):
        return closure(d, ns)
    box = {"prices": {"X": full}}

    def in_a_dict(d, ns):
        return _tomorrow_up(box["prices"]["X"], d)
    for fn in (closure, default_arg, nested, in_a_dict):
        # a dated frame "uses" future data; a bare array of 800 numbers "may use" it (over the capture budget)
        with pytest.raises(sandbox.LeakError, match="uses? future data") as e:
            expr.evaluate(fn, expr.Namespace(df, ticker="X"))
        assert "captured outside the run" in str(e.value)
        with pytest.raises(ValueError, match="uses? future data"):
            engine.run(Strategy(cash_rate=None, universe=["X"], entry=fn, hold_bars=1))
    with pytest.raises(sandbox.LeakError, match="global 'FULL_GLOBAL'"):
        expr.evaluate(_uses_global, expr.Namespace(df, ticker="X"))


FULL_GLOBAL = walk(3)


def _uses_global(d, ns):
    return _tomorrow_up(FULL_GLOBAL, d)


def test_small_constants_are_allowed_and_captured_past_data_is_measured(fake):
    df = walk(4)
    fake["X"] = df
    weights = [0.1] * 10
    level = float(df.close.iloc[:100].mean())      # a number computed from data before the run: one value

    def ok(d, ns):
        return (d.close > level * sum(weights)) & (d.close > d.close.shift(1))
    with expr.stream_from(df.index[200]):
        s = expr.evaluate(ok, expr.Namespace(df, ticker="X"))
    assert s.iloc[200:].any() and not s.iloc[:200].any()
    # round 12: a captured frame, even one ending before the run, is data (its values could encode anything) and
    # counts against the capture budget (64 values)
    past = df.iloc[:100].copy()

    def with_past(d, ns):
        return d.close > past.close.mean()
    with expr.stream_from(df.index[200]), pytest.raises(sandbox.LeakError, match="variable 'past' captured by"):
        expr.evaluate(with_past, expr.Namespace(df, ticker="X"))


# ------------------------------------------------------------ 4. data.load in a vectorized rule is cut

def test_data_load_inside_a_vectorized_rule_is_cut_and_refused(fake):
    df = walk(5)
    fake["X"] = df

    def loads(d, ns):
        full = data.load("X")
        return _tomorrow_up(full, d)
    loads.vectorized_causal = True
    with pytest.raises(ValueError, match="uses future data"):
        engine.run(Strategy(cash_rate=None, universe=["X"], entry=loads, hold_bars=1))
    # the whole-history call itself only gets data up to the last bar it is given
    out = expr._call_once(loads, expr.Namespace(df.iloc[:300], ticker="X"), "bool")
    assert not out.iloc[-1]


def test_files_processes_and_other_data_functions_are_refused(fake, tmp_path):
    df = walk(6)
    fake["X"] = df
    p = tmp_path / "X.csv"
    df.to_csv(p)

    def shell(d, ns):
        subprocess.check_output(["cat", str(p)])
        return d.close > 0

    def fileio(d, ns):
        import _io
        _io.FileIO(str(p)).read()
        return d.close > 0

    def other_data(d, ns):
        data.membership()
        return d.close > 0
    for fn in (shell, fileio, other_data):
        with pytest.raises(expr.CallableIOError):
            expr.evaluate(fn, expr.Namespace(df, ticker="X"))


# ------------------------------------------------------------ 5. portfolio functions

def _pf(fake, fn, n=900):
    fake["A"] = walk(7, n, start="2010-01-04")
    fake["B"] = walk(8, n, start="2010-01-04")
    return engine_run_pf(fn)


def engine_run_pf(fn):
    from backtester import runner
    return runner.run(Portfolio(tree={"custom": fn, "tickers": ["A", "B"]}, rebalance="monthly", cash_rate=None,
                                start="2010-06-01"))


def _honest(date, h):
    return {"A": 0.5, "B": 0.5}


def test_portfolio_closure_is_refused(fake):
    fake["A"] = walk(7, 900, start="2010-01-04")
    fake["B"] = walk(8, 900, start="2010-01-04")
    full = {t: data.load(t).close for t in ("A", "B")}      # loaded before the run

    def peek(date, h):
        nxt = {t: s.loc[date:].iloc[min(21, len(s.loc[date:]) - 1)] / s.loc[date] - 1 for t, s in full.items()}
        return {max(nxt, key=nxt.get): 1.0}
    with pytest.raises(sandbox.LeakError, match="portfolio function peek"):
        engine_run_pf(peek)


def test_portfolio_stack_walk_finds_only_the_history_up_to_the_day(fake):
    def stack(date, h):
        for fr in inspect.stack():
            for v in list(fr.frame.f_locals.values()):
                if isinstance(v, pd.DataFrame) and "close" in v and len(v) and v.index[-1] > date:
                    return {"A": 1.0}              # found a frame reaching past today
        for o in gc.get_objects():
            if isinstance(o, pd.DataFrame) and "close" in o.columns and len(o) and o.index[-1] > date:
                return {"A": 1.0}
        assert all(x.index[-1] <= date for x in h.values())
        return {"A": 0.5, "B": 0.5}
    honest = _pf(fake, _honest)
    probed = engine_run_pf(stack)
    pd.testing.assert_series_equal(honest.equity, probed.equity)


def test_portfolio_function_gets_the_history_and_state_starts_fresh(fake):
    def momentum(date, h, _seen={"n": 0}):
        _seen["n"] += 1
        r = {t: x.close.iloc[-1] / x.close.iloc[max(0, len(x) - 60)] - 1 for t, x in h.items()}
        return {max(r, key=r.get): 1.0}
    a = _pf(fake, momentum)
    b = engine_run_pf(momentum)
    pd.testing.assert_series_equal(a.equity, b.equity)      # the default dict starts empty in each run
    assert a.equity.iloc[-1] != _pf(fake, _honest).equity.iloc[-1]


# ------------------------------------------------------------ legitimate callables: same answers, timing

def test_legitimate_callables_are_unchanged_and_fast_enough(fake, capsys):
    df = walk(9, n=2000)
    fake["X"] = df

    def causal(d, ns):
        return (d.close < d.close.rolling(10).min().shift(1)) & (ns["rsi"](2) < 30)
    ref = pd.Series([bool(expr._last(causal(df.iloc[:i + 1], expr.Namespace(df.iloc[:i + 1], ticker="X")), "bool"))
                     for i in range(0, 2000, 97)], index=df.index[::97])
    sandbox._server()            # started once per process (about a second), not counted
    t = time.perf_counter()
    s = expr.evaluate(causal, expr.Namespace(df, ticker="X"))
    secs = time.perf_counter() - t
    assert (s.iloc[::97] == ref).all()
    with capsys.disabled():
        print(f"\n[sealed streaming] 2,000 bars in {secs:.2f}s ({secs / 2000 * 1e3:.2f} ms per bar)")
    assert secs < 30
