"""Round 13 integration: the three review branches together, as the data grows (thousands of ticker files, among
them English words - BAND, MEAN, P, ...). A lowercase word of the parser's vocabulary is never read as a ticker, a
letter joined by '&' ('S&P') is never a ticker, and leverage written in an if/otherwise branch stays in that branch."""
import re

import pytest

from backtester import data, parser

AVAIL = set(data.available_tickers())


@pytest.fixture
def every_ticker(monkeypatch):
    """Pretend a ticker file exists for every 1-5 letter word of the parser's vocabulary (and of `extra`)."""
    def install(extra=()):
        known = set(AVAIL) | {w.upper() for w in parser._vocabulary() if re.fullmatch(r"[a-z]{1,5}", w)}
        known |= {w.upper() for w in extra}
        monkeypatch.setattr(parser, "_known", lambda: known)
        return known
    return install


def test_no_vocabulary_word_is_uppercased_into_a_ticker(every_ticker):
    every_ticker()
    vocab = sorted(w for w in parser._vocabulary() if re.fullmatch(r"[a-z]{3,5}", w))
    assert {"band", "mean", "cross", "rsi", "lower", "upper", "above", "below", "sma", "bars"} <= set(vocab)
    text = " ".join(vocab)
    assert parser._lowercase_tickers(text) == text


def test_lowercase_tickers_still_read(every_ticker):
    every_ticker(extra=("tqqq", "uvxy", "spy", "tlt", "bil"))
    assert parser._lowercase_tickers("hold 60% spy and 40% tlt") == "hold 60% SPY and 40% TLT"
    assert parser._lowercase_tickers("hold bil") == "hold BIL"


@pytest.mark.skipif(not {"SPY", "QQQ", "TLT"} <= AVAIL, reason="price data not downloaded")
@pytest.mark.parametrize("text", [
    "buy SPY when it closes below the lower bollinger band, sell when it crosses above the mean",
    "buy SPY when the close crosses above the upper band, hold 5 days",
    "buy QQQ when RSI(2) is below 10 and the close is above the 200 day moving average, sell when RSI(2) is above 70",
    "if SPY is above its 200 day moving average hold QQQ, otherwise hold TLT, rebalance monthly",
    "buy SPY at the next open when it is down 3 days in a row, sell after 5 days with a 2% stop loss",
])
def test_sentences_read_the_same_with_every_word_a_ticker(every_ticker, text):
    """Every 3-5 letter lowercase word of the sentence gets a ticker file: nothing about the reading changes."""
    before = parser.parse(text)
    every_ticker(extra=re.findall(r"(?<![A-Za-z])[a-z]{3,5}(?![A-Za-z])", text))
    after = parser.parse(text)
    assert after == before


def test_a_letter_joined_by_ampersand_is_not_a_ticker(every_ticker):
    every_ticker(extra=("S", "P", "L", "R", "D", "SPY"))
    assert parser.find_tickers("buy the S&P 500") == ["SPY"]
    assert parser.find_tickers("P&L and R&D") == []
    assert parser.find_tickers("buy P when ...") == ["P"]


@pytest.mark.skipif("SPY" not in AVAIL, reason="price data not downloaded")
def test_the_s_and_p_500_with_a_p_ticker_file(every_ticker):
    every_ticker(extra=("P", "S"))
    assert parser.parse("buy the S&P 500 when RSI(2) is below 5, hold 3 days").universe == ["SPY"]
