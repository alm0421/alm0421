"""Shared test settings.

The processed-price cache (data.load's results kept in .cache/prices, see backtester/data.py) is off for the test
suite: many tests patch the processing (data files, reference series, overrides) and must see it run. The cache's own
tests (test_performance.py) turn it on for themselves."""
import os

import pytest

os.environ["BACKTESTER_CACHE"] = "0"


@pytest.fixture(autouse=True)
def _private_request_queue(tmp_path_factory, monkeypatch):
    """A missing-but-valid ticker is queued in data/requested_tickers.txt (backtester/coverage.py); tests (and the CLI
    subprocesses they start) write to a temporary queue instead of the repository's."""
    monkeypatch.setenv("BACKTESTER_QUEUE_FILE", str(tmp_path_factory.mktemp("queue") / "requested_tickers.txt"))
