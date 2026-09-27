import pytest


@pytest.fixture(autouse=True)
def _private_request_queue(tmp_path_factory, monkeypatch):
    """A missing-but-valid ticker is queued in data/requested_tickers.txt (backtester/coverage.py); tests (and the CLI
    subprocesses they start) write to a temporary queue instead of the repository's."""
    monkeypatch.setenv("BACKTESTER_QUEUE_FILE", str(tmp_path_factory.mktemp("queue") / "requested_tickers.txt"))
