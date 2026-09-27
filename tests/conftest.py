"""Shared test settings.

The processed-price cache (data.load's results kept in .cache/prices, see backtester/data.py) is off for the test
suite: many tests patch the processing (data files, reference series, overrides) and must see it run. The cache's own
tests (test_performance.py) turn it on for themselves."""
import os

os.environ["BACKTESTER_CACHE"] = "0"
