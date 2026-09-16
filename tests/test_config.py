"""Configuration loading and validation."""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from app.config import ConfigError, load_config
from app.enums import AssetClass, TradingMode


def test_default_mode_is_signal_only(config):
    """The safest mode must be the shipped default."""
    assert config.mode is TradingMode.SIGNAL_ONLY


def test_live_trading_disabled_in_shipped_config(config):
    assert config.execution.live_trading_enabled is False


def test_futures_not_in_allowed_asset_classes(config):
    """No shipped adapter supports futures, so it must not be routable."""
    assert AssetClass.FUTURES not in config.execution.allowed_asset_classes


def test_credentials_never_appear_in_redacted_dict(config):
    payload = config.as_redacted_dict()
    flattened = str(payload).lower()
    assert "secret" not in flattened.replace("secret_key", "")
    assert "api_key" not in flattened
    assert payload["credentials_present"] in (True, False)


def test_credentials_repr_does_not_leak(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER_API_KEY", "PKSUPERSECRETKEY")
    monkeypatch.setenv("ALPACA_PAPER_SECRET_KEY", "shhhhh-very-secret")
    config = load_config()
    assert config.credentials.is_complete
    assert "PKSUPERSECRETKEY" not in repr(config.credentials)
    assert "shhhhh" not in repr(config.credentials)
    assert "PKSUPERSECRETKEY" not in str(config.as_redacted_dict())


def test_trading_mode_env_override(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "paper")
    assert load_config().mode is TradingMode.PAPER


def test_invalid_mode_rejected(monkeypatch):
    monkeypatch.setenv("TRADING_MODE", "yolo")
    with pytest.raises(ConfigError, match="Invalid value"):
        load_config()


def _write_config(tmp_path, overrides: dict, base_config):
    raw = yaml.safe_load(base_config.source_path.read_text())
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if key:
            raw.setdefault(section, {})[key] = value
        else:
            raw[section] = value
    raw["universe"] = {"file": str(base_config.source_path.parent / "universe.yaml")}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_futures_in_allowed_asset_classes_is_rejected(tmp_path, config):
    path = _write_config(
        tmp_path, {"execution.allowed_asset_classes": ["us_equity", "futures"]}, config
    )
    with pytest.raises(ConfigError, match="futures"):
        load_config(path)


def test_live_mode_without_config_flag_is_rejected(tmp_path, config, monkeypatch):
    monkeypatch.delenv("TRADING_MODE", raising=False)
    path = _write_config(
        tmp_path, {"mode": "live", "execution.live_trading_enabled": False}, config
    )
    with pytest.raises(ConfigError, match="live_trading_enabled"):
        load_config(path)


def test_enabled_strategy_without_params_is_rejected(tmp_path, config):
    path = _write_config(tmp_path, {"strategies": {"enabled": ["ghost_strategy"]}}, config)
    with pytest.raises(ConfigError, match="ghost_strategy"):
        load_config(path)


def test_fraction_above_one_rejected(tmp_path, config):
    path = _write_config(tmp_path, {"risk.max_position_pct_equity": 1.5}, config)
    with pytest.raises(ConfigError, match="fraction"):
        load_config(path)


def test_negative_limit_rejected(tmp_path, config):
    path = _write_config(tmp_path, {"risk.max_daily_loss": -100}, config)
    with pytest.raises(ConfigError, match="positive"):
        load_config(path)


def test_missing_config_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")


def test_universe_symbols_are_uppercased(config):
    for symbols in config.universe.values():
        assert all(s == s.upper() for s in symbols)
