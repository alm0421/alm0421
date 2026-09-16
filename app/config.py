"""Configuration loading, validation and credential resolution.

Design rules enforced here:

* The config file never contains secrets. Credentials come only from the
  environment (optionally seeded from a gitignored ``.env``).
* An unknown or malformed value is a hard error at load time, not a surprise at
  order-submission time.
* ``__repr__``/``as_redacted_dict`` never expose key material.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import load_dotenv

from app.enums import AssetClass, OrderType, TimeInForce, TradingMode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration is missing, malformed or internally inconsistent."""


# ---------------------------------------------------------------------------
# Typed sections
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    provider: str
    feed: str
    stale_quote_seconds: int
    stale_bar_seconds: int
    display_timezone: str
    max_retries: int
    retry_backoff_seconds: float


@dataclass(frozen=True)
class ScannerConfig:
    enabled: tuple[str, ...]
    refresh_seconds: int
    max_results: int
    lookback_bars: int
    bar_timeframe: str


@dataclass(frozen=True)
class StrategyConfig:
    enabled: tuple[str, ...]
    params: Mapping[str, Mapping[str, Any]]

    def params_for(self, name: str) -> Mapping[str, Any]:
        return self.params.get(name, {})


@dataclass(frozen=True)
class RiskConfig:
    max_open_positions: int
    max_position_notional: float
    max_position_pct_equity: float
    max_symbol_concentration_pct: float
    max_daily_loss: float
    max_daily_loss_pct: float
    max_trades_per_day: int
    min_account_equity: float
    require_stop_loss: bool
    allow_extended_hours: bool
    block_when_market_closed: bool
    duplicate_order_window_seconds: int
    kill_switch_file: Path


@dataclass(frozen=True)
class ExecutionConfig:
    live_trading_enabled: bool
    default_time_in_force: TimeInForce
    allowed_order_types: frozenset[OrderType]
    allowed_asset_classes: frozenset[AssetClass]
    order_submit_retries: int
    order_submit_backoff_seconds: float


@dataclass(frozen=True)
class LoggingConfig:
    level: str
    dir: Path
    json_logs: bool
    console: bool


@dataclass(frozen=True)
class Credentials:
    """Broker credentials resolved from the environment.

    ``api_key``/``secret_key`` are held in memory only. ``__repr__`` is
    overridden so a stray log or traceback cannot leak them.
    """

    api_key: str | None
    secret_key: str | None
    paper: bool

    @property
    def is_complete(self) -> bool:
        return bool(self.api_key) and bool(self.secret_key)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        state = "set" if self.is_complete else "missing"
        return f"Credentials(paper={self.paper}, credentials={state})"

    __str__ = __repr__


@dataclass(frozen=True)
class AppConfig:
    mode: TradingMode
    broker: str
    data: DataConfig
    scanner: ScannerConfig
    strategies: StrategyConfig
    risk: RiskConfig
    execution: ExecutionConfig
    logging: LoggingConfig
    outputs_dir: Path
    universe: Mapping[AssetClass, tuple[str, ...]]
    source_path: Path
    _credentials: Credentials = field(repr=False, default=Credentials(None, None, True))

    @property
    def credentials(self) -> Credentials:
        return self._credentials

    def as_redacted_dict(self) -> dict[str, Any]:
        """A log-safe view of the configuration. Contains no key material."""
        return {
            "mode": self.mode.value,
            "broker": self.broker,
            "data": {
                "provider": self.data.provider,
                "feed": self.data.feed,
                "stale_quote_seconds": self.data.stale_quote_seconds,
                "display_timezone": self.data.display_timezone,
            },
            "scanner": {"enabled": list(self.scanner.enabled)},
            "strategies": {"enabled": list(self.strategies.enabled)},
            "risk": {
                "max_open_positions": self.risk.max_open_positions,
                "max_position_notional": self.risk.max_position_notional,
                "max_daily_loss": self.risk.max_daily_loss,
                "require_stop_loss": self.risk.require_stop_loss,
                "kill_switch_file": str(self.risk.kill_switch_file),
            },
            "execution": {
                "live_trading_enabled": self.execution.live_trading_enabled,
                "allowed_asset_classes": sorted(
                    a.value for a in self.execution.allowed_asset_classes
                ),
                "default_time_in_force": self.execution.default_time_in_force.value,
            },
            "credentials_present": self.credentials.is_complete,
            "source_path": str(self.source_path),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require(section: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing required config key '{where}.{key}'")
    return section[key]


def _as_path(value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _enum(enum_cls, value: Any, where: str):
    try:
        return enum_cls(str(value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(e.value for e in enum_cls)
        raise ConfigError(
            f"Invalid value {value!r} for '{where}'. Allowed: {allowed}"
        ) from exc


def _positive(value: Any, where: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'{where}' must be a number, got {value!r}") from exc
    if number < 0 or (number == 0 and not allow_zero):
        raise ConfigError(f"'{where}' must be positive, got {number}")
    return number


def _fraction(value: Any, where: str) -> float:
    number = _positive(value, where)
    if number > 1:
        raise ConfigError(f"'{where}' is a fraction and must be <= 1.0, got {number}")
    return number


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_universe(path: Path) -> dict[AssetClass, tuple[str, ...]]:
    if not path.exists():
        raise ConfigError(f"Universe file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Universe file {path} must contain a mapping")

    universe: dict[AssetClass, tuple[str, ...]] = {}
    for key, symbols in raw.items():
        asset_class = _enum(AssetClass, key, f"universe.{key}")
        if symbols is None:
            symbols = []
        if not isinstance(symbols, list):
            raise ConfigError(f"universe.{key} must be a list of symbols")
        cleaned = tuple(str(s).strip().upper() for s in symbols if str(s).strip())
        duplicates = {s for s in cleaned if cleaned.count(s) > 1}
        if duplicates:
            raise ConfigError(
                f"Duplicate symbols in universe.{key}: {sorted(duplicates)}"
            )
        universe[asset_class] = cleaned
    return universe


def _resolve_credentials(mode: TradingMode) -> Credentials:
    """Pick the credential pair matching the mode.

    Paper and live credentials are held in *separate* environment variables so
    that a mode change can never accidentally reuse the wrong account's keys.
    """
    if mode is TradingMode.LIVE:
        return Credentials(
            api_key=os.environ.get("ALPACA_LIVE_API_KEY") or None,
            secret_key=os.environ.get("ALPACA_LIVE_SECRET_KEY") or None,
            paper=False,
        )
    return Credentials(
        api_key=os.environ.get("ALPACA_PAPER_API_KEY") or None,
        secret_key=os.environ.get("ALPACA_PAPER_SECRET_KEY") or None,
        paper=True,
    )


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load, validate and freeze the application configuration.

    Resolution order for the config path: explicit argument, then the
    ``TRADING_CONFIG`` environment variable, then ``config/config.yaml``.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    config_path = _as_path(path or os.environ.get("TRADING_CONFIG") or DEFAULT_CONFIG_PATH)
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"Config file {config_path} must contain a mapping")

    # --- mode (env override wins) ---
    mode_value = os.environ.get("TRADING_MODE") or _require(raw, "mode", "root")
    mode = _enum(TradingMode, mode_value, "mode")

    # --- data ---
    data_raw = _require(raw, "data", "root")
    data = DataConfig(
        provider=str(_require(data_raw, "provider", "data")).lower(),
        feed=str(_require(data_raw, "feed", "data")).lower(),
        stale_quote_seconds=int(_positive(_require(data_raw, "stale_quote_seconds", "data"), "data.stale_quote_seconds")),
        stale_bar_seconds=int(_positive(_require(data_raw, "stale_bar_seconds", "data"), "data.stale_bar_seconds")),
        display_timezone=str(data_raw.get("display_timezone", "America/New_York")),
        max_retries=int(_positive(data_raw.get("max_retries", 3), "data.max_retries", allow_zero=True)),
        retry_backoff_seconds=_positive(data_raw.get("retry_backoff_seconds", 1.0), "data.retry_backoff_seconds", allow_zero=True),
    )

    # --- scanner ---
    scanner_raw = raw.get("scanner", {}) or {}
    scanner = ScannerConfig(
        enabled=tuple(str(s).lower() for s in scanner_raw.get("enabled", []) or []),
        refresh_seconds=int(_positive(scanner_raw.get("refresh_seconds", 60), "scanner.refresh_seconds")),
        max_results=int(_positive(scanner_raw.get("max_results", 25), "scanner.max_results")),
        lookback_bars=int(_positive(scanner_raw.get("lookback_bars", 120), "scanner.lookback_bars")),
        bar_timeframe=str(scanner_raw.get("bar_timeframe", "5Min")),
    )

    # --- strategies ---
    strategies_raw = dict(raw.get("strategies", {}) or {})
    enabled_strategies = tuple(str(s).lower() for s in strategies_raw.pop("enabled", []) or [])
    strategy_params = {
        str(k).lower(): dict(v) for k, v in strategies_raw.items() if isinstance(v, Mapping)
    }
    for name in enabled_strategies:
        if name not in strategy_params:
            raise ConfigError(
                f"Strategy '{name}' is enabled but has no parameter block under 'strategies.{name}'"
            )
    strategies = StrategyConfig(enabled=enabled_strategies, params=strategy_params)

    # --- risk ---
    risk_raw = _require(raw, "risk", "root")
    risk = RiskConfig(
        max_open_positions=int(_positive(_require(risk_raw, "max_open_positions", "risk"), "risk.max_open_positions")),
        max_position_notional=_positive(_require(risk_raw, "max_position_notional", "risk"), "risk.max_position_notional"),
        max_position_pct_equity=_fraction(_require(risk_raw, "max_position_pct_equity", "risk"), "risk.max_position_pct_equity"),
        max_symbol_concentration_pct=_fraction(risk_raw.get("max_symbol_concentration_pct", 0.25), "risk.max_symbol_concentration_pct"),
        max_daily_loss=_positive(_require(risk_raw, "max_daily_loss", "risk"), "risk.max_daily_loss"),
        max_daily_loss_pct=_fraction(risk_raw.get("max_daily_loss_pct", 0.02), "risk.max_daily_loss_pct"),
        max_trades_per_day=int(_positive(risk_raw.get("max_trades_per_day", 20), "risk.max_trades_per_day")),
        min_account_equity=_positive(risk_raw.get("min_account_equity", 0.0), "risk.min_account_equity", allow_zero=True),
        require_stop_loss=bool(risk_raw.get("require_stop_loss", True)),
        allow_extended_hours=bool(risk_raw.get("allow_extended_hours", False)),
        block_when_market_closed=bool(risk_raw.get("block_when_market_closed", True)),
        duplicate_order_window_seconds=int(_positive(risk_raw.get("duplicate_order_window_seconds", 300), "risk.duplicate_order_window_seconds", allow_zero=True)),
        kill_switch_file=_as_path(risk_raw.get("kill_switch_file", "logs/KILL_SWITCH")),
    )

    # --- execution ---
    exec_raw = _require(raw, "execution", "root")
    allowed_types = exec_raw.get("allowed_order_types") or []
    allowed_classes = exec_raw.get("allowed_asset_classes") or []
    if not allowed_types:
        raise ConfigError("'execution.allowed_order_types' must list at least one order type")
    if not allowed_classes:
        raise ConfigError("'execution.allowed_asset_classes' must list at least one asset class")
    execution = ExecutionConfig(
        live_trading_enabled=bool(exec_raw.get("live_trading_enabled", False)),
        default_time_in_force=_enum(TimeInForce, exec_raw.get("default_time_in_force", "day"), "execution.default_time_in_force"),
        allowed_order_types=frozenset(
            _enum(OrderType, t, "execution.allowed_order_types") for t in allowed_types
        ),
        allowed_asset_classes=frozenset(
            _enum(AssetClass, c, "execution.allowed_asset_classes") for c in allowed_classes
        ),
        order_submit_retries=int(_positive(exec_raw.get("order_submit_retries", 2), "execution.order_submit_retries", allow_zero=True)),
        order_submit_backoff_seconds=_positive(exec_raw.get("order_submit_backoff_seconds", 1.0), "execution.order_submit_backoff_seconds", allow_zero=True),
    )

    # --- logging / outputs ---
    log_raw = raw.get("logging", {}) or {}
    logging_cfg = LoggingConfig(
        level=str(log_raw.get("level", "INFO")).upper(),
        dir=_as_path(log_raw.get("dir", "logs")),
        json_logs=bool(log_raw.get("json_logs", True)),
        console=bool(log_raw.get("console", True)),
    )
    outputs_dir = _as_path((raw.get("outputs", {}) or {}).get("dir", "outputs"))

    # --- universe ---
    universe_path = _as_path((raw.get("universe", {}) or {}).get("file", "config/universe.yaml"))
    universe = _load_universe(universe_path)

    config = AppConfig(
        mode=mode,
        broker=str((raw.get("account", {}) or {}).get("broker", "alpaca")).lower(),
        data=data,
        scanner=scanner,
        strategies=strategies,
        risk=risk,
        execution=execution,
        logging=logging_cfg,
        outputs_dir=outputs_dir,
        universe=universe,
        source_path=config_path,
        _credentials=_resolve_credentials(mode),
    )
    _validate_cross_section(config)
    return config


def _validate_cross_section(config: AppConfig) -> None:
    """Checks that span more than one config section."""
    # Every symbol we intend to trade must belong to a permitted asset class.
    for asset_class, symbols in config.universe.items():
        if symbols and asset_class not in config.execution.allowed_asset_classes:
            # Not fatal: signal-only analysis of a non-tradable class is valid.
            # Order validation refuses it later, so this stays a load-time no-op.
            continue

    if config.risk.max_position_notional <= 0:
        raise ConfigError("risk.max_position_notional must be greater than zero")

    if AssetClass.FUTURES in config.execution.allowed_asset_classes:
        raise ConfigError(
            "execution.allowed_asset_classes includes 'futures', but no shipped "
            "broker adapter supports futures trading. Alpaca does not offer "
            "futures. Remove it, or add a futures-capable adapter first."
        )

    if config.mode is TradingMode.LIVE and not config.execution.live_trading_enabled:
        raise ConfigError(
            "mode is 'live' but execution.live_trading_enabled is false. "
            "Live trading requires every gate to be opened deliberately."
        )
