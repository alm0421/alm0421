"""The live-trading gate.

Live trading is refused unless **three independent gates** are all open. They
are deliberately in three different places so that no single edit, copied
config, or leaked environment file can enable real-money trading by accident:

1. **Build gate** - :data:`LIVE_TRADING_APPROVED` in this module. A source
   change, reviewed and committed.
2. **Config gate** - ``execution.live_trading_enabled`` in ``config.yaml``,
   plus ``mode: live``.
3. **Runtime gate** - the ``ALPACA_LIVE_TRADING_ACK`` environment variable set
   to :data:`LIVE_ACK_PHRASE` on the machine actually running the platform.

Gate 1 is currently CLOSED. Live trading is disabled in this release and has
never been validated against a live account.

To enable live trading later, all of the following must happen, in order:

* Validate the full paper path end to end against a real paper account, and
  confirm order construction, risk limits and reconciliation behave correctly.
* Obtain explicit written approval from the account owner.
* Set ``LIVE_TRADING_APPROVED = True`` below in a reviewed commit.
* Set ``execution.live_trading_enabled: true`` and ``mode: live`` in config.
* Export ``ALPACA_LIVE_TRADING_ACK`` with the exact acknowledgement phrase.
* Populate ``ALPACA_LIVE_API_KEY`` / ``ALPACA_LIVE_SECRET_KEY``.
"""

from __future__ import annotations

import os

from app.config import AppConfig
from app.enums import TradingMode

#: Gate 1. Changing this to True is a deliberate, reviewable source change.
#: DO NOT flip this without the validation and written approval described above.
LIVE_TRADING_APPROVED: bool = False

#: Gate 3.
LIVE_ACK_ENV_VAR = "ALPACA_LIVE_TRADING_ACK"
LIVE_ACK_PHRASE = "I_ACCEPT_LIVE_TRADING_RISK"


class LiveTradingDisabled(RuntimeError):
    """Raised when live execution is requested but not fully authorised."""


def live_gate_status(config: AppConfig) -> dict[str, object]:
    """Report each gate's state. Used by preflight and the dashboard.

    Never raises - this is a status query, not an authorisation check.
    """
    ack = os.environ.get(LIVE_ACK_ENV_VAR, "")
    gates = {
        "build_approved": LIVE_TRADING_APPROVED,
        "config_enabled": config.execution.live_trading_enabled,
        "mode_is_live": config.mode is TradingMode.LIVE,
        "runtime_ack_present": ack == LIVE_ACK_PHRASE,
        "live_credentials_present": (
            config.mode is TradingMode.LIVE and config.credentials.is_complete
        ),
    }
    gates["all_gates_open"] = all(
        (gates["build_approved"], gates["config_enabled"], gates["runtime_ack_present"])
    )
    return gates


def assert_live_trading_allowed(config: AppConfig) -> None:
    """Raise :class:`LiveTradingDisabled` unless every gate is open.

    Failure is the default. Any gate that cannot be positively confirmed open
    blocks execution.
    """
    closed: list[str] = []

    if not LIVE_TRADING_APPROVED:
        closed.append(
            "build gate: LIVE_TRADING_APPROVED is False in app/execution/live_gate.py. "
            "Live trading is disabled in this release and has not been validated "
            "against a live account."
        )
    if not config.execution.live_trading_enabled:
        closed.append(
            "config gate: execution.live_trading_enabled is false in "
            f"{config.source_path}."
        )
    if os.environ.get(LIVE_ACK_ENV_VAR, "") != LIVE_ACK_PHRASE:
        closed.append(
            f"runtime gate: environment variable {LIVE_ACK_ENV_VAR} is not set to "
            "the required acknowledgement phrase."
        )
    if not config.credentials.is_complete:
        closed.append(
            "credentials gate: ALPACA_LIVE_API_KEY / ALPACA_LIVE_SECRET_KEY are not set."
        )

    if closed:
        detail = "\n  - ".join(closed)
        raise LiveTradingDisabled(
            "Live trading is BLOCKED. The following gate(s) are closed:\n  - "
            + detail
            + "\n\nNo order was sent. See app/execution/live_gate.py for the full "
            "enablement procedure."
        )
