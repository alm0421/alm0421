"""Trading platform application package.

Subsystems
----------
app.enums       Shared vocabulary (modes, asset classes, data freshness).
app.config      Typed configuration loading and validation.
app.clock       Market sessions, timezones, and staleness arithmetic.
app.data        Market data providers with explicit freshness labelling.
app.scanners    Universe filtering and ranking.
app.signals     Indicators and the signal schema.
app.strategies  Strategy evaluation producing signals.
app.risk        Pre-trade risk validation and the kill switch.
app.execution   Broker adapters, order validation, and mode gating.
app.portfolio   Position and account state.
app.dashboard   Streamlit operator dashboard.
app.logging     Structured logging and the audit trail.
"""

__version__ = "0.1.0"
