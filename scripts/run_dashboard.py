"""Launch the Streamlit operator dashboard.

A thin wrapper so the dashboard starts the same way from a shell, a .bat file
or a scheduled task, without the caller needing to know the Streamlit
invocation or the module path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = PROJECT_ROOT / "app" / "dashboard" / "app.py"


def main() -> int:
    if not DASHBOARD.exists():
        print(f"Dashboard not found: {DASHBOARD}", file=sys.stderr)
        return 1

    # Create runtime directories up front so the dashboard never fails on a
    # missing log or output path.
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs").mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "-m", "streamlit", "run", str(DASHBOARD),
        "--server.headless", "true",
        *sys.argv[1:],
    ]
    print("Starting dashboard:", " ".join(command))
    try:
        return subprocess.call(command, cwd=str(PROJECT_ROOT))
    except FileNotFoundError:
        print(
            "Streamlit is not installed. Run:  pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
