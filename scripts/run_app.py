from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    try:
        raise SystemExit(
            subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=ROOT)
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
