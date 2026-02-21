from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", "app.py"], cwd=ROOT))


if __name__ == "__main__":
    main()
