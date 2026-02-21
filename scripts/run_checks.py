from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_command(args: list[str]) -> None:
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    run_command([sys.executable, "-m", "py_compile", "project_config.py", "ml_pipeline_complete.py", "app.py"])
    run_command([sys.executable, "ml_pipeline_complete.py"])


if __name__ == "__main__":
    main()
