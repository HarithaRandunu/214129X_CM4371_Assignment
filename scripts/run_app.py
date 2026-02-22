from __future__ import annotations

import io
import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _resolve_app_url() -> str:
    host = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")
    port = os.getenv("STREAMLIT_SERVER_PORT", "8501")

    if host in {"0.0.0.0", "127.0.0.1", "localhost"}:
        try:
            host = socket.gethostbyname(socket.gethostname())
        except OSError:
            host = "localhost"

    return f"http://{host}:{port}"


def _print_terminal_qr(url: str) -> None:
    print(f"\nApp URL: {url}")
    try:
        import qrcode

        print("\nScan this QR code to open the app:")
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        stream = io.StringIO()
        qr.print_ascii(out=stream, invert=True)
        print(stream.getvalue())
    except Exception:
        print(
            "QR code not shown (install with: pip install qrcode[pil])."
        )


def main() -> None:
    url = _resolve_app_url()
    _print_terminal_qr(url)

    try:
        raise SystemExit(
            subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "app.py",
                    "--server.address",
                    "0.0.0.0",
                ],
                cwd=ROOT,
            )
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
