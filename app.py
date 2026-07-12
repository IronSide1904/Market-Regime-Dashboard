from __future__ import annotations

import os


def _remove_dead_local_proxy() -> None:
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.environ.get(key, "")
        if value.rstrip("/").lower() == "http://127.0.0.1:9":
            os.environ.pop(key, None)
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")


_remove_dead_local_proxy()

from dashboard import render_dashboard


if __name__ == "__main__":
    render_dashboard()
