"""
main.py
=======
Entry point for the Algorithmic Trading Strategy Backtester dashboard.

This is intentionally thin — all UI logic lives in ``app/dashboard.py``.
This file is responsible only for:

1. Setting Streamlit's page configuration (must be the very first
   Streamlit call in the script).
2. Verifying the project's working directory and data files are in a
   runnable state before handing off to the dashboard, so a missing
   file produces a clear on-page message instead of a raw traceback.
3. Invoking ``app.dashboard.render_dashboard()``.

Run with
--------
::

    streamlit run main.py

Must be run from the project root (the directory containing ``data/``,
``src/``, and ``app/``) so relative paths resolve correctly.
"""

from __future__ import annotations

import logging
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import streamlit as st

# ── Ensure the project root is importable regardless of invocation cwd ──────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Page configuration — must run before any other Streamlit command ────────
st.set_page_config(
    page_title="Algo Trading Backtester — TCS / Reliance / Infosys",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _verify_startup_requirements() -> list[str]:
    """
    Check that the directories this app depends on actually exist.

    Returns
    -------
    list[str]
        Human-readable problem descriptions. Empty if everything checks out.
    """
    problems: list[str] = []

    data_dir = os.path.join(_PROJECT_ROOT, "data", "processed")
    if not os.path.isdir(data_dir):
        problems.append(
            f"Processed data directory not found: `{data_dir}`. "
            "Run `python src/data/fetch_data.py` then "
            "`python src/data/preprocess.py` from the project root first."
        )

    src_dir = os.path.join(_PROJECT_ROOT, "src")
    if not os.path.isdir(src_dir):
        problems.append(
            f"`src/` package not found next to `main.py` (looked in `{_PROJECT_ROOT}`). "
            "Make sure you're running `streamlit run main.py` from the project root."
        )

    return problems


def main() -> None:
    """Verify startup requirements, then hand off to the dashboard."""
    problems = _verify_startup_requirements()
    if problems:
        st.error("⚠️ The dashboard could not start due to a configuration issue:")
        for p in problems:
            st.markdown(f"- {p}")
        logger.error("Startup checks failed: %s", problems)
        st.stop()
    st.warning(f"Python Executable Path: {sys.executable}")
    st.info(f"Python Version: {sys.version}")
    try:
        from app.dashboard import render_dashboard
        render_dashboard()
    except Exception as exc:                                     # noqa: BLE001
        logger.exception("Unhandled error while rendering the dashboard")
        st.error(
            "An unexpected error occurred while rendering the dashboard. "
            "Check the terminal logs for the full traceback."
        )
        st.exception(exc)


if __name__ == "__main__":
    main()
