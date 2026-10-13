"""Streamlit entry point: ``streamlit run app.py``.

All application logic lives in the ``winemood`` package (``src/winemood/app``).
"""

import sys
from pathlib import Path

# Allow running from a fresh clone without ``pip install -e .``.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from winemood.app.main import run  # noqa: E402

run()
