"""Launch the Sheffield Crime Trends Visualiser Dash app."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.app.app import main  # noqa: E402

if __name__ == "__main__":
    main()
