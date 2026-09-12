"""Pytest config - make the src-layout `pqdna` package importable without install."""

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
