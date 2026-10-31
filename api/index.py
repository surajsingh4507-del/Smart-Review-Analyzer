import sys
import os
from pathlib import Path

# Add project root directory and api directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
api_dir = Path(__file__).resolve().parent

for p in [str(root_dir), str(api_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from main import app
