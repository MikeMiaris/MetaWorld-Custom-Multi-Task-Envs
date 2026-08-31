"""Convenience wrapper for training button_push."""

import sys
from pathlib import Path

# Allow importing train_custom_mt_pair when executed from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_custom_mt_pair import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--pair", "button_push"] + sys.argv[1:]
    main()
