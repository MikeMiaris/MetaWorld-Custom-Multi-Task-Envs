"""Convenience wrapper for evaluating button_push."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_custom_mt_pair import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--pair", "button_push"] + sys.argv[1:]
    main()
