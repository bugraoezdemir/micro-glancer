"""``python -m micro_viewer [PATH ...]``: micro-viewer's app window (as the command)."""
import sys

from .cli import main

sys.exit(main())
