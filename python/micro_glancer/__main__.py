"""``python -m micro_glancer [PATH ...]``: micro-glancer's app window (as the command)."""
import sys

from .cli import main

sys.exit(main())
