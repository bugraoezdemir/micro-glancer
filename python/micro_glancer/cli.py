"""The ``micro-glancer`` command: the app window, or one line saying what to install."""
from __future__ import annotations

import sys


def main(argv=None) -> int:
    """``micro-glancer [PATH ...]``: the window (``micro_glancer.app``), which needs PyQt6."""
    try:
        import PyQt6.QtWidgets  # noqa: F401
    except ModuleNotFoundError:
        print('micro-glancer\'s window needs PyQt6: pip install "micro-glancer[qt]"',
              file=sys.stderr)
        return 1
    from .app import main as app_main

    return app_main(argv)


if __name__ == "__main__":
    sys.exit(main())
