"""The single AUTOPS command-line surface."""

from autops.commands import main, parser

__all__ = ["main", "parser"]

if __name__ == "__main__":
    raise SystemExit(main())
