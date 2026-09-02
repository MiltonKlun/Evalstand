"""Built-in scorers.

These live outside the seven-name public surface: importing a scorer is an
explicit act, and the library will grow past what a top-level namespace should
carry.
"""

from evalstand.scorers.text import exact

__all__ = ["exact"]
