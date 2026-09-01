"""Phase 0 smoke test: the package imports and reports a version."""

import evalstand


def test_package_imports() -> None:
    assert evalstand.__version__
