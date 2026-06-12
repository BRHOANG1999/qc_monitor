"""Unit tests for the app-level Google OAuth helpers (auth.py).

The OAuth round-trip itself is exercised live, but the secret-loading
precedence and the token-verification guards are unit-testable.

Run with: pytest tests/test_auth_google.py -q
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dashboard.auth import (  # noqa: E402
    _read_secret,
    verify_google_id_token,
)


def test_read_secret_prefers_explicit_value(tmp_path, monkeypatch):
    monkeypatch.setenv("X_SECRET", "from-env")
    f = tmp_path / "s"
    f.write_text("from-file")
    assert _read_secret("explicit", "X_SECRET", str(f)) == "explicit"


def test_read_secret_falls_back_to_env(tmp_path, monkeypatch):
    monkeypatch.setenv("X_SECRET", "from-env")
    f = tmp_path / "s"
    f.write_text("from-file")
    assert _read_secret(None, "X_SECRET", str(f)) == "from-env"


def test_read_secret_falls_back_to_file(tmp_path, monkeypatch):
    monkeypatch.delenv("X_SECRET", raising=False)
    f = tmp_path / "s"
    f.write_text("  from-file\n")
    assert _read_secret(None, "X_SECRET", str(f)) == "from-file"


def test_read_secret_none_when_missing(monkeypatch):
    monkeypatch.delenv("X_SECRET", raising=False)
    assert _read_secret(None, "X_SECRET", None) is None
    assert _read_secret(None, "X_SECRET", "/no/such/file") is None


def test_verify_google_id_token_rejects_empty():
    assert verify_google_id_token("", "client-id") is None
    assert verify_google_id_token(None, "client-id") is None


def test_verify_google_id_token_rejects_garbage():
    # A non-JWT string must not raise; just return None.
    assert verify_google_id_token("not-a-jwt", "client-id") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
