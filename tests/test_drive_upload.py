"""drive_upload: folder ensure (idempotent) + PNG upload/permission/url, driven
by a fake Drive client (no network).

Run with: pytest tests/test_drive_upload.py -q
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import drive_upload as du  # noqa: E402


class _Req:
    def __init__(self, result):
        self._r = result

    def execute(self):
        return self._r


class _Files:
    def __init__(self, store):
        self.store = store

    def list(self, **kw):
        self.store["list_calls"].append(kw)
        return _Req({"files": list(self.store["existing"])})

    def create(self, body=None, media_body=None, fields=None,
               supportsAllDrives=None):
        self.store["create_calls"].append(
            {"body": body, "media": media_body, "sad": supportsAllDrives})
        fid = f"id{len(self.store['created'])}"
        rec = {"id": fid, "webViewLink": f"https://drive/view/{fid}"}
        self.store["created"].append(rec)
        return _Req(rec)


class _Perms:
    def __init__(self, store):
        self.store = store

    def create(self, **kw):
        self.store["perm_calls"].append(kw)
        return _Req({"id": "perm0"})


class FakeDrive:
    def __init__(self, existing=None):
        self.store = {"existing": existing or [], "created": [],
                      "list_calls": [], "create_calls": [], "perm_calls": []}

    def files(self):
        return _Files(self.store)

    def permissions(self):
        return _Perms(self.store)


def test_ensure_folder_creates_when_absent():
    d = FakeDrive(existing=[])
    out = du.ensure_folder(d, "2026-W34", "PARENT")
    assert out["id"] == "id0"
    assert len(d.store["create_calls"]) == 1
    body = d.store["create_calls"][0]["body"]
    assert body["mimeType"] == du._FOLDER_MIME
    assert body["parents"] == ["PARENT"]
    # list query must scope to the parent + a folder mime + not-trashed
    q = d.store["list_calls"][0]["q"]
    assert "PARENT" in q and du._FOLDER_MIME in q and "trashed = false" in q


def test_ensure_folder_reuses_when_present():
    d = FakeDrive(existing=[{"id": "existing1", "webViewLink": "http://x"}])
    out = du.ensure_folder(d, "2026-W34", "PARENT")
    assert out["id"] == "existing1"
    assert d.store["create_calls"] == []          # no duplicate folder


def test_upload_png_public(tmp_path):
    png = tmp_path / "fig.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    d = FakeDrive()
    out = du.upload_png(d, str(png), "FOLDER")
    assert out["id"] == "id0"
    assert out["image_url"] == "https://drive.google.com/uc?export=view&id=id0"
    assert d.store["create_calls"][0]["body"]["parents"] == ["FOLDER"]
    # anyone-reader grant so =IMAGE can fetch it unauthenticated
    assert d.store["perm_calls"] and \
        d.store["perm_calls"][0]["body"] == {"role": "reader", "type": "anyone"}


def test_upload_png_private_skips_permission(tmp_path):
    png = tmp_path / "fig.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    d = FakeDrive()
    du.upload_png(d, str(png), "FOLDER", make_public=False)
    assert d.store["perm_calls"] == []


def test_upload_png_survives_permission_failure(tmp_path):
    png = tmp_path / "fig.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    d = FakeDrive()

    def boom(**kw):
        raise RuntimeError("org policy blocks anyone")
    d.permissions = lambda: type("P", (), {"create": staticmethod(boom)})()
    out = du.upload_png(d, str(png), "FOLDER")     # must not raise
    assert out["id"] == "id0"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
