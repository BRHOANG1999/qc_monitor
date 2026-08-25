"""Google Drive v3 upload helper for the weekly stimulus-stability figures.

Mirrors ``sheets_write._sheets_api_rw``: one lazily-built, path-cached client from
the SAME shared service-account JSON, only with a Drive scope added. Provides just
what the weekly job needs -- create-or-get a folder, upload a PNG, make it
link-viewable, and hand back a URL usable in a Sheets ``=IMAGE(...)`` cell.

Two hard constraints baked in (see docs / the plan):
  * A bare service account has NO personal-Drive storage quota, so the target
    must live in a **Shared Drive**; every call passes ``supportsAllDrives=True``
    (and list calls ``includeItemsFromAllDrives=True``) or it 404s / quota-fails.
  * ``=IMAGE`` fetches the file UNAUTHENTICATED, so an uploaded PNG needs an
    ``anyone: reader`` permission before its ``uc?export=view`` URL will load.

The Drive API must be enabled in the SA's GCP project and the Shared Drive folder
shared with the SA's ``client_email`` as Content Manager (one-time setup).
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("qc_monitor.utils.drive_upload")

# drive.file = least privilege: the SA may only touch files IT creates. Enough
# for a write-only publisher; use full "/auth/drive" only to manage pre-existing
# files (we don't).
_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.file"]
_FOLDER_MIME = "application/vnd.google-apps.folder"

_drive_cache: dict = {}
_drive_lock = threading.Lock()


def _drive_api(service_account_file: str):
    """Lazy-build a Drive v3 client, cached by SA path (like _sheets_api_rw)."""
    assert service_account_file, "service_account_file required"
    with _drive_lock:
        svc = _drive_cache.get(service_account_file)
        if svc is not None:
            return svc
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    creds = Credentials.from_service_account_file(
        service_account_file, scopes=_DRIVE_SCOPES)
    svc = build("drive", "v3", credentials=creds, cache_discovery=False)
    with _drive_lock:
        _drive_cache[service_account_file] = svc
    return svc


def _escape(name: str) -> str:
    """Escape a name for a Drive query string literal (single quotes + backslash)."""
    return str(name).replace("\\", "\\\\").replace("'", "\\'")


def ensure_folder(drive, name: str, parent_id: str) -> dict:
    """Create-or-get folder *name* under *parent_id*; return ``{id, webViewLink}``.

    Idempotent: an existing non-trashed folder of the same name is reused so
    re-running a week doesn't spawn duplicate folders."""
    assert drive is not None and name and parent_id, "drive, name, parent required"
    q = (f"name = '{_escape(name)}' and '{_escape(parent_id)}' in parents "
         f"and mimeType = '{_FOLDER_MIME}' and trashed = false")
    resp = drive.files().list(
        q=q, fields="files(id, webViewLink)", spaces="drive",
        supportsAllDrives=True, includeItemsFromAllDrives=True,
        corpora="allDrives").execute()
    hits = resp.get("files") or []
    if hits:
        return {"id": hits[0]["id"], "webViewLink": hits[0].get("webViewLink", "")}
    made = drive.files().create(
        body={"name": name, "mimeType": _FOLDER_MIME, "parents": [parent_id]},
        fields="id, webViewLink", supportsAllDrives=True).execute()
    return {"id": made["id"], "webViewLink": made.get("webViewLink", "")}


def upload_png(drive, path: str, parent_id: str, *, name: str | None = None,
               make_public: bool = True, throttle_sec: float = 0.0) -> dict:
    """Upload PNG *path* into *parent_id*; return ``{id, webViewLink, image_url}``.

    ``image_url`` is the ``uc?export=view`` form that renders in a Sheets
    ``=IMAGE(...)`` cell (needs the anyone-reader grant, applied when
    *make_public*). A failed public grant is logged, not raised -- the file still
    uploaded and org members can open ``webViewLink``."""
    from googleapiclient.http import MediaFileUpload
    assert drive is not None and path and parent_id, "drive, path, parent required"
    fname = name or path.replace("\\", "/").rsplit("/", 1)[-1]
    media = MediaFileUpload(path, mimetype="image/png", resumable=False)
    f = drive.files().create(
        body={"name": fname, "parents": [parent_id]}, media_body=media,
        fields="id, webViewLink", supportsAllDrives=True).execute()
    fid = f["id"]
    if make_public:
        try:
            drive.permissions().create(
                fileId=fid, body={"role": "reader", "type": "anyone"},
                supportsAllDrives=True).execute()
        except Exception as e:  # noqa: BLE001 -- non-fatal; link still works internally
            logger.warning("could not make %s public (=IMAGE may not load): %s",
                           fname, e)
    if throttle_sec > 0:
        time.sleep(throttle_sec)
    return {"id": fid, "webViewLink": f.get("webViewLink", ""),
            "image_url": f"https://drive.google.com/uc?export=view&id={fid}"}
