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
import os
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


# --------------------------------------------------------------------- #
#  OAuth (personal Google account) path
#
#  A service account can't own files on a PERSONAL (non-Workspace) Drive -- it
#  has no storage quota. To publish into a personal Gmail Drive the uploads must
#  authenticate AS THE USER via OAuth: files are then owned by the user and count
#  against their 15 GB. The one-time browser consent is done interactively with
#  ``--authorize`` (below); the daemon only LOADS + refreshes the saved token and
#  never opens a browser. Publish the OAuth consent screen to "In production" or
#  Google expires the refresh token after 7 days.
# --------------------------------------------------------------------- #

def _project_path(rel: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    return rel if os.path.isabs(rel) else os.path.join(root, rel)


def _save_token(token_file: str, creds) -> None:
    os.makedirs(os.path.dirname(token_file) or ".", exist_ok=True)
    with open(token_file, "w", encoding="utf-8") as f:
        f.write(creds.to_json())


def _oauth_creds(token_file: str):
    """Load + (silently) refresh the stored user OAuth credentials. Raises with a
    clear next step if no valid token exists -- the daemon must NEVER try to open
    a browser."""
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    if not os.path.exists(token_file):
        raise RuntimeError(
            f"No Drive OAuth token at {token_file}. Run once on a machine with a "
            f"browser: python -m src.utils.drive_upload --authorize")
    creds = Credentials.from_authorized_user_file(token_file, _DRIVE_SCOPES)
    if creds.valid:
        return creds
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_token(token_file, creds)
        return creds
    raise RuntimeError(
        f"Drive OAuth token at {token_file} is invalid/expired without a refresh "
        f"token. Re-run: python -m src.utils.drive_upload --authorize "
        f"(and set the OAuth consent screen to 'In production').")


def drive_from_config(auth_mode: str, *, sa_path: str = "",
                      token_file: str = ""):
    """Build a Drive client for the configured auth mode: 'oauth' (personal
    account, via a stored user token) or 'service_account' (Shared Drive)."""
    mode = (auth_mode or "service_account").lower()
    if mode == "oauth":
        from googleapiclient.discovery import build
        assert token_file, "google_drive.token_file required for oauth mode"
        return build("drive", "v3", credentials=_oauth_creds(token_file),
                     cache_discovery=False)
    return _drive_api(sa_path)


def authorize(client_file: str, token_file: str) -> str:
    """One-time interactive OAuth consent (opens a browser) -> saves the refresh
    token to *token_file*. Run from a machine with a browser, not the daemon."""
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_secrets_file(client_file, _DRIVE_SCOPES)
    creds = flow.run_local_server(port=0)
    _save_token(token_file, creds)
    return token_file


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


def _main(argv=None) -> int:
    """`python -m src.utils.drive_upload --authorize` : one-time OAuth consent to
    upload into a personal Google Drive as yourself. Reads the client-secret +
    token paths from config.google_drive unless overridden."""
    import argparse

    import yaml

    ap = argparse.ArgumentParser(description="Drive OAuth authorize (one-time)")
    ap.add_argument("--authorize", action="store_true")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--client-file", default=None,
                    help="OAuth client-secret JSON (Desktop app)")
    ap.add_argument("--token-file", default=None)
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as fh:
        gd = (yaml.safe_load(fh).get("google_drive", {}) or {})
    client = args.client_file or gd.get("oauth_client_file")
    token = (args.token_file or gd.get("token_file")
             or "secrets/drive_oauth_token.json")
    if not args.authorize:
        print("Pass --authorize to run the one-time browser consent.")
        return 0
    if not client:
        print("Set google_drive.oauth_client_file (or --client-file) to your "
              "downloaded OAuth Desktop-app client JSON first.")
        return 2
    out = authorize(_project_path(client), _project_path(token))
    print(f"Saved Drive OAuth token -> {out}")
    print("If the token stops working after ~7 days, set the OAuth consent "
          "screen to 'In production'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
