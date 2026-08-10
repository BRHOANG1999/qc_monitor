"""Local-first resolution for recording paths.

The .mat recordings live on a slow Tailscale SMB share (//100.106.104.22/...).
Reading one is minutes over the VPN, which drives both the dashboard GIL-hold
durations and the daemon throughput deficit. A scheduled robocopy job keeps a
ROLLING local mirror of the recent window (see sync_mirror.bat) on fast local
disk; this module maps a share path to its local mirror copy and returns that
when it exists, falling back to the share for anything not mirrored (older files
outside the window).

Deliberately env-driven, not config-object-driven: the dashboard reads .mat files
in spawned child processes (chunk_reader / snapshot_reader), and spawn children
inherit the parent's ENVIRONMENT but not its in-memory config. configure() stamps
the env once at daemon/dashboard boot; local_first() reads the env, so it works
identically in the parent and every child. A miss is a pure local stat -- it never
touches the share.
"""

from __future__ import annotations

import os

# Default share root that gets mirrored. Overridable via config/env for a
# different host. Compared case-insensitively against forward-slash-normalized
# paths.
_DEFAULT_SHARE_PREFIX = "//100.106.104.22/"

_ENV_ROOT = "QC_MIRROR_ROOT"        # e.g. D:\qc_mirror  (empty/unset => disabled)
_ENV_PREFIX = "QC_MIRROR_SHARE"     # e.g. //100.106.104.22/


def configure(config: dict | None) -> None:
    """Stamp the mirror env from config at boot (idempotent). Called in both the
    daemon (main.main) and the dashboard (create_app). With mirror.enabled false
    or unset, clears the env so local_first is a no-op."""
    m = ((config or {}).get("mirror", {}) or {})
    root = m.get("root") if m.get("enabled") else None
    if root:
        os.environ[_ENV_ROOT] = str(root)
        os.environ[_ENV_PREFIX] = str(m.get("share_prefix") or _DEFAULT_SHARE_PREFIX)
    else:
        os.environ.pop(_ENV_ROOT, None)


def local_first(path: str) -> str:
    """Return the local-mirror copy of *path* if it exists, else *path* unchanged.

    Pure local filesystem check (os.path.exists on the mapped local path) -- never
    stats the share. No-op when the mirror env is unset (mirror disabled)."""
    root = os.environ.get(_ENV_ROOT)
    if not root or not path:
        return path
    prefix = os.environ.get(_ENV_PREFIX, _DEFAULT_SHARE_PREFIX)
    norm = str(path).replace("\\", "/")
    if not norm.lower().startswith(prefix.lower()):
        return path                       # not a share path (mapped drive / local)
    rel = norm[len(prefix):].lstrip("/")
    local = os.path.join(root, *rel.split("/"))
    try:
        if os.path.exists(local):
            return local
    except OSError:
        pass
    return path                           # not mirrored yet -> read from the share


def is_enabled() -> bool:
    return bool(os.environ.get(_ENV_ROOT))
