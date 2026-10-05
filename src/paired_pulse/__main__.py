"""CLI: python -m src.paired_pulse [--since YYYY-MM-DD] [--force] [--dry]

Default writes the figures + CSV + manifest to data/BCH111_paired_pulse/.
--dry only builds the matrix and prints PPR medians. --force rebuilds the cache.
"""

from __future__ import annotations

import datetime as _dt
import sys

from . import run as _run


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    since = None
    if "--since" in argv:
        since = _dt.datetime.strptime(argv[argv.index("--since") + 1], "%Y-%m-%d")
    force = "--force" in argv
    if "--periictal" in argv:
        _run.run_periictal(since=since, force=force)
    elif "--null" in argv:
        _run.run_null(since=since, force=force)
    elif "--continuous" in argv:
        _run.run_continuous(since=since, force=force)
    elif "--linbins" in argv:
        from . import linear_bins as _lb
        _lb.run(since=since, force=force)
    elif "--all" in argv:
        _run.run(since=since, force=force, apply=True)
        _run.run_periictal(since=since)
        _run.run_null(since=since)
        _run.run_continuous(since=since)
        from . import linear_bins as _lb
        _lb.run(since=since)
    else:
        _run.run(since=since, force=force, apply=("--dry" not in argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
