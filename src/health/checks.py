"""Read-only health probes. Each returns a list of HealthResult; no side effects, so
they are unit-testable against synthetic state/log/config fixtures.

A result's ``remedy`` names a bounded auto-fix the watchdog may apply (only
``restart:<service>`` today); ``None`` means detect-and-alert only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta

_LOG_TAIL_BYTES = 512 * 1024        # how much of the log to scan for recent errors
_TS_RX = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
_DAEMON = "QCMonitorDaemon"


@dataclass
class HealthResult:
    name: str
    status: str                      # 'ok' | 'warn' | 'fail'
    detail: str
    remedy: str | None = None        # e.g. 'restart:QCMonitorDaemon'
    severity: str = "warning"        # email severity when this fails
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------- #
#  Scheduled-job registry (state key -> config path under notifications)
# --------------------------------------------------------------------- #
@dataclass(frozen=True)
class Job:
    state_key: str
    cfg_path: tuple                  # keys under config['notifications']
    cadence: str                     # 'daily' | 'weekly'
    log_label: str                   # substring the scheduler logs for this job


JOBS: tuple = (
    Job("stim_stability_daily", ("stim_stability_daily",), "daily",
        "Stim-stability daily"),
    Job("stim_stability_weekly", ("stim_stability_weekly",), "weekly",
        "Stim-stability weekly"),
    Job("evoked_daily", ("evoked_daily",), "daily", "Evoked digest daily"),
    Job("evoked_resp_weekly", ("evoked_daily", "weekly"), "weekly",
        "Evoked digest weekly"),
    Job("evoked_windowed_daily", ("evoked_windowed",), "daily", "windowed"),
    Job("evoked_windowed_weekly", ("evoked_windowed", "weekly"), "weekly",
        "windowed"),
    Job("coverage", ("coverage",), "daily", "coverage"),
    Job("review_weekly", ("review_weekly",), "weekly", "Review digest"),
    Job("video_weekly", ("video_weekly",), "weekly", "Video digest"),
)


def _hcfg(config: dict) -> dict:
    return (config or {}).get("health", {}) or {}


def _dig(d: dict, path: tuple):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


# --------------------------------------------------------------------- #
#  1. Windows services up
# --------------------------------------------------------------------- #
def _service_status(name: str) -> str:
    """Running / Stopped / NotFound / Unknown via Get-Service (read-only)."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(Get-Service -Name '{name}' -ErrorAction SilentlyContinue).Status"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "Unknown"
    s = (out.stdout or "").strip()
    return s if s else "NotFound"


def check_services(config: dict) -> list[HealthResult]:
    names = _hcfg(config).get("services") or [_DAEMON, "QCMonitorDashboard"]
    out: list[HealthResult] = []
    for name in names:
        st = _service_status(name)
        if st == "Running":
            out.append(HealthResult(f"service:{name}", "ok", "Running"))
        elif st in ("NotFound", "Unknown"):
            out.append(HealthResult(f"service:{name}", "warn",
                                    f"status {st} (cannot probe)", remedy=None))
        else:
            out.append(HealthResult(
                f"service:{name}", "fail", f"status {st}",
                remedy=f"restart:{name}",
                severity="critical" if name == _DAEMON else "warning"))
    return out


# --------------------------------------------------------------------- #
#  2. Scheduled email jobs: fresh + not error-looping
# --------------------------------------------------------------------- #
def _read_state(config: dict) -> dict:
    path = (_dig(config, ("notifications",)) or {}).get("state_file") \
        or os.path.join("data", "notification_state.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _recent_error_labels(log_path: str, since: datetime) -> list[str]:
    """The ERROR lines in the log's tail newer than *since* (timestamped lines only).
    Used to spot a job stuck in a retry loop."""
    try:
        size = os.path.getsize(log_path)
        with open(log_path, "rb") as f:
            if size > _LOG_TAIL_BYTES:
                f.seek(size - _LOG_TAIL_BYTES)
            raw = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    hits = []
    for line in raw.splitlines():
        m = _TS_RX.match(line)
        if not m or "ERROR" not in line:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if ts >= since:
            hits.append(line)
    return hits


def _expected_daily(now: datetime, hour: int, grace_h: float) -> bool:
    """True once we are past today's scheduled hour + grace (so a fresh state should
    read today's date)."""
    due = now.replace(hour=int(hour), minute=0, second=0, microsecond=0)
    return now >= due + timedelta(hours=grace_h)


def check_email_jobs(config: dict, *, now: datetime | None = None) -> list[HealthResult]:
    now = now or datetime.now()
    hc = _hcfg(config)
    grace_h = float(hc.get("job_grace_hours", 3.0))
    err_min = int(hc.get("error_loop_min", 2))
    # Only count RECENT errors: a loop that stopped (e.g. after a restart) must not
    # keep flagging FAIL, or the auto-fix would release the claim and re-fire a
    # duplicate send. The window is short enough that only an ACTIVE loop (retries
    # every ~90 s) accumulates >= err_min hits.
    err_win_min = float(hc.get("error_window_min", 15.0))
    log_path = hc.get("log_file") or os.path.join("logs", "qc_monitor.log")
    state = _read_state(config)
    notif = _dig(config, ("notifications",)) or {}
    errs = _recent_error_labels(log_path, now - timedelta(minutes=err_win_min))
    out: list[HealthResult] = []
    for job in JOBS:
        cfg = _dig(notif, job.cfg_path) or {}
        if not cfg.get("enabled", False):
            continue
        last = str(state.get(job.state_key) or "")
        n_err = sum(1 for e in errs if job.log_label.lower() in e.lower())
        overdue = _job_overdue(job, cfg, last, now, grace_h)
        if n_err >= err_min:
            # Compound remedy: a mid-send crash can leave a stale dedup claim that a
            # plain restart does NOT clear (the claim then silently suppresses the
            # resend), so the fix is "release today's claim + restart the daemon".
            out.append(HealthResult(
                f"job:{job.state_key}", "fail",
                f"raising repeatedly ({n_err} errors/last hour); last sent {last or 'never'}",
                remedy=f"fixjob:{job.state_key}", severity="warning",
                meta={"errors": n_err, "last": last}))
        elif overdue:
            out.append(HealthResult(
                f"job:{job.state_key}", "warn",
                f"overdue (last sent {last or 'never'}, no errors logged)",
                remedy=None, meta={"last": last}))
        else:
            out.append(HealthResult(f"job:{job.state_key}", "ok",
                                    f"last sent {last or 'n/a'}"))
    return out


def _job_overdue(job: Job, cfg: dict, last: str, now: datetime,
                 grace_h: float) -> bool:
    """Daily: overdue if past today's hour+grace and state is not today. Weekly:
    overdue if the last send is more than 8 days ago."""
    try:
        last_d = datetime.fromisoformat(last).date() if last else None
    except ValueError:
        last_d = None
    if job.cadence == "daily":
        if not _expected_daily(now, int(cfg.get("hour", 8)), grace_h):
            return False
        return last_d is None or last_d < now.date()
    if last_d is None:
        return True
    return (now.date() - last_d).days > 8


# --------------------------------------------------------------------- #
#  3. Processing pipeline not stalled
# --------------------------------------------------------------------- #
def check_processing(config: dict, store) -> list[HealthResult]:
    if store is None:
        return [HealthResult("pipeline", "warn", "no store handle")]
    hc = _hcfg(config)
    err_max = int(hc.get("errored_max", 25))
    try:
        gaps = store.coverage_pipeline_gaps(cap=1)
    except Exception as e:                                # noqa: BLE001
        return [HealthResult("pipeline", "warn", f"gap query failed: {e}")]
    errored = int((gaps.get("errored") or {}).get("count", 0))
    stale = int((gaps.get("stale_processing") or {}).get("count", 0))
    if errored > err_max:
        return [HealthResult(
            "pipeline", "warn",
            f"{errored} files in error state (>{err_max}); {stale} processing",
            remedy=None, meta=gaps)]
    return [HealthResult("pipeline", "ok",
                         f"errored={errored}, processing={stale}")]


# --------------------------------------------------------------------- #
#  4. Disk, log size, daemon memory
# --------------------------------------------------------------------- #
def check_disk_and_logs(config: dict) -> list[HealthResult]:
    hc = _hcfg(config)
    out: list[HealthResult] = []
    db_path = (_dig(config, ("database",)) or {}).get("path") or "data"
    drive = os.path.splitdrive(os.path.abspath(db_path))[0] or os.path.abspath(db_path)
    min_free_gb = float(hc.get("min_free_gb", 20.0))
    try:
        import shutil
        free_gb = shutil.disk_usage(drive + os.sep).free / 1e9
        out.append(HealthResult(
            "disk", "fail" if free_gb < min_free_gb else "ok",
            f"{free_gb:.1f} GB free on {drive} (min {min_free_gb:.0f})",
            remedy=None, severity="warning"))
    except OSError as e:
        out.append(HealthResult("disk", "warn", f"disk_usage failed: {e}"))

    log_path = hc.get("log_file") or os.path.join("logs", "qc_monitor.log")
    max_log_mb = float(hc.get("max_log_mb", 500.0))
    try:
        mb = os.path.getsize(log_path) / 1e6
        out.append(HealthResult(
            "log_size", "warn" if mb > max_log_mb else "ok",
            f"{mb:.0f} MB (max {max_log_mb:.0f})"))
    except OSError:
        pass

    out.append(_daemon_memory(hc))
    return out


def _daemon_memory(hc: dict) -> HealthResult:
    """Total RSS of the daemon python process(es), flagged over the known-leak
    threshold (restart is the remedy)."""
    max_rss_gb = float(hc.get("max_rss_gb", 30.0))
    # The daemon runs "python main.py --no-dashboard"; every substring must be present
    # in the process cmdline to match it (and NOT the dashboard, which omits the flag).
    needles = hc.get("daemon_cmdline_match") or ["main.py", "--no-dashboard"]
    try:
        import psutil
    except ImportError:
        return HealthResult("daemon_memory", "warn", "psutil not installed")
    total = 0
    denied = 0
    for p in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            cl = " ".join(p.info.get("cmdline") or [])
            if all(n in cl for n in needles):
                total += p.info["memory_info"].rss
        except psutil.AccessDenied:
            denied += 1
        except psutil.NoSuchProcess:
            continue
    gb = total / 1e9
    if total == 0:
        why = (f"no daemon process matched ({denied} cmdlines unreadable "
               f"without elevation)" if denied else "no daemon process matched")
        return HealthResult("daemon_memory", "ok", why)
    return HealthResult(
        "daemon_memory", "fail" if gb > max_rss_gb else "ok",
        f"{gb:.1f} GB RSS (max {max_rss_gb:.0f})",
        remedy=f"restart:{_DAEMON}" if gb > max_rss_gb else None)


# --------------------------------------------------------------------- #
#  5. CPU runaway (e.g. orphaned MATLAB pegging cores for hours)
# --------------------------------------------------------------------- #
def _eval_cpu_runaway(current: dict, prev: dict, now_ts: float, *,
                      cores_thresh: float, total_hours_thresh: float) -> list[dict]:
    """Flag watched processes that are either pegging cores right now (average cores
    used between the previous snapshot and this one >= cores_thresh) or have burned an
    absurd amount of cumulative CPU (>= total_hours_thresh). Pure: *current*/*prev* map
    pid -> {name, cpu_s, create}. Returns one dict per flagged pid."""
    flagged = []
    for pid, cur in current.items():
        total_h = cur["cpu_s"] / 3600.0
        cores = None
        p = prev.get(str(pid)) or prev.get(pid)
        # Same process only: cpu_s must be monotonic (a reused pid resets it) and the
        # create time must match.
        if (p and abs(p.get("create", -1) - cur["create"]) < 1.0
                and cur["cpu_s"] >= p.get("cpu_s", 0.0)
                and now_ts > p.get("ts", now_ts)):
            cores = (cur["cpu_s"] - p["cpu_s"]) / (now_ts - p["ts"])
        pegging = cores is not None and cores >= cores_thresh
        bloated = total_h >= total_hours_thresh
        if pegging or bloated:
            flagged.append({
                "pid": int(pid), "name": cur["name"],
                "cores": (round(cores, 1) if cores is not None else None),
                "total_hours": round(total_h, 1),
                "reason": ("pegging cores" if pegging else "huge cumulative CPU")})
    return flagged


def _gather_cpu(names: list[str]) -> dict:
    """pid -> {name, cpu_s (cumulative user+system), create} for running processes
    whose name matches any of *names* (case-insensitive substring)."""
    try:
        import psutil
    except ImportError:
        return {}
    wanted = [n.lower() for n in names]
    out: dict = {}
    for p in psutil.process_iter(["name", "cpu_times", "create_time"]):
        try:
            nm = (p.info.get("name") or "")
            if not any(w in nm.lower() for w in wanted):
                continue
            ct = p.info.get("cpu_times")
            cpu_s = float(ct.user + ct.system) if ct else 0.0
            out[p.pid] = {"name": nm, "cpu_s": cpu_s,
                          "create": float(p.info.get("create_time") or 0.0)}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def check_cpu_runaway(config: dict, *, now: datetime | None = None) -> list[HealthResult]:
    """Flag watched processes (orphaned/runaway MATLAB by default) pegging cores for a
    long time. Alert-only: auto-killing a MATLAB worker can corrupt in-flight pipeline
    output, so this never carries a remedy -- it surfaces the PIDs for a human."""
    import json
    import os
    import time
    hc = _hcfg(config)
    names = hc.get("cpu_watch_names") or ["MATLAB"]
    cores_thresh = float(hc.get("cpu_runaway_cores", 1.5))
    total_h = float(hc.get("cpu_runaway_total_hours", 24.0))
    snap_path = hc.get("cpu_snapshot_file") or os.path.join(
        "data", "health_cpu_snapshot.json")
    now_ts = (now.timestamp() if now else time.time())
    current = _gather_cpu(names)
    try:
        with open(snap_path, encoding="utf-8") as f:
            prev = json.load(f)
    except (OSError, json.JSONDecodeError):
        prev = {}
    flagged = _eval_cpu_runaway(current, prev.get("procs", {}), now_ts,
                                cores_thresh=cores_thresh,
                                total_hours_thresh=total_h)
    snap = {"ts": now_ts,
            "procs": {str(pid): {**v, "ts": now_ts} for pid, v in current.items()}}
    try:
        os.makedirs(os.path.dirname(snap_path) or ".", exist_ok=True)
        tmp = snap_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(snap, f)
        os.replace(tmp, snap_path)
    except OSError:
        pass
    if not flagged:
        return [HealthResult("cpu_runaway", "ok",
                             f"no watched process ({'/'.join(names)}) over threshold")]
    out = []
    for f in flagged:
        cores = f"{f['cores']} cores" if f["cores"] is not None else "n/a cores"
        out.append(HealthResult(
            f"cpu_runaway:{f['name']}:{f['pid']}", "warn",
            f"{f['name']} pid {f['pid']} {f['reason']} "
            f"({cores}, {f['total_hours']} CPU-h cumulative)", remedy=None))
    return out


def run_all(config: dict, store=None, *, now: datetime | None = None) -> list[HealthResult]:
    """Every probe, concatenated. ``store`` optional (pipeline check degrades to a
    warn without it)."""
    results: list[HealthResult] = []
    results += check_services(config)
    results += check_email_jobs(config, now=now)
    results += check_processing(config, store)
    results += check_disk_and_logs(config)
    results += check_cpu_runaway(config, now=now)
    return results
