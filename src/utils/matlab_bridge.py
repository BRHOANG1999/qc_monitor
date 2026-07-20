"""Generic MATLAB subprocess bridge for calling .m scripts via matlab -batch.

Windows process-tree discipline (learned the hard way -- see the notes on
``run_matlab_batch``): ``matlab -batch`` is TWO processes (a thin launcher in
``MATLAB\\bin`` plus the real multi-GB engine in ``bin\\win64``). Killing the
launcher leaves the engine running, and capturing via PIPEs lets that orphan
hold the pipe's write handle open forever -- which deadlocks the parent inside
``communicate()`` and permanently wedges the single-threaded dispatcher. This
module therefore (a) redirects the child's output to TEMP FILES rather than
pipes, so there is no inherited pipe handle to deadlock on and the captured
output can't grow unbounded in the heap, and (b) puts the child in a Windows
Job Object with KILL_ON_JOB_CLOSE so the entire tree dies on timeout AND if
this process exits -- no orphaned engines, even across a service restart.
"""

import subprocess
import json
import tempfile
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("qc_monitor.matlab_bridge")

MATLAB_EXE_DEFAULT = "matlab"
PIPELINE_SCRIPT_DIR = str(Path(__file__).parent.parent.parent / "matlab")

# Bound how much MATLAB chatter we pull back into memory (we keep the tail).
_MAX_CAPTURE_BYTES = 64_000
# How long to wait for the tree to actually die after we kill it.
_KILL_GRACE_SEC = 30

_IS_WINDOWS = os.name == "nt"
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JobObjectExtendedLimitInformation = 9


def _make_kill_on_close_job():
    """A Windows Job Object that kills every assigned process when the job
    handle closes. Returns (handle, kernel32) or (None, None) when
    unavailable -- callers must tolerate None and fall back to a tree kill."""
    if not _IS_WINDOWS:
        return None, None
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class _IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in
                        ("ReadOperationCount", "WriteOperationCount",
                         "OtherOperationCount", "ReadTransferCount",
                         "WriteTransferCount", "OtherTransferCount")]

        class _BASIC(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                        ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class _EXTENDED(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", _BASIC),
                        ("IoInfo", _IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32.CreateJobObjectW.restype = wintypes.HANDLE
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None, None
        info = _EXTENDED()
        info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = k32.SetInformationJobObject(
            job, _JobObjectExtendedLimitInformation,
            ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            k32.CloseHandle(job)
            return None, None
        return job, k32
    except Exception as e:                       # noqa: BLE001 -- never block the run
        logger.debug("Job Object unavailable (%s); falling back to tree kill", e)
        return None, None


def _assign_to_job(job, k32, proc) -> bool:
    if not job or not k32:
        return False
    try:
        return bool(k32.AssignProcessToJobObject(job, int(proc._handle)))
    except Exception as e:                       # noqa: BLE001
        logger.debug("AssignProcessToJobObject failed: %s", e)
        return False


def _kill_tree(proc, job, k32) -> None:
    """Kill the child AND every descendant. The Job Object is authoritative
    (it survives re-parenting); psutil is the fallback, and plain kill() the
    last resort."""
    if job and k32:
        try:
            k32.TerminateJobObject(job, 1)
            return
        except Exception as e:                   # noqa: BLE001
            logger.debug("TerminateJobObject failed: %s", e)
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        victims = parent.children(recursive=True) + [parent]
        for p in victims:
            try:
                p.kill()
            except Exception:                    # noqa: BLE001
                pass
        psutil.wait_procs(victims, timeout=_KILL_GRACE_SEC)
        return
    except Exception as e:                       # noqa: BLE001
        logger.debug("psutil tree kill failed: %s", e)
    try:
        proc.kill()
    except Exception:                            # noqa: BLE001
        pass


def _tail_text(path: str, limit: int = _MAX_CAPTURE_BYTES) -> str:
    """Last *limit* bytes of a capture file, decoded leniently. Bounded so a
    chatty MATLAB run can never balloon the Python heap."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > limit:
                f.seek(size - limit)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except OSError:
        return ""


def run_matlab_batch(script: str, timeout: int = 600,
                     matlab_exe: str = MATLAB_EXE_DEFAULT) -> subprocess.CompletedProcess:
    """Run a MATLAB script via ``matlab -batch``.

    Output goes to temp FILES (not pipes) and the child runs inside a
    kill-on-close Job Object, so a hung MATLAB can neither deadlock us nor
    survive us. Raises ``subprocess.TimeoutExpired`` after killing the whole
    tree -- that handler is now actually reachable on Windows.
    """
    cmd = [matlab_exe, "-batch", script]
    logger.info("MATLAB cmd: %s", " ".join(cmd))

    out_fd, out_path = tempfile.mkstemp(suffix=".out", prefix="qc_matlab_")
    err_fd, err_path = tempfile.mkstemp(suffix=".err", prefix="qc_matlab_")
    job, k32 = _make_kill_on_close_job()
    proc = None
    try:
        with os.fdopen(out_fd, "wb") as fo, os.fdopen(err_fd, "wb") as fe:
            proc = subprocess.Popen(cmd, stdout=fo, stderr=fe,
                                    stdin=subprocess.DEVNULL,
                                    cwd=PIPELINE_SCRIPT_DIR)
            if _IS_WINDOWS and not _assign_to_job(job, k32, proc):
                logger.debug("MATLAB child not assigned to a Job Object; "
                             "relying on the psutil tree-kill fallback")
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.error("MATLAB timed out after %ss -- killing the "
                             "process tree (pid=%s)", timeout, proc.pid)
                _kill_tree(proc, job, k32)
                try:
                    proc.wait(timeout=_KILL_GRACE_SEC)
                except subprocess.TimeoutExpired:
                    logger.error("MATLAB pid=%s did not die after the tree "
                                 "kill; abandoning it", proc.pid)
                raise
        stdout = _tail_text(out_path)
        stderr = _tail_text(err_path)
        if proc.returncode != 0:
            logger.error("MATLAB failed (rc=%d): %s", proc.returncode,
                         stderr[:500])
        else:
            logger.info("MATLAB completed successfully")
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
    finally:
        # Closing the job handle kills anything still assigned to it, so an
        # abandoned engine cannot outlive this call.
        if job and k32:
            try:
                k32.CloseHandle(job)
            except Exception:                    # noqa: BLE001
                pass
        for tmp in (out_path, err_path):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def run_pipeline(input_file: str, matlab_exe: str = MATLAB_EXE_DEFAULT,
                 timeout: int = 600, config: Optional[dict] = None) -> dict:
    """Run the full analysis pipeline on a single .mat file.

    Parameters
    ----------
    input_file : str
        Path to the input .mat file.
    matlab_exe : str
        Path to the MATLAB executable.
    timeout : int
        Maximum seconds to wait for MATLAB to finish.
    config : dict, optional
        Pipeline configuration dict (evoked/criticality/feature settings
        from config.yaml).  Written to a temp JSON file and passed as the
        third argument to run_pipeline.m.

    Returns
    -------
    dict
        Parsed pipeline results containing:
        - features: dict of feature_name -> [epoch values]
        - criticality: dict with time_points, db_values, db_stds, sigmas
        - epoch_times: [stimulus times]
        - channel_names, eeg_channel, stim_channel
        - evoked_output_path, num_stimuli, num_traces
        - matlab_stdout, matlab_stderr
        Or an error dict with exit_status="error" and error_message.
    """
    # Create temp output path for JSON results
    fd, output_json = tempfile.mkstemp(suffix=".json", prefix="qc_pipeline_")
    os.close(fd)

    # Write config to a temp JSON file if provided
    config_json = None
    if config is not None:
        fd_cfg, config_json = tempfile.mkstemp(suffix=".json", prefix="qc_config_")
        os.close(fd_cfg)
        with open(config_json, "w") as f:
            json.dump(config, f)

    # Normalize paths for MATLAB (forward slashes)
    input_norm = input_file.replace("\\", "/")
    output_norm = output_json.replace("\\", "/")
    pipeline_dir = PIPELINE_SCRIPT_DIR.replace("\\", "/")

    if config_json is not None:
        config_norm = config_json.replace("\\", "/")
        script = (
            f"addpath('{pipeline_dir}'); "
            f"run_pipeline('{input_norm}', '{output_norm}', '{config_norm}')"
        )
    else:
        script = (
            f"addpath('{pipeline_dir}'); "
            f"run_pipeline('{input_norm}', '{output_norm}')"
        )

    try:
        proc = run_matlab_batch(script, timeout=timeout, matlab_exe=matlab_exe)

        if os.path.exists(output_json) and os.path.getsize(output_json) > 0:
            with open(output_json, "r") as f:
                result = json.load(f)
            result["matlab_stdout"] = proc.stdout[-500:] if proc.stdout else ""
            result["matlab_stderr"] = proc.stderr[-500:] if proc.stderr else ""
            return result
        else:
            return {
                "exit_status": "error",
                "error_message": f"No output JSON produced. stderr: {proc.stderr[:500]}",
                "matlab_stdout": proc.stdout[-500:] if proc.stdout else "",
                "matlab_stderr": proc.stderr[-500:] if proc.stderr else "",
            }
    except subprocess.TimeoutExpired:
        return {
            "exit_status": "error",
            "error_message": f"MATLAB timed out after {timeout}s",
        }
    except Exception as e:
        return {
            "exit_status": "error",
            "error_message": str(e),
        }
    finally:
        for tmp in (output_json, config_json):
            if tmp is not None and os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
