"""Generic MATLAB subprocess bridge for calling .m scripts via matlab -batch."""

import subprocess
import json
import tempfile
import os
import logging
from pathlib import Path

logger = logging.getLogger("qc_monitor.matlab_bridge")

MATLAB_EXE_DEFAULT = "matlab"
PIPELINE_SCRIPT_DIR = str(Path(__file__).parent.parent.parent / "matlab")


def run_matlab_batch(script: str, timeout: int = 600,
                     matlab_exe: str = MATLAB_EXE_DEFAULT) -> subprocess.CompletedProcess:
    """Run a MATLAB script via matlab -batch."""
    cmd = [matlab_exe, "-batch", script]
    logger.info("MATLAB cmd: %s", " ".join(cmd))

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        cwd=PIPELINE_SCRIPT_DIR
    )

    if result.returncode != 0:
        logger.error("MATLAB failed (rc=%d): %s", result.returncode, result.stderr[:500])
    else:
        logger.info("MATLAB completed successfully")

    return result


def run_pipeline(input_file: str, matlab_exe: str = MATLAB_EXE_DEFAULT,
                 timeout: int = 600) -> dict:
    """Run the full analysis pipeline on a single .mat file.

    Returns a dict with results from run_pipeline.m, or an error dict.
    """
    # Create temp output path for JSON results
    fd, output_json = tempfile.mkstemp(suffix=".json", prefix="qc_pipeline_")
    os.close(fd)

    # Normalize paths for MATLAB (forward slashes)
    input_norm = input_file.replace("\\", "/")
    output_norm = output_json.replace("\\", "/")
    pipeline_dir = PIPELINE_SCRIPT_DIR.replace("\\", "/")

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
        if os.path.exists(output_json):
            try:
                os.unlink(output_json)
            except OSError:
                pass
