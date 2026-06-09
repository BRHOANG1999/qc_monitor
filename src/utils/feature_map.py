"""MATLAB feature-name -> DB column name mapping.

The Tier-2 pipeline (run via ``src/utils/matlab_bridge.run_pipeline``)
emits per-epoch features keyed by struct field names like
``Peak_Amplitude``, ``Log_AUC_``, etc. The Python side persists those
to ``evoked_features`` columns named ``peak_amplitude``, ``log_auc``,
... -- the names are similar but the contract is exact and there's no
algorithmic translation between them.

Keeping the map next to the MATLAB bridge (rather than buried inside
the dispatcher) makes the rename surface explicit: when a MATLAB
struct field name changes, this is the only Python file that needs an
edit.
"""

from __future__ import annotations


# MATLAB struct field name -> evoked_features column name.
_FEATURE_MAP: dict[str, str] = {
    "Line_Length": "line_length",
    "Log_AUC_": "log_auc",
    "Peak_Amplitude": "peak_amplitude",
    "Trough_Amplitude": "trough_amplitude",
    "Peak_to_Trough": "peak_to_trough",
    "RMS_Amplitude": "rms_amplitude",
    "Peak_Latency": "peak_latency_ms",
    "Trough_Latency": "trough_latency_ms",
    "Max_Slope": "max_slope",
    "Max_Slope_Time": "max_slope_time_ms",
    "Early_Area": "early_area",
    "Late_Area": "late_area",
    "Early_Late_Ratio": "early_late_ratio",
    "Recovery_Tau": "recovery_tau",
    "Recovery_Slope": "recovery_slope",
    "Template_Correlation": "template_correlation",
    "PCA_Recon_Error": "pca_recon_error",
    "Variance": "variance",
    "Autocorrelation": "autocorrelation",
    "AC_Width": "ac_width",
    "Exp_Fit_A": "exp_fit_a",
    "Sum_Power_Low": "sum_power_low",
    "Freq_Moment_Low": "freq_moment_low",
    "Sum_Power_High": "sum_power_high",
    "Freq_Moment_High": "freq_moment_high",
}


def feature_to_column(matlab_field: str) -> str | None:
    """Return the evoked_features column name for *matlab_field*, or
    None if the MATLAB pipeline emitted an unknown field name. The
    None result is the dispatcher's signal to silently drop that
    feature rather than crash the write -- MATLAB pipelines often
    grow new fields ahead of the Python side learning about them.
    """
    return _FEATURE_MAP.get(matlab_field)
