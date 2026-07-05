"""Tests for the transfer-impedance drift alert (rules.check_impedance_shift)."""

from src.alerting.rules import AlertRuleEngine, _drift_stat


class _FakeStore:
    def __init__(self, series, last_alert=None, active=None):
        self._series = series
        self._last_alert = last_alert
        # Default: every channel is "active" so existing tests are unaffected.
        self._active = active if active is not None else set(series.keys())
        self.inserted = []

    def impedance_series_by_channel(self, exclude=None):
        return self._series

    def active_impedance_channel_keys(self):
        return self._active

    def get_last_alert_time(self, alert_type):
        return self._last_alert

    def insert_alert(self, *args):
        self.inserted.append(args)


class _FakeEmailer:
    def __init__(self):
        self.sent = []

    def send(self, subject, body, severity):
        self.sent.append((subject, body, severity))


def _engine(series, last_alert=None, active=None, **rule_overrides):
    rules = {"impedance_shift_pct": 40, "impedance_shift_pct_critical": 75,
             "impedance_baseline_window": 10, "impedance_min_history": 4}
    rules.update(rule_overrides)
    cfg = {"alerting": {"rules": rules}}
    store = _FakeStore(series, last_alert, active)
    return AlertRuleEngine(store, _FakeEmailer(), cfg), store


def _rows(vals):
    """Access-resistance series rows (oldest→newest)."""
    return [{"file_id": i, "chunk_datetime": f"2026_07_0{i}__00_00_00",
             "access_r_kohm": v}
            for i, v in enumerate(vals, start=1)]


def test_drift_stat_basic():
    st = _drift_stat([1.0, 1.0, 1.0, 1.5], window=10, min_history=4)
    assert st["baseline"] == 1.0
    assert st["latest"] == 1.5
    assert st["drift_pct"] == 50.0


def test_drift_stat_needs_min_history():
    assert _drift_stat([1.0, 1.5], window=10, min_history=4) is None


def test_alert_fires_on_drift():
    series = {("BCH110", "BCH110SLM"): _rows([1.0, 1.0, 1.0, 1.6])}  # +60%
    eng, store = _engine(series)
    eng.check_impedance_shift()
    assert len(store.inserted) == 1
    alert_type, severity, message = store.inserted[0][:3]
    assert alert_type == "impedance_shift"
    assert severity == "warning"
    assert "BCH110SLM" in message and "access resistance" in message


def test_alert_critical_on_large_drift():
    series = {("BCH062", "BCH062SR"): _rows([1.0, 1.0, 1.0, 2.0])}  # +100%
    eng, store = _engine(series)
    eng.check_impedance_shift()
    assert store.inserted[0][1] == "critical"


def test_no_alert_when_stable():
    series = {("BCH110", "BCH110SLM"): _rows([1.0, 1.0, 1.0, 1.02])}  # +2%
    eng, store = _engine(series)
    eng.check_impedance_shift()
    assert store.inserted == []


def test_no_alert_below_min_history():
    series = {("BCH110", "BCH110SLM"): _rows([1.0, 2.0])}  # only 2 points
    eng, store = _engine(series)
    eng.check_impedance_shift()
    assert store.inserted == []


def test_inactive_channel_not_alerted():
    # A drifting channel that is NOT in the most-recent session is skipped.
    series = {("BCH111", "BCH111SR"): _rows([1.0, 1.0, 1.0, 1.6])}  # +60%
    eng, store = _engine(series, active={("BCH062", "BCH062SR")})
    eng.check_impedance_shift()
    assert store.inserted == []


def test_rate_limited(monkeypatch):
    from datetime import datetime
    series = {("BCH110", "BCH110SLM"): _rows([1.0, 1.0, 1.0, 1.6])}
    eng, store = _engine(series, last_alert=datetime.now())  # just fired
    eng.check_impedance_shift()
    assert store.inserted == []  # suppressed by rate limit


# --------------------------------------------------------------------- #
#  Current-delivery fidelity (two-edge ratio) — check_current_sag
# --------------------------------------------------------------------- #

def _fid_rows(vals):
    """Current-fidelity series rows (oldest→newest); vals are signed %."""
    return [{"file_id": i, "chunk_datetime": f"2026_07_0{i}__00_00_00",
             "current_fidelity_pct": v}
            for i, v in enumerate(vals, start=1)]


def test_current_sag_fires_when_reversal_edge_sags():
    # Rolling-median well below the -15% default → sagging.
    series = {("BCH110", "BCH110SLM"): _fid_rows([-20, -22, -25, -30])}
    eng, store = _engine(series)
    eng.check_current_sag()
    assert len(store.inserted) == 1
    alert_type, severity, message = store.inserted[0][:3]
    assert alert_type == "current_sag"
    assert severity == "warning"
    assert "BCH110SLM" in message


def test_no_current_sag_when_edges_agree():
    # Fidelity ≈ 0 (edges agree) → current honest, no alert.
    series = {("BCH062", "BCH062SR"): _fid_rows([0.5, -0.3, 0.1, 0.4])}
    eng, store = _engine(series)
    eng.check_current_sag()
    assert store.inserted == []


def test_no_current_sag_above_min_history():
    series = {("BCH110", "BCH110SLM"): _fid_rows([-30, -40])}  # 2 points
    eng, store = _engine(series)
    eng.check_current_sag()
    assert store.inserted == []


def test_current_sag_scoped_to_active():
    series = {("BCH111", "BCH111SR"): _fid_rows([-20, -25, -30, -35])}
    eng, store = _engine(series, active={("BCH062", "BCH062SR")})
    eng.check_current_sag()
    assert store.inserted == []


def test_current_sag_transient_dip_not_flagged():
    # One noisy edge, but the rolling median stays above the threshold.
    series = {("BCH062", "BCH062SR"): _fid_rows([1.0, 0.5, -40, 0.8])}
    eng, store = _engine(series)
    eng.check_current_sag()
    assert store.inserted == []
