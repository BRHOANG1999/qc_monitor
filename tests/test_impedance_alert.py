"""Tests for the transfer-impedance drift alerts: access resistance
(rules.check_impedance_shift), current-delivery fidelity
(rules.check_current_sag), and slow-phase steady-state impedance / Z_ss
(rules.check_impedance_ss_shift)."""

from datetime import datetime, timedelta

from src.alerting.rules import AlertRuleEngine, _drift_stat


def _ts(hours_ago: float) -> str:
    """A chunk_datetime `hours_ago` before now (fresh by default, so the
    staleness guard doesn't suppress; pass a large value to make it stale)."""
    return (datetime.now() - timedelta(hours=hours_ago)).strftime(
        "%Y_%m_%d__%H_%M_%S")


class _FakeStore:
    def __init__(self, series, last_alert=None, active=None):
        self._series = series
        self._last_alert = last_alert
        # Default: every channel is "active" so existing tests are unaffected.
        self._active = active if active is not None else set(series.keys())
        self.inserted = []
        self._claims = set()

    def impedance_series_by_channel(self, exclude=None):
        return self._series

    def active_impedance_channel_keys(self):
        return self._active

    def get_last_alert_time(self, alert_type):
        return self._last_alert

    def insert_alert(self, *args):
        self.inserted.append(args)

    # notification_log dedup (per-condition daily) --------------------------
    def record_notification_sent(self, digest, sent_date):
        key = (digest, sent_date)
        if key in self._claims:
            return False
        self._claims.add(key)
        return True

    def notification_already_sent(self, digest, sent_date):
        return (digest, sent_date) in self._claims

    def release_notification_sent(self, digest, sent_date):
        self._claims.discard((digest, sent_date))


class _FakeEmailer:
    def __init__(self):
        self.sent = []

    def send(self, subject, body, severity, **kwargs):
        self.sent.append({"subject": subject, "body": body,
                          "severity": severity, **kwargs})
        return True


def _engine(series, last_alert=None, active=None, **rule_overrides):
    # Figures OFF by default in tests so the alert path doesn't render PNGs;
    # the figure test flips it on explicitly.
    rules = {"impedance_shift_pct": 40, "impedance_shift_pct_critical": 75,
             "impedance_baseline_window": 10, "impedance_min_history": 4,
             "impedance_alert_figures": False}
    rules.update(rule_overrides)
    cfg = {"alerting": {"rules": rules}}
    store = _FakeStore(series, last_alert, active)
    return AlertRuleEngine(store, _FakeEmailer(), cfg), store


def _rows(vals):
    """Access-resistance series rows (oldest→newest, all fresh)."""
    n = len(vals)
    return [{"file_id": i, "chunk_datetime": _ts(n - i), "access_r_kohm": v}
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
    """Current-fidelity series rows (oldest→newest, all fresh); vals signed %."""
    n = len(vals)
    return [{"file_id": i, "chunk_datetime": _ts(n - i),
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


# --------------------------------------------------------------------- #
#  Slow-phase steady-state impedance (Z_ss) — check_impedance_ss_shift
# --------------------------------------------------------------------- #

def _ss_rows(vals):
    """Slow-phase steady-state impedance series rows (oldest→newest, all fresh)."""
    n = len(vals)
    return [{"file_id": i, "chunk_datetime": _ts(n - i), "slow_ss_kohm": v}
            for i, v in enumerate(vals, start=1)]


def test_zss_alert_fires_on_drift():
    series = {("BCH110", "BCH110SLM"): _ss_rows([2.0, 2.0, 2.0, 3.2])}  # +60%
    eng, store = _engine(series)
    eng.check_impedance_ss_shift()
    assert len(store.inserted) == 1
    alert_type, severity, message = store.inserted[0][:3]
    assert alert_type == "impedance_ss_shift"
    assert severity == "warning"
    assert "BCH110SLM" in message and "steady-state" in message and "Z_ss" in message


def test_zss_alert_critical_on_large_drift():
    series = {("BCH062", "BCH062SR"): _ss_rows([1.0, 1.0, 1.0, 2.0])}  # +100%
    eng, store = _engine(series)
    eng.check_impedance_ss_shift()
    assert store.inserted[0][1] == "critical"


def test_zss_no_alert_when_stable():
    series = {("BCH110", "BCH110SLM"): _ss_rows([2.0, 2.0, 2.0, 2.04])}  # +2%
    eng, store = _engine(series)
    eng.check_impedance_ss_shift()
    assert store.inserted == []


def test_zss_no_alert_below_min_history():
    series = {("BCH110", "BCH110SLM"): _ss_rows([1.0, 2.0])}  # only 2 points
    eng, store = _engine(series)
    eng.check_impedance_ss_shift()
    assert store.inserted == []


def test_zss_scoped_to_active():
    series = {("BCH111", "BCH111SR"): _ss_rows([1.0, 1.0, 1.0, 1.6])}  # +60%
    eng, store = _engine(series, active={("BCH062", "BCH062SR")})
    eng.check_impedance_ss_shift()
    assert store.inserted == []


def test_zss_thresholds_inherit_access_r_when_unset():
    # _engine's config sets only the access-resistance thresholds; the Z_ss
    # alert must inherit 40/75 from them (so a +100% drift is critical).
    series = {("BCH062", "BCH062SR"): _ss_rows([1.0, 1.0, 1.0, 2.0])}
    eng, _store = _engine(series)
    assert eng.impedance_ss_shift_pct == 40.0
    assert eng.impedance_ss_shift_pct_critical == 75.0


def test_zss_independent_of_access_r():
    # Rows carry ONLY slow_ss_kohm: Z_ss drifts, but Rₐ can't be computed, so the
    # access-resistance alert stays silent while the Z_ss alert fires.
    series = {("BCH110", "BCH110SLM"): _ss_rows([2.0, 2.0, 2.0, 3.2])}  # +60%
    eng, store = _engine(series)
    eng.check_impedance_shift()          # Rₐ series all-None -> silent
    assert store.inserted == []
    eng.check_impedance_ss_shift()       # Z_ss drift -> fires
    assert len(store.inserted) == 1
    assert store.inserted[0][0] == "impedance_ss_shift"


def test_zss_rate_limited():
    from datetime import datetime
    series = {("BCH110", "BCH110SLM"): _ss_rows([2.0, 2.0, 2.0, 3.2])}
    eng, store = _engine(series, last_alert=datetime.now())  # just fired
    eng.check_impedance_ss_shift()
    assert store.inserted == []  # suppressed by rate limit


# --------------------------------------------------------------------- #
#  Staleness guard: a frozen (protocol-stopped) series must not re-fire
# --------------------------------------------------------------------- #

def _stale_rows(key, vals, days_old=30):
    """Series rows whose NEWEST measurement is `days_old` days old."""
    n = len(vals)
    return [{"file_id": i, "chunk_datetime": _ts(days_old * 24 + (n - i)),
             key: v}
            for i, v in enumerate(vals, start=1)]


def test_stale_series_suppresses_ra_alert():
    # +60% drift but the newest point is 30 days old -> no recent basis -> silent
    # (this is the BCH111SR "phantom": a frozen July series driving hourly email).
    series = {("BCH111", "BCH111SR"):
              _stale_rows("access_r_kohm", [1.0, 1.0, 1.0, 1.6])}
    eng, store = _engine(series)
    eng.check_impedance_shift()
    assert store.inserted == []


def test_stale_series_suppresses_zss_alert():
    series = {("BCH111", "BCH111SR"):
              _stale_rows("slow_ss_kohm", [2.0, 2.0, 2.0, 3.2])}
    eng, store = _engine(series)
    eng.check_impedance_ss_shift()
    assert store.inserted == []


# --------------------------------------------------------------------- #
#  Per-condition DAILY dedup: a standing condition emails once/day, not
#  once/hour; a new channel / escalation is a new signature and re-fires.
# --------------------------------------------------------------------- #

def test_standing_condition_deduped_same_day():
    series = {("BCH110", "BCH110SLM"): _rows([1.0, 1.0, 1.0, 1.6])}  # +60%
    eng, store = _engine(series)
    eng.check_impedance_shift()
    eng.check_impedance_shift()          # same condition, same day
    assert len(store.inserted) == 1      # 2nd suppressed by the daily dedup


def test_new_drifting_channel_fires_despite_dedup():
    eng, store = _engine({("A", "A1"): _rows([1.0, 1.0, 1.0, 1.6])})
    eng.check_impedance_shift()
    assert len(store.inserted) == 1
    # a SECOND channel now drifts -> different signature -> fires again same day
    store._series = {("A", "A1"): _rows([1.0, 1.0, 1.0, 1.6]),
                     ("B", "B1"): _rows([1.0, 1.0, 1.0, 1.6])}
    store._active = {("A", "A1"), ("B", "B1")}
    eng.check_impedance_shift()
    assert len(store.inserted) == 2


# --------------------------------------------------------------------- #
#  Z_ss saturation guard: at high charge the plateau collapses (Z_ss << Ra)
# --------------------------------------------------------------------- #

def test_zss_saturated_rows_dropped():
    # Z_ss all << 0.3*Ra (0.12) -> every row saturated -> no valid Z_ss -> silent,
    # even though the raw values "drift". (The 10 nC therapeutic-protocol case.)
    n = 4
    rows = [{"file_id": i, "chunk_datetime": _ts(n - i),
             "slow_ss_kohm": z, "access_r_kohm": 0.40}
            for i, z in enumerate([0.010, 0.010, 0.010, 0.005], start=1)]
    eng, store = _engine({("BCH111", "BCH111SR"): rows})
    eng.check_impedance_ss_shift()
    assert store.inserted == []


def test_zss_channel_skipped_when_mostly_saturated():
    # The BCH111SR-at-10nC case: most Z_ss rows are saturated (≪ 0.3·Rₐ) with a
    # few noisy bounces above the floor. The surviving rows "drift" hugely, but
    # the channel is mostly saturated -> skip it entirely (no alert).
    n = 10
    zs = [0.01, 0.01, 0.15, 0.01, 0.01, 0.21, 0.01, 0.01, 0.01, 0.30]  # 7/10 sat
    rows = [{"file_id": i, "chunk_datetime": _ts(n - i),
             "slow_ss_kohm": z, "access_r_kohm": 0.30}   # floor = 0.3·0.3 = 0.09
            for i, z in enumerate(zs, start=1)]
    eng, store = _engine({("BCH111", "BCH111SR"): rows})
    eng.check_impedance_ss_shift()
    assert store.inserted == []


def test_zss_fires_when_not_saturated_with_ra():
    # Z_ss > 0.3*Ra (not saturated) and drifting -> fires normally.
    n = 4
    rows = [{"file_id": i, "chunk_datetime": _ts(n - i),
             "slow_ss_kohm": z, "access_r_kohm": 0.40}
            for i, z in enumerate([0.30, 0.30, 0.30, 0.48], start=1)]  # +60%
    eng, store = _engine({("BCH111", "BCH111SR"): rows})
    eng.check_impedance_ss_shift()
    assert len(store.inserted) == 1
    assert store.inserted[0][0] == "impedance_ss_shift"


# --------------------------------------------------------------------- #
#  Figure embedded in the alert email
# --------------------------------------------------------------------- #

def test_alert_email_includes_trend_figure():
    series = {("BCH110", "BCH110SLM"): _rows([1.0, 1.0, 1.0, 1.6])}
    eng, store = _engine(series, impedance_alert_figures=True)
    eng.check_impedance_shift()
    assert len(store.inserted) == 1
    sent = eng.emailer.sent[0]
    assert sent.get("attachments")                       # >=1 rendered PNG path
    assert "cid:" in (sent.get("body_html") or "")       # referenced inline


# --------------------------------------------------------------------- #
#  Recent-window dominant charge: the trend follows a protocol change
# --------------------------------------------------------------------- #

def test_dominant_charge_follows_recent_protocol():
    from src.db.store import Store
    # Old protocol: many 2 nC rows, 50 days old (the all-time mode). New protocol:
    # fewer 10 nC rows, 2 days old. Recent-window mode = 10 nC -> keep only those.
    old = [{"chunk_datetime": _ts(50 * 24), "charge_nc": 2.0} for _ in range(20)]
    new = [{"chunk_datetime": _ts(2 * 24 - i), "charge_nc": 10.0}
           for i in range(5)]
    kept = Store._dominant_charge_only(old + new, recent_days=30)
    assert kept and all(r["charge_nc"] == 10.0 for r in kept)
    assert len(kept) == 5
