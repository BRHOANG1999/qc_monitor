# Page loading in QC Monitor — strategy & method

*How the dashboard renders a tab without freezing, why the strategy is shaped
the way it is, and a card-by-card analysis of the Overview tab (the hardest
case). Last verified against the code on 2026-08-07.*

---

## 1. Audience & scope

This is for anyone touching the Dash dashboard (`src/dashboard/`) who needs to
add a tab, add a card, or debug "why is this page taking forever." It documents
the **loading pipeline** — how a click on a tab becomes a painted, populated
page — not the individual tab contents.

The one sentence to keep in mind:

> **Every slow load in this app is a *contention* problem, not a *slow-query*
> problem.** In isolation every query is milliseconds; under concurrency they
> became minutes. The entire strategy exists to keep concurrent work from
> piling up on a single shared SQLite file and a fixed thread pool.

---

## 2. The architecture that forces the strategy (the "why")

You cannot understand the loading design without the deployment shape, because
the design is a direct response to it.

```
              ┌──────────────────────  daemon process (main.py) ──────────────────────┐
              │                                                                        │
  browser ──► │  waitress pool (16 request threads)  ──►  render_tab / fill callbacks │
  (many       │            │                                     │                     │
   tabs open) │            └───────────────┐        ┌────────────┘                     │
              │                            ▼        ▼                                   │
              │                     ┌──────────────────────┐                           │
              │   watch loop  ────► │   ONE Store / SQLite   │ ◄──── ~6 writer threads  │
              │   (main loop)       │   (~10 GB, WAL mode,   │       (impedance, mass_  │
              │                     │    busy_timeout 30 s)  │        analyze, preictal,│
              │                     └──────────────────────┘        heartbeat, clips…) │
              └────────────────────────────────────────────────────────────────────────┘
```

Key facts that shape everything:

- **The dashboard is a thread inside the daemon**, not a separate service
  (`main.py` launches `run_dashboard` on a daemon thread). It shares **one
  `Store` and one SQLite file** with the daemon's watch loop and ~6 background
  writer threads.
- **SQLite allows exactly one writer at a time.** WAL mode lets readers run
  concurrently with a writer, but writers serialize on the file lock and wait
  out the 30 s `busy_timeout` before failing with `database is locked`.
- **`Store` is connection-per-operation** — every method opens, queries,
  commits, closes. There is no pool.
- **Serving is multi-threaded** (waitress pool / Flask `threaded=True`). Two
  requests — e.g. a slow tab render and a status poll — run on *different*
  threads simultaneously. This is what makes the live-progress readout possible
  (§5.2), and also what makes callback pile-ups dangerous (a slow callback that
  fires every 10 s stacks copies of itself across the pool until it starves).

The corollary: a card whose query is 400 ms alone can take **tens of minutes**
in production if a dozen copies of it are running concurrently on a
disk-saturated DB while the daemon writes a 12 M-row table. The fix is never
"make the query faster" first — it's **stop the copies from stacking**.

---

## 3. The loading strategy, layer by layer

Nine mechanisms, applied together. Each has a one-line principle, the code that
implements it, and the failure it prevents.

### 3.1 Progressive render — shell first, cards later

**Principle:** `render_tab` must return a *light shell* instantly; every heavy
card is a placeholder that a follow-up callback fills.

`render_tab` (`app.py:1306`) is the single callback bound to `tabs.value`. For a
heavy tab it returns a layout whose expensive slots are `_pending_placeholder`
stand-ins (`overview.py:137`) — a small `⏳ <label>` div — instead of the built
card. The page paints at once; each card appears when its own fill callback
lands. Without this, `render_tab` would block on ~12 synchronous DB-backed
builds and the browser would show nothing (a "freeze") until the slowest
finished.

The fills are driven by two signals:

- **`overview-mount`** — a **once-fire** `dcc.Interval` that fires ~once right
  after the shell mounts, kicking the initial fill.
- **`refresh-trigger`** — a global `dcc.Store` counter bumped by `on_refresh`
  (`app.py:783`) on the manual refresh button and on each `refresh` Interval
  tick. This drives ongoing auto-refresh.

### 3.2 Tab-gating — only build what's on screen

**Principle:** a fill callback must do nothing when its tab isn't visible.

The app is built with `suppress_callback_exceptions=True`, which means a
callback fires whenever its `Input`s change **even if its `Output`s aren't in
the current layout**. `refresh-trigger` is *always* mounted, so before gating,
every one of the ~19 refresh-driven callbacks ran its heavy DB work on every
tick no matter which tab you were on — a permanent background storm.

Every fill callback now takes `State("tabs", "value")` and early-returns
`no_update` when it isn't its tab (see all of `refresh_overview_dynamic`,
`_fill_home_grid`, `_fill_km_log`, `_fill_bsz_status`,
`refresh_overview_thumbnail` — each opens with `if current_tab != "overview":
return …`). Two header callbacks (health dot, update check) legitimately stay
global; they are TTL-cached instead (§3.4).

### 3.3 Single-flight — never let a build overlap itself

**Principle:** if a build is already running, the next trigger must **not**
start a second copy.

Implemented with a **non-blocking** lock: `Lock.acquire(blocking=False)`. If the
acquire fails, a rebuild is already in flight, so the callback bails with
`no_update` (or serves stale) instead of piling on.

- `_REFRESH_LOCK` (`overview.py:68`) guards the 6-card `refresh_overview_dynamic`.
- `_THUMB_LOCK` (`overview.py:134`) guards the evoked-thumbnail build.
- `_CachedBuilder` (§3.4) embeds the same non-blocking single-flight for the
  behavioral-seizure card and the two Google-Sheets cards.

This is the single most important mechanism. The behavioral-seizure card was a
~2 s build that ballooned to **~36 minutes per call** purely because, with no
guard, it fired every 10 s and stacked dozens of concurrent copies. Adding the
guard alone returned it to seconds.

### 3.4 Stale-while-revalidate TTL cache — serve last-good, refresh once

**Principle:** within a TTL, serve the cached card and don't rebuild at all;
when it expires, exactly **one** caller rebuilds while everyone else serves the
stale value.

Two implementations of the same idea:

- **`_CachedBuilder`** (`overview.py:71`) — server-side, per-card. `.get(build,
  label=…)`: fresh within TTL → return cached; expired → non-blocking single
  acquire, one thread rebuilds, others serve stale (or `no_update` on a cold
  cache). TTLs: `_BSZ_CACHE` 45 s, `_HOME_GRID_CACHE` / `_KM_LOG_CACHE` 300 s.
- **`_ttl_cached`** (`app.py:28`) — app-level, keyed. Used by the header health
  dot (20 s) and update check (60 s). Only blocks for the lock when there is
  **nothing** to serve; otherwise serves stale and lets the lock-holder refresh.

Both exist to prevent a **cache stampede** — the pattern where an entry expires
and a dozen concurrent callers all run the slow producer at once. The header dot
was doing exactly that (a dozen 20 s recomputes piling up); TTL + single-refresh
cut it from ~4.3 s to ~0.9 s.

### 3.5 Isolate slow / networked cards into their own callbacks

**Principle:** never bundle a card that can hang (network) with cards that are
always fast (local DB).

The Google-Sheets–backed **home grid** and **KM-log** cards were originally in
the same 8-output callback as the fast DB cards. When the Sheets API was
unreachable (VPN / `*.ts.net` routing), the whole callback blocked on the socket
timeout, so the *DB* cards (status pills, review queue) showed their placeholder
for as long as the *Sheets* read took — minutes. They're now split into
`_fill_home_grid` / `_fill_km_log` (`overview.py:4444`, `:4460`), each on its own
callback + 300 s cache, so a slow sheet can never hold a fast card hostage. The
Sheets client itself is built over `httplib2.Http(timeout=10 s)` so an
unreachable API **fails fast** to the stale cache instead of hanging
indefinitely (`src/utils/sheets.py`).

### 3.6 Cap payload size

**Principle:** don't serialize a million points to the browser.

The stim-artifact overlay is capped to the newest `_OVERLAY_MAX_FILES = 30`
files, each decimated to ~800 points (`overview.py:1341`) — it was ~420 k
points across 300 traces. The "last 24 recordings" (hist24) thumbnail is
computed on a **background worker** with a poll: a cache miss kicks the worker,
shows a `Computing…` placeholder, enables the poll Interval so the count
advances, then renders and stops polling (`refresh_overview_thumbnail`,
`overview.py:4598`). The default "mean" trace stays cheap and synchronous.

### 3.7 Pause when the tab is hidden

**Principle:** a backgrounded browser tab must not keep polling.

Auto-refresh is disabled unless **both** the toggle is on **and** the tab is
foregrounded. A `visibilitychange` listener (installed once, clientside) writes
`document.visibilityState === 'visible'` into the `page-visible` Store; a
clientside callback ANDs that with the toggle to set `refresh.disabled`
(`app.py:691`–`720`). No server round-trip, and hidden tabs stop generating
`refresh-trigger` load entirely.

### 3.8 Never block the request thread on a write

**Principle:** a page render must never wait on the SQLite write lock.

`render_tab` records navigation (`_activity.track`, `app.py:1321`) — a write to
`user_activity` on **every tab switch**. If that write blocked on the write lock
(held during the boot write-storm, up to 30 s), every tab switch would stall.
`activity.track` therefore does the INSERT **fire-and-forget on a daemon
thread** (`activity.py:46`) and returns immediately. Rule: any dashboard-side
write must never block the request thread.

### 3.9 Efficient queries underneath (the last layer, not the first)

Once contention is controlled, query shape still matters at the margin:

- `latest_zss_per_channel` and `latest_current_fidelity_per_channel` were
  rewritten from a per-row correlated `MAX()` subquery to a `ROW_NUMBER()`
  window (149–198 ms → ~2 ms, verified byte-identical).
- Added indexes on the hot path: `idx_video_qc_analyzed`, `idx_seizure_events_*`,
  `idx_system_health_ts`; `newest_health()` is an O(1) indexed read replacing a
  187 k-row scan.

Note the ordering: this is **§3.9, not §3.1.** Query tuning was the *smallest*
win. The plan measured it as ~400 ms → ~50 ms per tick — real, but trivial next
to the minutes recovered by single-flight and isolation.

---

## 4. The driver signals, end to end

What actually fires, in order, when you click a heavy tab:

1. **`tabs.value` changes** (click). The clientside status-pill callback fires
   instantly → amber `⏳ Loading <Tab>… 0.0s` (§5.1). `render_tab` starts on a
   request thread.
2. **`render_tab` returns the shell** (light; placeholders for heavy cards).
   `tab-content.children` lands → pill flips to green `● Ready`.
3. **`overview-mount` fires once** (shell mounted) → kicks each fill callback.
4. **`refresh-trigger` bumps** thereafter (manual button / `refresh` Interval)
   → fill callbacks re-run, gated to the visible tab and single-flighted.

The **mount + refresh double-fire on load** (both signals hit within ~120 ms) is
deliberately left harmless rather than prevented: the second fire finds the
single-flight lock held or the TTL cache fresh, so it's a no-op. (`on_refresh`
has no `prevent_initial_call`; the guards make it unnecessary.)

---

## 5. The loading-feedback layer (so slow never reads as frozen)

Three coordinated pieces answer the user's real question — *"is it working or
is it stuck?"*

### 5.1 Status pill (bottom-right, fixed, clientside)

A single clientside callback (`app.py:843`) drives a fixed pill:

- `tabs.value` → **amber** `⏳ Loading <Tab>… 0.0s` (fires before the server has
  built anything).
- `nav-load-tick` (200 ms Interval) → ticks the elapsed seconds; at **≥ 8 s** it
  appends `· still working, no need to refresh`.
- `tab-content.children` lands → **green** `● Ready`, which then **fades out**
  after 2.5 s. A persistent "Ready" while cards are still filling would mislead;
  the pill signals the *current view's* load and then gets out of the way.

It's clientside so it updates with zero server round-trips — critical, since the
server threads are exactly what's busy during a slow load.

### 5.2 Live stage readout (cross-thread)

`src/dashboard/nav_progress.py` is a tiny cross-thread channel. The building
thread calls `begin(user)` / `stage(text)` / `end()`; a **separate** poll
callback (`_nav_stage_detail`, `app.py:1426`) running on another server thread
reads `get(user)` and shows `▸ <stage>` under the pill. This works *because*
serving is multi-threaded — while `render_tab` blocks on thread A, thread B can
report what stage A is in. The publisher uses a thread-local for "who am I
building for" so deep call-tree code can call `stage()` without threading a user
id through every function. It's best-effort telemetry: every function swallows
its own exceptions so instrumentation can never break a render. The stage-poll
Interval is enabled **only while a tab is loading** (`app.py:915`), so it adds
zero round-trips at idle.

### 5.3 Per-card placeholders

Each heavy card ships a `⏳ <label>` placeholder (§3.1) so the user sees exactly
*which* cards are still coming, not a blank region.

---

## 6. The observability layer (so slowness is diagnosable)

When someone says "everything is slow," you need it to become "these three
labels dominate." That's `src/dashboard/perf.py`:

- A Flask `after_request` hook (`app.py:373`) times **every** callback
  (`/_dash-update-component`) and records it under `cb:<output>`. One hook covers
  all ~285 callbacks.
- Tab renders are timed under `tab:<Tab>`, Overview stages under
  `overview:<stage>`, via `_perf.Timer`.
- **System → Performance tab** (`tabs/perf_monitor.py`) reads `perf.top()` and
  ranks by Total (cumulative) or Max (worst single stall). **Reset** zeroes the
  counters so you can time one clean reproduction.
- Any callback over `dashboard.slow_callback_ms` (default 800) is also logged as
  `slow callback <ms>  cb:<label>`.

This is the tool that confirmed the fixes: `cb:overview-bsz-status` collapsing
from ~36 min to seconds, jobs-monitor 4795 → 504 ms, header-dot 4285 → 922 ms.

---

## 7. Deep dive — the Overview tab

Overview is the hardest page in the app and the reason most of this machinery
exists. It packs, on one screen: ~6 DB-backed status/quality cards, a review
queue, an impedance-trend family, two **Google-Sheets**-backed cards, and a
**`.mat`-backed** evoked thumbnail. Every category of slowness lives here.

### 7.1 The shell

`_overview_tab` (`overview.py:2227`) returns placeholders for each heavy slot:

| Slot id | Placeholder label | Source | Filled by |
|---|---|---|---|
| `overview-cards` | status pills | DB | `refresh_overview_dynamic` |
| `overview-bsz-status` | behavioral seizure status | DB (~3 k files) | `_fill_bsz_status` |
| `overview-impedance` | impedance trend | DB | `refresh_overview_dynamic` |
| `overview-zss` | stim-step consistency | DB | `refresh_overview_dynamic` |
| `overview-current-fidelity` | current fidelity | DB | `refresh_overview_dynamic` |
| `overview-region-drift` | region drift | DB | `refresh_overview_dynamic` |
| `overview-queue` | review queue | DB | `refresh_overview_dynamic` |
| `overview-km-log` | Kaplan–Meier log summary | **Sheets** | `_fill_km_log` |
| `overview-home-grid` | lab home grid | **Sheets** | `_fill_home_grid` |
| `overview-thumbnail` | evoked thumbnail | **`.mat`** | `refresh_overview_thumbnail` |

### 7.2 The fill callbacks — how each category is protected

- **`refresh_overview_dynamic`** (`overview.py:4415`) — the 6 fast DB cards, in
  one callback, behind `_REFRESH_LOCK` (single-flight) and tab-gated. Bundled
  because they're all sub-second local reads; there's no benefit to splitting
  them and a small benefit to sharing one thread.
- **`_fill_bsz_status`** (`overview.py:4477`) — its own callback so it builds in
  **parallel** (own thread) with the 6-card bundle, behind `_BSZ_CACHE` (45 s
  single-flight + TTL). This is the card that pile-up hit hardest.
- **`_fill_home_grid` / `_fill_km_log`** (`overview.py:4444`, `:4460`) — the
  Sheets cards, isolated (§3.5) behind 300 s caches so an unreachable API can't
  block the DB cards.
- **`refresh_overview_thumbnail`** (`overview.py:4598`) — behind `_THUMB_LOCK`,
  with a cheap **signature short-circuit**: the thumbnail only needs to change
  when a new recording arrives, so it returns `no_update` unless
  `f"{mode}:{store.max_processed_file_id()}"` differs from the last-drawn
  signature. hist24 runs on a background worker + poll (§3.6).

### 7.3 The measured diagnosis (why it was minutes)

From the read-only investigation (`~/.claude/plans/resilient-baking-hanrahan.md`),
production timings were in **minutes** while every isolated query was
milliseconds. Ranked causes and the fix applied to each:

| # | Cause | Prod cost | Fix | After |
|---|---|---|---|---|
| 1 | `_fill_bsz_status` pile-up (no single-flight, fires every 10 s) | ~36 min/call | `_BSZ_CACHE` single-flight + 45 s TTL | ~1.7 s |
| 2 | 8-card bundle hung on unreachable Google Sheets | 11–34 min | split Sheets cards out; `httplib2` 10 s timeout; 300 s cache | seconds |
| 3 | thumbnail pile-up + oversized figures (~420 k pts) | 1.6–3.8 min | `_THUMB_LOCK` + overlay cap 30 files/~800 pts + hist24 worker | 1.1–2.2 s |
| 4 | mount+refresh double-fire built everything twice on load | 2× work | subsumed by single-flight/TTL (§4) | 1× |
| 5 | correlated `MAX()` subqueries; duplicate per-tick queries | ~400 ms/tick | `ROW_NUMBER()` window rewrite | ~50 ms |
| 6 | missing indexes on `video_qc`, `seizure_events` | small | `CREATE INDEX` | negligible |

Whole-render figure: Overview `render_tab` went from ~30 s to ~47 ms warm (it now
returns only the shell).

### 7.4 The deeper root cause (the systemic tier)

The card-level fixes above are necessary but sit on top of two systemic issues
that were the *dominant* stall and are documented in the perf memory:

1. **DB write-lock contention.** `render_tab`'s per-switch write blocked on the
   SQLite write lock, held by a **boot write-storm**: `main.py` started every
   worker at t=0 with no stagger, and the impedance backfill re-read hundreds of
   non-stim files' multi-MB evoked-waveform blobs on every run — GBs of disk
   reads that saturated the disk so even dashboard *reads* took 18–23 s. Fixes:
   `activity.track` fire-and-forget (§3.8); `transaction()` uses `BEGIN
   IMMEDIATE`; a targeted `Store._WRITE_LOCK` serializes multi-statement writes
   and batched impedance upserts (reads never take it, so WAL stays concurrent);
   the impedance backfill now checks the cheap stim flag **before** the blob read
   and throttles between files; a periodic passive WAL checkpoint keeps the WAL
   from ballooning. **A blanket per-connection write-serializing wrapper was
   tried and reverted** — it funneled all writes through one choke point and
   risked a deadlock on any leaked connection. Keep the locks *targeted*.
2. **Boot worker storm** — the reason a restart didn't immediately help. Workers
   are now staggered (impedance sleeps 90 s before its first sweep; the
   mass-analyze auto-filter sweep is delayed ~120 s; the training import is
   chunked 50/transaction).

**The lesson, restated:** the freeze was *contention*, not slow SQL. Every
stage-1 query is < 0.02 s alone but took 15–33 s live because of the
double-build and global pollers competing for the thread pool on a
disk-saturated file.

---

## 8. Anti-patterns — what NOT to do

- ❌ **Don't do heavy work in `render_tab`.** Return a shell + placeholders and
  fill in a callback. A synchronous heavy render blocks the browser's first
  paint.
- ❌ **Don't add a `refresh-trigger` callback without tab-gating it.** It will
  run on every tick on every tab (`suppress_callback_exceptions=True`).
- ❌ **Don't add a periodic callback without single-flight.** If a build can take
  longer than its interval, un-guarded copies stack until the pool starves.
- ❌ **Don't bundle a network/Sheets card with DB cards.** Isolate anything that
  can hang, and give it a hard timeout + TTL cache.
- ❌ **Don't block the request thread on a write.** Fire-and-forget it.
- ❌ **Don't re-add a blanket write-serializing connection wrapper.** Targeted
  `Store._WRITE_LOCK` only.
- ❌ **Don't serialize huge figures to the browser.** Cap and decimate; offload
  the expensive variants to a worker + poll.
- ❌ **Don't leave a file/network read unbounded, especially under a lock.** A
  read with no timeout, guarded by single-flight, is the worst case: one hang
  wedges the lock forever and every later tick silently returns `no_update`.
  Deadline-bound it (`call_with_deadline`) — see §11.
- ❌ **Don't "fix slow" by optimizing the query first.** Confirm on the
  Performance tab whether it's cost or contention — it's almost always
  contention.

---

## 9. Checklist — adding a new heavy card or tab

1. Does `render_tab` stay light? Put the heavy build behind a placeholder +
   fill callback (§3.1).
2. Does the fill callback early-return `no_update` off its tab (§3.2)?
3. Can it overlap itself? Add a non-blocking single-flight guard, or wrap the
   build in a `_CachedBuilder` (§3.3–3.4).
4. Does it read the network (Sheets) or a file (`.mat`)? Isolate it, give it a
   **hard deadline** (`call_with_deadline`, §11) + TTL, and consider a background
   worker + poll (§3.5–3.6). A read behind a single-flight lock **must** be
   bounded, or one hang freezes the card forever.
5. Does it write? Fire-and-forget off the request thread (§3.8).
6. Is the payload bounded? Cap/decimate points (§3.6).
7. Verify on **System → Performance** that its `cb:` label is sub-second under a
   real reproduction (§6).

---

## 10. File & symbol reference

| Concern | File · symbol |
|---|---|
| Tab render entry point | `src/dashboard/app.py` · `render_tab` (1306) |
| Refresh signal | `app.py` · `on_refresh` (783), `refresh-trigger` Store (629) |
| Status pill (clientside) | `app.py` (843) + `nav-load-tick` Interval (471) |
| Pause-when-hidden | `app.py` (691–720), `page-visible` Store (567) |
| App-level TTL cache | `app.py` · `_ttl_cached` (28) |
| Perf hook | `app.py` · `_perf_after` (373) |
| Live stage channel | `src/dashboard/nav_progress.py` |
| Perf ledger | `src/dashboard/perf.py`; tab `tabs/perf_monitor.py` |
| Progressive placeholder | `tabs/overview.py` · `_pending_placeholder` (137) |
| Single-flight + TTL builder | `tabs/overview.py` · `_CachedBuilder` (71) |
| Overview caches / locks | `overview.py` · `_REFRESH_LOCK` (68), `_BSZ_CACHE` (119), `_HOME_GRID_CACHE`/`_KM_LOG_CACHE` (127–128), `_THUMB_LOCK` (134) |
| Overview shell | `overview.py` · `_overview_tab` (2227) |
| Overview fills | `overview.py` · `refresh_overview_dynamic` (4415), `_fill_home_grid` (4444), `_fill_km_log` (4460), `_fill_bsz_status` (4477), `refresh_overview_thumbnail` (4598) |
| Fire-and-forget write | `src/dashboard/activity.py` · `track` (22) |
| Sheets fast-fail timeout | `src/utils/sheets.py` |
| Deadline helper | `src/utils/deadline.py` · `call_with_deadline` |
| Watchdog guard | `src/dashboard/single_flight.py` · `SingleFlight`, `STALL_SEC` |
| Card freshness / stalled surfacing | `overview.py` · `_freshness_footer`, `_guarded_card`, `_stalled_banner`, `_thumb_unavailable` |
| Measured diagnosis | `~/.claude/plans/resilient-baking-hanrahan.md` |

*Line numbers drift as the files change; treat them as starting points and
confirm the symbol.*

---

## 11. Single-flight needs a watchdog (2026-08 hardening)

Single-flight (§3.3) has a failure mode that took a live incident to surface:
**Overview cards spun on `⏳` for an hour with zero errors anywhere.** Root cause
— the evoked-thumbnail's fallback read a session `.mat` off a network share with
**no timeout, under `_THUMB_LOCK`**. When the share hung, the lock was never
released, so every later refresh tick did `try_begin()` → fail → `no_update`,
forever. The guard worked exactly as designed; that was the problem — a wedged
build is *supposed* to make others back off, but with no bound and no signal it
made them back off permanently and silently.

Three properties every single-flight-guarded build now must have:

1. **Bounded** — any file/network read that can hang is wrapped in
   `call_with_deadline(fn, timeout, default)` (`src/utils/deadline.py`). It runs
   `fn` on a throwaway daemon thread and returns `default` if it doesn't finish
   in time; the request thread is never blocked past the deadline. This is the
   portable bound: Windows has no `SIGALRM`, and a blocked C-level read
   (`os.stat`, `h5py.File`) can't be interrupted in-thread anyway. The thumbnail
   `.mat` fallback (8 s), the hist24 worker reads, and the Sheets `.execute()`
   (a backstop over the httplib2 socket timeout) are all bounded now. A hung
   read self-clears instead of wedging the lock.
2. **Observable + audible** — the bare `threading.Lock()` guards were replaced by
   `SingleFlight` (`src/dashboard/single_flight.py`), which records when the
   current build started. A later caller that finds the guard held past
   `STALL_SEC` (90 s — well past the 30 s DB / 10 s Sheets ceilings) calls
   `warn_if_stalled()`, which logs one loud `ERROR` per minute naming the wedged
   card. `_CachedBuilder` composes a `SingleFlight` and exposes `built_at()` /
   `age()` / `inflight_for()` so callbacks can render freshness/stall state.
3. **Honest in the UI** — `_guarded_card` wraps the cached fill callbacks: a
   built card carries a **`_freshness_footer`** ("updated 14:03:22 · 3 min ago",
   green/amber/red by age — the same `_fmt_age` bands as the KM-log stamp), and a
   rebuild wedged past `STALL_SEC` renders a **`_stalled_banner`** ("⚠ … hasn't
   updated in N min … click Refresh to retry") instead of the mute `⏳`. The
   thumbnail shows `_thumb_unavailable` ("source slow or unreachable") on a
   deadline miss. Stale data can no longer masquerade as live, and a stall is
   visible, not silent. (Retry reuses the global Refresh button rather than a
   per-card button, which would collide on `id` if two cards stalled at once.)

**Lesson:** single-flight without a deadline and a watchdog converts a transient
infra stall into a permanent, invisible freeze — the worst possible property for
a welfare/QC dashboard. Bound the read; time the guard; surface the state.

## 12. Accepted debt — why one process, and when to split

The nine mechanisms in §3 all exist because the dashboard shares **one process,
one `Store`, one SQLite file** with the daemon and its ~6 writer threads (§2).
That is a deliberate trade-off, not an accident: for a single-rig lab tool with a
handful of concurrent viewers, in-process means zero deployment/IPC overhead, a
warm shared cache, and one thing to run — and SQLite in WAL mode comfortably
serves many readers alongside one writer. The cost is that every heavy read
competes for the same file and thread pool, which is precisely what §3 manages.

The trigger to split the dashboard into its **own read-mostly process** (talking
to Postgres, or to a read replica / WAL-copy of the SQLite file) is when any of
these becomes true: more than a handful of simultaneous users; the daemon's write
volume saturates the disk badly enough that read latency stays high *after* the
§3 mitigations; or a second rig/DB is added. Until then, in-process is the right
call — but a future maintainer should make the split knowingly, not cargo-cult a
tenth mechanism onto a design that has outgrown its assumptions.
