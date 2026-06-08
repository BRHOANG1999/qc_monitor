/* Picture-in-picture for the Step 2 grid (video + LFP + Hilbert).
 *
 * When the entire #video-step2-grid scrolls out of viewport the
 * grid pops out of layout and floats fixed at the bottom-right.
 * Inside the floating PiP it re-flows to a single vertical
 * column: video on top, LFP underneath, Hilbert underneath that.
 * The <video> keeps playing, the LFP cursor + clientside zoom
 * sync stay live, and Plotly receives a synthetic resize event
 * after the class flips so the trace widths adjust to the
 * narrower PiP column.
 *
 * Why the IntersectionObserver had to grow:
 *   * Without a debounce, every micro-reflow toggled the class
 *     and the PiP flashed in/out. We now wait 220 ms of
 *     stability before flipping.
 *   * A small `rootMargin` adds hysteresis -- the PiP doesn't
 *     toggle right at the edge of the viewport. Less twitchy
 *     when the user scrolls slowly.
 *
 * Why setInterval polls for the grid: Video Review is rendered
 * on-demand (suppress_callback_exceptions), so the grid Div
 * doesn't exist at DOMContentLoaded. We re-check until we find
 * it, then stop. Capped at ~15 min of retries.
 */
(function () {
    'use strict';

    var GRID_ID = 'video-step2-grid';
    var PIP_CLASS = 'qc-pip-grid';
    var DEBOUNCE_MS = 220;

    var observer = null;
    var observed = null;
    var pendingState = null;
    var debounceTimer = null;
    var maxRetries = 600;
    var retries = 0;

    function applyPipClass(grid, on) {
        if (!grid) { return; }
        var has = grid.classList.contains(PIP_CLASS);
        if (on && !has) {
            grid.classList.add(PIP_CLASS);
        } else if (!on && has) {
            grid.classList.remove(PIP_CLASS);
        } else {
            return;
        }
        // Plotly only reflows on window resize. Fire a synthetic
        // one so the LFP + Hilbert traces re-scale to the new
        // container width (narrower in PiP, normal in grid).
        // Two ticks: one immediate, one ~200 ms later to catch
        // late-loaded plots.
        try {
            window.dispatchEvent(new Event('resize'));
            setTimeout(function () {
                window.dispatchEvent(new Event('resize'));
            }, 200);
        } catch (e) { /* old browsers; ignore */ }
    }

    function commitDebounced(grid, target) {
        if (debounceTimer) { clearTimeout(debounceTimer); }
        pendingState = target;
        debounceTimer = setTimeout(function () {
            debounceTimer = null;
            if (pendingState === null) { return; }
            applyPipClass(grid, pendingState);
            pendingState = null;
        }, DEBOUNCE_MS);
    }

    function ensureObserver() {
        if (observer) { return; }
        if (typeof IntersectionObserver === 'undefined') { return; }
        observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                var grid = document.getElementById(GRID_ID);
                if (!grid) { return; }
                // Add 80 px hysteresis: the PiP turns ON only
                // when the grid is well past the viewport edge,
                // and OFF only when it's well back in. Cuts
                // flicker at the boundary.
                commitDebounced(grid, !e.isIntersecting);
            });
        }, {threshold: 0, rootMargin: '-80px 0px -80px 0px'});
    }

    function trySetup() {
        retries += 1;
        var grid = document.getElementById(GRID_ID);
        if (!grid) { return retries < maxRetries; }
        if (observed === grid) { return true; }
        ensureObserver();
        if (!observer) { return true; }
        if (observed) { observer.unobserve(observed); }
        observer.observe(grid);
        observed = grid;
        return true;
    }

    document.addEventListener('DOMContentLoaded', trySetup);
    var pollId = setInterval(function () {
        var keepGoing = trySetup();
        if (!keepGoing) { clearInterval(pollId); }
    }, 1500);
})();
