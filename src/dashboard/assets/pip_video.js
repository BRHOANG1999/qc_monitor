/* Picture-in-picture for the Step 2 grid (video + LFP).
 *
 * Manual toggle, NOT auto-pinned. The previous
 * IntersectionObserver implementation auto-popped the grid
 * into the floating PiP whenever it scrolled out of view; the
 * reviewer asked for explicit control. This file is now just
 * the DOM-side helpers (placeholder lifecycle, Plotly resize
 * trigger) and exposes ``applyPipClass(on)`` on
 * ``window.qcPipVideo`` so a Dash clientside callback in
 * ``video.py`` can call it on store change.
 *
 * Toggle paths:
 *   * Refresh-bar "PiP" button in ``app.py``.
 *   * "x Dock" button rendered inside the floating .qc-pip-grid.
 *   * ``P`` hotkey (routed through the kbd-event bus).
 *
 * All three flip the ``video-pip-state`` Store; the clientside
 * callback observes that Store and calls this helper.
 */
(function () {
    'use strict';

    var GRID_ID = 'video-step2-grid';
    var PIP_CLASS = 'qc-pip-grid';

    function ensurePlaceholder(grid) {
        var parent = grid.parentElement;
        if (!parent) { return null; }
        var existing = parent.querySelector(
            '.qc-pip-placeholder');
        if (existing) { return existing; }
        var ph = document.createElement('div');
        ph.className = 'qc-pip-placeholder';
        parent.insertBefore(ph, grid);
        return ph;
    }

    function removePlaceholder(grid) {
        var parent = grid.parentElement;
        if (!parent) { return; }
        var existing = parent.querySelector(
            '.qc-pip-placeholder');
        if (existing) { existing.remove(); }
    }

    function applyPipClass(on) {
        var grid = document.getElementById(GRID_ID);
        if (!grid) { return; }
        var has = grid.classList.contains(PIP_CLASS);
        if (on && !has) {
            // Page-jump fix: before popping the grid out of
            // document flow (position: fixed), insert a sibling
            // placeholder of the SAME height so the document
            // doesn't shrink and the browser doesn't snap the
            // scroll position to compensate.
            var h = Math.round(
                grid.getBoundingClientRect().height);
            var ph = ensurePlaceholder(grid);
            if (ph && h > 0) {
                ph.style.height = h + 'px';
            }
            grid.classList.add(PIP_CLASS);
        } else if (!on && has) {
            grid.classList.remove(PIP_CLASS);
            removePlaceholder(grid);
        } else {
            return;
        }
        // Plotly only reflows on window resize. Two synthetic
        // resize ticks so the LFP trace re-scales to the new
        // container width (narrower in PiP, normal in grid).
        try {
            window.dispatchEvent(new Event('resize'));
            setTimeout(function () {
                window.dispatchEvent(new Event('resize'));
            }, 200);
        } catch (e) { /* old browsers; ignore */ }
    }

    // Expose the helper so the Dash clientside callback in
    // video.py (subscribed to the video-pip-state Store) can
    // call it on every state change.
    window.qcPipVideo = {applyPipClass: applyPipClass};
})();
