/* Stim-overlay panel: collapse + draggable/resizable pop-out.
 *
 * The overlay graph (under the LFP feature trace in the Video Review
 * tab) lives in #video-overlay-panel. This file is the DOM-side helper
 * that two Dash clientside callbacks in video.py call when their Stores
 * change:
 *   * window.qcOverlay.applyCollapsed(on) -- hide/show the graph body
 *     (#video-overlay-body) and trigger a Plotly reflow on expand.
 *   * window.qcOverlay.applyPip(on) -- float the panel (position:fixed,
 *     CSS resize:both) and lazily wire a header drag-to-move handler.
 *
 * Mirrors the proven pattern in pip_video.js (class toggle + synthetic
 * resize ticks so Plotly re-scales). Not the browser PiP API.
 */
(function () {
    'use strict';

    var PANEL_ID = 'video-overlay-panel';
    var BODY_ID = 'video-overlay-body';
    var PIP_CLASS = 'qc-overlay-pip';
    var COLLAPSED_CLASS = 'qc-overlay-collapsed';
    var dragWired = false;

    function panel() { return document.getElementById(PANEL_ID); }

    // Plotly only reflows on window resize. Two synthetic ticks so the
    // overlay trace re-scales to the new container size.
    function nudgeResize() {
        try {
            window.dispatchEvent(new Event('resize'));
            setTimeout(function () {
                window.dispatchEvent(new Event('resize'));
            }, 120);
        } catch (e) { /* old browsers; ignore */ }
    }

    function applyCollapsed(on) {
        var p = panel();
        if (!p) { return; }
        var body = document.getElementById(BODY_ID);
        if (on) {
            p.classList.add(COLLAPSED_CLASS);
        } else {
            p.classList.remove(COLLAPSED_CLASS);
            if (body) { nudgeResize(); }  // re-render at full width
        }
    }

    // Header drag-to-move. Wired once; ignores mousedowns that start on
    // the controls (input/button/label) so they keep working.
    function wireDrag(p) {
        if (dragWired) { return; }
        var header = p.querySelector('.qc-overlay-header');
        if (!header) { return; }
        dragWired = true;
        var sx = 0, sy = 0, ox = 0, oy = 0, dragging = false;
        header.addEventListener('mousedown', function (e) {
            if (e.target.closest('input, button, label')) { return; }
            if (!p.classList.contains(PIP_CLASS)) { return; }
            var r = p.getBoundingClientRect();
            // Switch to left/top anchoring so dragging is absolute.
            p.style.left = r.left + 'px';
            p.style.top = r.top + 'px';
            p.style.right = 'auto';
            p.style.bottom = 'auto';
            sx = e.clientX; sy = e.clientY; ox = r.left; oy = r.top;
            dragging = true;
            e.preventDefault();
        });
        window.addEventListener('mousemove', function (e) {
            if (!dragging) { return; }
            p.style.left = (ox + e.clientX - sx) + 'px';
            p.style.top = (oy + e.clientY - sy) + 'px';
        });
        window.addEventListener('mouseup', function () {
            dragging = false;
        });
    }

    function applyPip(on) {
        var p = panel();
        if (!p) { return; }
        var has = p.classList.contains(PIP_CLASS);
        if (on && !has) {
            p.classList.add(PIP_CLASS);
            wireDrag(p);
        } else if (!on && has) {
            p.classList.remove(PIP_CLASS);
            // Clear inline drag offsets so it docks back cleanly.
            p.style.left = '';
            p.style.top = '';
            p.style.right = '';
            p.style.bottom = '';
        } else {
            return;
        }
        nudgeResize();
    }

    window.qcOverlay = {
        applyCollapsed: applyCollapsed,
        applyPip: applyPip
    };
})();
