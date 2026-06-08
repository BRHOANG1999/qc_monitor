/* QC Monitor keyboard layer.
 *
 * One job: forward raw `keydown` events on `document` into the
 * `kbd-keydown` Dash Store. All key->action mapping and routing
 * happens in Python (see src/dashboard/keyboard.py) so adding a
 * shortcut is a one-place change there.
 *
 * Bail-outs (don't push):
 *   * focus is inside an <input>, <textarea>, or [contenteditable]
 *     -- typing 'n' in a note must NOT fire mark_no_events.
 *   * Ctrl / Meta is held -- preserve browser/OS shortcuts.
 *   * the tab is not visible (visibilityState !== 'visible').
 *   * Dash hasn't finished booting yet (dash_clientside absent).
 *
 * NASA Rule 4 cap: this file should stay <60 logical lines.
 */
(function () {
    'use strict';

    function isTypingTarget(el) {
        if (!el) { return false; }
        var tag = (el.tagName || '').toUpperCase();
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
            return true;
        }
        if (el.isContentEditable) { return true; }
        return false;
    }

    function pushKeydown(e) {
        if (e.ctrlKey || e.metaKey) { return; }
        if (document.visibilityState !== 'visible') { return; }
        if (isTypingTarget(e.target)) {
            // Special case: ESC inside a text input still closes
            // the overlay. Everything else is yours to type.
            if (e.key !== 'Escape') { return; }
        }
        if (typeof window.dash_clientside === 'undefined' ||
                !window.dash_clientside.set_props) {
            return;
        }
        // Block defaults for the keys we actually own so the
        // browser doesn't scroll on Space / ArrowDown.
        var owned = {
            'j': 1, 'k': 1, 'n': 1, 'e': 1, 'm': 1, 'x': 1,
            'u': 1, 'r': 1, '/': 1, '?': 1, 'Escape': 1, ' ': 1,
            'ArrowUp': 1, 'ArrowDown': 1,
            'ArrowLeft': 1, 'ArrowRight': 1,
        };
        if (owned[e.key]) { e.preventDefault(); }
        window.dash_clientside.set_props('kbd-keydown', {
            data: {
                key: e.key,
                shift: !!e.shiftKey,
                ctrl: !!e.ctrlKey,
                meta: !!e.metaKey,
                alt: !!e.altKey,
                ts: Date.now(),
            },
        });
    }

    // Use capture so we see the event before Dash's own React
    // tree consumes it (e.g. radio buttons listening to space).
    document.addEventListener('keydown', pushKeydown, true);

    // Backdrop click on the help overlay closes it. Cheap to
    // attach here; the overlay element may not exist yet at
    // DOMContentLoaded, so we listen on document and filter.
    document.addEventListener('click', function (e) {
        var overlay = document.getElementById('kbd-help-overlay');
        if (!overlay) { return; }
        if (overlay.style.display === 'none' ||
                overlay.style.display === '') { return; }
        if (e.target === overlay) {
            overlay.style.display = 'none';
        }
    });
})();
