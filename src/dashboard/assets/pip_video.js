/* Picture-in-picture floating video.
 *
 * When the video container scrolls out of viewport, this script
 * adds the qc-video-pip class to it -- the CSS in theme.css
 * positions it fixed at the bottom-right so the reviewer can
 * keep watching while scoring events in Step 4. When the
 * container scrolls back into view (the reviewer scrolls up),
 * the class is removed and the player snaps back into the
 * Step 2 grid.
 *
 * The container's actual <video> element keeps playing through
 * the transition -- the only thing changing is the wrapping
 * Div's CSS position. Time-locking with the LFP cursor remains
 * intact because the cursor callback only reads
 * v.currentTime, not anything about the layout.
 *
 * Why setInterval instead of just DOMContentLoaded: the Video
 * Review tab is rendered on demand (suppress_callback_exceptions
 * is on); the container Div doesn't exist at page load. We
 * re-check every 1.5 s and attach the observer once it appears.
 */
(function () {
    'use strict';

    var observer = null;
    var observed = null;
    var maxRetries = 600;  // ~15 min then stop
    var retries = 0;

    function ensureObserver() {
        if (observer) { return; }
        if (typeof IntersectionObserver === 'undefined') { return; }
        observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (e) {
                var container = document.getElementById(
                    'video-player-container');
                if (!container) { return; }
                if (e.isIntersecting) {
                    container.classList.remove('qc-video-pip');
                } else {
                    container.classList.add('qc-video-pip');
                }
            });
        }, {threshold: 0.15});
    }

    function trySetup() {
        retries += 1;
        var container = document.getElementById('video-player-container');
        if (!container) { return retries < maxRetries; }
        if (observed === container) { return true; }
        ensureObserver();
        if (!observer) { return true; }
        if (observed) { observer.unobserve(observed); }
        observer.observe(container);
        observed = container;
        return true;
    }

    document.addEventListener('DOMContentLoaded', trySetup);
    var pollId = setInterval(function () {
        var keepGoing = trySetup();
        if (!keepGoing) { clearInterval(pollId); }
    }, 1500);
})();
