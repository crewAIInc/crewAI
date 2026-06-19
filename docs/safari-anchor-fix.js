/**
 * Fix for anchor/hash links not scrolling on Safari.
 *
 * Safari does not re-scroll to the hash target after Next.js hydration
 * replaces the server-rendered DOM. This script watches for the target
 * element and scrolls to it once it appears.
 */
(function () {
  if (typeof window === 'undefined') return;

  function scrollToHash() {
    var hash = window.location.hash;
    if (!hash) return;

    var id = hash.substring(1);
    var el = document.getElementById(id);
    if (el) {
      el.scrollIntoView();
      return true;
    }
    return false;
  }

  // Try immediately (works in Chrome/Firefox).
  if (scrollToHash()) return;

  // For Safari: retry after hydration settles.
  // Use requestAnimationFrame + small delay to wait for React hydration.
  var attempts = 0;
  var maxAttempts = 20; // ~2 seconds max

  function retry() {
    attempts++;
    if (scrollToHash() || attempts >= maxAttempts) return;
    setTimeout(retry, 100);
  }

  if (document.readyState === 'complete') {
    retry();
  } else {
    window.addEventListener('load', retry);
  }
})();
