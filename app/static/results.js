// EstimateX — results page animations: numbers counting up, and
// feature-importance bars growing from 0, both purely CSS/vanilla-JS
// (no libraries). Cosmetic only — the underlying values are already
// rendered server-side in each element's data attributes, this only
// animates how they appear.
(function () {
  var prefersReducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function formatNumber(value, decimals, locale) {
    return value.toLocaleString(locale || "en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }

  function animateCounter(el) {
    var final = parseFloat(el.dataset.final);
    if (isNaN(final)) return;
    var decimals = parseInt(el.dataset.decimals || "0", 10);
    var prefix = el.dataset.prefix || "";
    var locale = el.dataset.locale || "en-US";

    if (prefersReducedMotion) {
      el.textContent = prefix + formatNumber(final, decimals, locale);
      return;
    }

    var duration = 800;
    var start = null;

    function step(timestamp) {
      if (start === null) start = timestamp;
      var progress = Math.min((timestamp - start) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
      var current = final * eased;
      el.textContent = prefix + formatNumber(current, decimals, locale);
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        el.textContent = prefix + formatNumber(final, decimals, locale);
      }
    }
    requestAnimationFrame(step);
  }

  function animateBars() {
    var bars = document.querySelectorAll("[data-bar-width]");
    bars.forEach(function (bar) {
      var width = bar.dataset.barWidth;
      var index = parseInt(bar.dataset.barIndex || "0", 10);
      var delay = prefersReducedMotion ? 0 : index * 80;
      setTimeout(function () {
        bar.style.width = width + "%";
      }, delay + 100); // small base delay so the card fade-in starts first
    });
  }

  document.querySelectorAll("[data-counter]").forEach(animateCounter);
  animateBars();
})();
