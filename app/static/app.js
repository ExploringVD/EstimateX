// EstimateX front end: form behaviour and results-page animations. Each block is a no-op
// when its elements aren't on the current page.

// Domain toggle: show the software or construction fields with a fade instead of a reload.
(function () {
  var domainSelect = document.getElementById("domain");
  var softwareGroup = document.getElementById("software-fields");
  var constructionGroups = [
    document.getElementById("construction-fields"),
    document.getElementById("construction-fields-2"),
  ];
  var TRANSITION_MS = 220;

  if (!domainSelect || !softwareGroup) return;

  function setGroupState(group, visible) {
    if (!group) return;
    var fields = group.querySelectorAll("input, select");

    // Cancel a pending hide from an earlier toggle so it can't re-hide a group just shown.
    if (group._hideTimeoutId) {
      window.clearTimeout(group._hideTimeoutId);
      group._hideTimeoutId = null;
    }

    if (visible) {
      group.hidden = false;
      // Force layout so the transition starts from the hidden state.
      // eslint-disable-next-line no-unused-expressions
      group.offsetHeight;
      group.classList.remove("toggle-hidden");
      fields.forEach(function (field) {
        if (field.dataset.wasRequired === "true") field.required = true;
      });
    } else {
      group.classList.add("toggle-hidden");
      fields.forEach(function (field) {
        field.dataset.wasRequired = field.required ? "true" : "false";
        field.required = false;
      });
      group._hideTimeoutId = window.setTimeout(function () {
        group.hidden = true;
        group._hideTimeoutId = null;
      }, TRANSITION_MS);
    }
  }

  function markOriginalRequired(group) {
    if (!group) return;
    group.querySelectorAll("input, select").forEach(function (field) {
      field.dataset.wasRequired = field.required ? "true" : "false";
    });
  }

  markOriginalRequired(softwareGroup);
  constructionGroups.forEach(markOriginalRequired);

  function applyDomain() {
    var isConstruction = domainSelect.value === "construction";
    setGroupState(softwareGroup, !isConstruction);
    constructionGroups.forEach(function (group) {
      setGroupState(group, isConstruction);
    });
  }

  domainSelect.addEventListener("change", applyDomain);
  // Apply the initial state without animating.
  softwareGroup.classList.add("no-transition");
  constructionGroups.forEach(function (g) {
    if (g) g.classList.add("no-transition");
  });
  applyDomain();
  window.requestAnimationFrame(function () {
    softwareGroup.classList.remove("no-transition");
    constructionGroups.forEach(function (g) {
      if (g) g.classList.remove("no-transition");
    });
  });
})();

// KLOC presets fill the project size field; typing a different value clears the highlight.
(function () {
  var presetButtons = document.querySelectorAll(".kloc-preset-btn");
  var klocInput = document.getElementById("project_size_kloc");

  if (!presetButtons.length || !klocInput) return;

  presetButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      klocInput.value = btn.dataset.klocValue;
      presetButtons.forEach(function (b) {
        b.classList.remove("kloc-preset-active");
      });
      btn.classList.add("kloc-preset-active");
      klocInput.dispatchEvent(new Event("input", { bubbles: true }));
      klocInput.focus();
    });
  });

  klocInput.addEventListener("input", function () {
    var active = document.querySelector(".kloc-preset-active");
    if (active && active.dataset.klocValue !== klocInput.value) {
      active.classList.remove("kloc-preset-active");
    }
  });
})();

// Development mode hint follows the selected option.
(function () {
  var select = document.getElementById("dev_mode");
  var hint = document.getElementById("dev-mode-hint");

  if (!select || !hint) return;

  select.addEventListener("change", function () {
    var selected = select.options[select.selectedIndex];
    hint.textContent = selected ? selected.dataset.hint || "" : "";
  });
})();

// Results page: numbers count up and factor bars grow from zero.
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
      el.textContent = prefix + formatNumber(final * eased, decimals, locale);
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        el.textContent = prefix + formatNumber(final, decimals, locale);
      }
    }
    requestAnimationFrame(step);
  }

  document.querySelectorAll("[data-counter]").forEach(animateCounter);
  document.querySelectorAll("[data-bar-width]").forEach(function (bar) {
    var index = parseInt(bar.dataset.barIndex || "0", 10);
    var delay = prefersReducedMotion ? 0 : index * 80;
    setTimeout(function () {
      bar.style.width = bar.dataset.barWidth + "%";
    }, delay + 100); // let the card fade in first
  });
})();
