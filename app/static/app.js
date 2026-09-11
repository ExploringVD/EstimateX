// EstimateX — domain-aware form toggle (no page reload), with a
// fade/slide transition instead of an instant show/hide.
//
// Only "construction" shows the construction field group; every other
// domain value (software_web, mobile_app, enterprise_other) shows the
// software field group — they all route to the same unified software
// model (see app.py), the domain split here is purely so the fields
// shown feel natural, not because there are 3 separate models.
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

    // Cancel any pending hide-after-transition callback from a PREVIOUS
    // call for this same group — without this, a stale timeout (e.g.
    // from the initial page-load state) can fire after a later toggle
    // and re-hide a group that was just shown (a real race condition
    // caught in testing: switching domains before the first timeout
    // fired left the construction fields invisible).
    if (group._hideTimeoutId) {
      window.clearTimeout(group._hideTimeoutId);
      group._hideTimeoutId = null;
    }

    if (visible) {
      group.hidden = false;
      // Force layout so the browser registers the "hidden" starting
      // state before we remove it — otherwise the transition is skipped.
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
  // Initial state applies instantly (no transition) so the form doesn't
  // visibly animate on first load.
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
