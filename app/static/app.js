// EstimateX — domain-aware form toggle (no page reload).
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

  if (!domainSelect || !softwareGroup) return;

  function setGroupState(group, visible) {
    if (!group) return;
    group.hidden = !visible;
    var fields = group.querySelectorAll("input, select");
    for (var i = 0; i < fields.length; i++) {
      var field = fields[i];
      if (visible) {
        if (field.dataset.wasRequired === "true") {
          field.required = true;
        }
      } else {
        field.dataset.wasRequired = field.required ? "true" : "false";
        field.required = false;
      }
    }
  }

  // Record each field's original `required` state once, before any
  // toggling happens, so re-showing a group restores it correctly.
  function markOriginalRequired(group) {
    if (!group) return;
    var fields = group.querySelectorAll("input, select");
    for (var i = 0; i < fields.length; i++) {
      fields[i].dataset.wasRequired = fields[i].required ? "true" : "false";
    }
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
  applyDomain();
})();
