// EstimateX — Development Mode one-line hint.
//
// Pure UI convenience: each <option> in #dev_mode carries its hint text
// as a data-hint attribute (set from app.py's DEV_MODE_OPTIONS in
// form.html); this just copies the selected option's hint into the
// #dev-mode-hint span. No new form field, no new /predict parameter —
// the backend only ever sees the existing dev_mode value.
(function () {
  var select = document.getElementById("dev_mode");
  var hint = document.getElementById("dev-mode-hint");

  if (!select || !hint) return;

  select.addEventListener("change", function () {
    var selected = select.options[select.selectedIndex];
    hint.textContent = selected ? selected.dataset.hint || "" : "";
  });
})();
