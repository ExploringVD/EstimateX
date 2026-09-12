// EstimateX — Project Size (KLOC) quick-fill presets.
//
// Pure UI convenience: clicking a preset button sets the SAME existing
// #project_size_kloc number input's value — no new field, no new
// /predict parameter, the backend still only ever sees
// project_size_kloc exactly as before this feature existed. The field
// stays a normal, freely-editable number input both before and after a
// preset click; typing a different number just clears whichever preset
// was marked active, it never locks or disables the input.
//
// No-op harmlessly if the presets/field aren't on the page (e.g. this
// script loading on a page without the software fields for any reason).
(function () {
  var presetButtons = document.querySelectorAll(".kloc-preset-btn");
  var klocInput = document.getElementById("project_size_kloc");

  if (!presetButtons.length || !klocInput) return;

  function clearActivePreset() {
    presetButtons.forEach(function (btn) {
      btn.classList.remove("kloc-preset-active");
    });
  }

  presetButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      klocInput.value = btn.dataset.klocValue;
      clearActivePreset();
      btn.classList.add("kloc-preset-active");
      // So the change is visible immediately in anything else listening
      // for input on this field (there isn't currently, but this keeps
      // the field's real event contract intact rather than silently
      // bypassing it).
      klocInput.dispatchEvent(new Event("input", { bubbles: true }));
      klocInput.focus();
    });
  });

  // Typing a different value than the active preset's should clear the
  // active highlight — the field is never "locked" into a preset choice.
  klocInput.addEventListener("input", function () {
    var active = document.querySelector(".kloc-preset-active");
    if (active && active.dataset.klocValue !== klocInput.value) {
      active.classList.remove("kloc-preset-active");
    }
  });
})();
