// Shows a "working..." overlay for slow actions (resume analysis, job search, AI).
document.addEventListener("submit", (e) => {
  const msg = e.target.dataset.busy;
  if (msg) showBusy(msg);
});
document.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-busy]");
  if (a) showBusy(a.dataset.busy);
});
function showBusy(msg) {
  const o = document.createElement("div");
  o.className = "busy-overlay";
  o.setAttribute("role", "status");
  o.innerHTML = "<div></div>";
  o.firstChild.textContent = msg;
  document.body.appendChild(o);
}
window.addEventListener("pageshow", () => document.querySelectorAll(".busy-overlay").forEach((o) => o.remove()));
