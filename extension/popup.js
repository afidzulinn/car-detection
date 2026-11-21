const statusEl = document.getElementById("status");

document.getElementById("start").addEventListener("click", () => {
  chrome.runtime.sendMessage({ action: "START" });
  if (statusEl) {
    statusEl.textContent = "Status: Streaming active (1 FPS)";
    statusEl.style.color = "#22c55e";
  }
});

document.getElementById("stop").addEventListener("click", () => {
  chrome.runtime.sendMessage({ action: "STOP" });
  if (statusEl) {
    statusEl.textContent = "Status: Stopped";
    statusEl.style.color = "#ef4444";
  }
});
