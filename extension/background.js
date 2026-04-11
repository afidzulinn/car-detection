const WS_URL = "ws://localhost:8000/ws/extension";
let ws = null;
let captureInterval = null;

// ── Continuous streaming (Objective 1) ──────────────────────
function connectWS() {
  ws = new WebSocket(WS_URL);
  ws.onopen  = () => console.log("WebSocket connected to backend");
  ws.onclose = () => {
    console.log("WS closed — reconnecting in 3s");
    setTimeout(connectWS, 3000);
  };
  ws.onerror = (e) => console.error("WS error:", e);
}

async function captureAndSend() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) return;

    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
      format: "jpeg",
      quality: 80
    });

    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(dataUrl);
    }
  } catch (e) {
    console.error("Capture error:", e);
  }
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.action === "START") {
    connectWS();
    captureInterval = setInterval(captureAndSend, 1000); // 1 FPS
  }
  if (msg.action === "STOP") {
    clearInterval(captureInterval);
    ws?.close();
  }
});

// ── One-shot capture (Objective 2 & 3) ─────────────────────
chrome.runtime.onMessageExternal.addListener(
  async (message, sender, sendResponse) => {
    if (message.action === "CAPTURE_ONCE") {
      try {
        const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (!tab) {
          sendResponse({ error: "No active tab found" });
          return true;
        }
        const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, {
          format: "jpeg",
          quality: 85
        });
        sendResponse({ image: dataUrl });
      } catch (e) {
        sendResponse({ error: e.message });
      }
    }
    return true; // Keep message channel open for async response
  }
);
