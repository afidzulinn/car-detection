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

let targetTabId = null;

async function captureAndSend() {
  try {
    // If targetTabId was set when Start was clicked, find that window
    let windowId = null;
    if (targetTabId) {
      try {
        const targetTab = await chrome.tabs.get(targetTabId);
        // If the target tab is active in its window, capture that window
        if (targetTab && targetTab.active) {
          windowId = targetTab.windowId;
        }
      } catch (err) {
        targetTabId = null; // Tab might have been closed
      }
    }

    // Otherwise, find the active tab that is NOT the dashboard (localhost:8000)
    if (!windowId) {
      const tabs = await chrome.tabs.query({ active: true });
      const carTab = tabs.find(t => t.url && !t.url.includes("localhost:8000") && !t.url.includes("127.0.0.1:8000"));
      if (carTab) {
        windowId = carTab.windowId;
      } else if (tabs.length > 0) {
        // Fallback: don't capture if user is solely looking at localhost:8000 dashboard to prevent infinite mirroring loop
        if (tabs[0].url && (tabs[0].url.includes("localhost:8000") || tabs[0].url.includes("127.0.0.1:8000"))) {
          return;
        }
        windowId = tabs[0].windowId;
      }
    }

    if (!windowId) return;

    const dataUrl = await chrome.tabs.captureVisibleTab(windowId, {
      format: "jpeg",
      quality: 85
    });

    if (ws?.readyState === WebSocket.OPEN) {
      ws.send(dataUrl);
    }
  } catch (e) {
    console.error("Capture error:", e);
  }
}

chrome.runtime.onMessage.addListener(async (msg) => {
  if (msg.action === "START") {
    // Capture the tab where user clicked Start
    const [currentTab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (currentTab && !currentTab.url?.includes("localhost:8000")) {
      targetTabId = currentTab.id;
    }
    connectWS();
    if (captureInterval) clearInterval(captureInterval);
    captureInterval = setInterval(captureAndSend, 1000); // 1 FPS
  }
  if (msg.action === "STOP") {
    if (captureInterval) clearInterval(captureInterval);
    captureInterval = null;
    targetTabId = null;
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
