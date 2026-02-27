# distill Browser Extension

Chrome extension that scores any web page for content quality using the distill local server.

## Setup

1. **Start the server:**
   ```bash
   pip install -e ".[server]"
   distill serve
   ```
   The server runs on `http://127.0.0.1:7331` by default.

2. **Load the extension:**
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" (top right)
   - Click "Load unpacked" and select this `extension/` directory

3. **Use it:**
   - Navigate to any web page
   - Click the distill icon in the toolbar
   - Click "Score Page" to analyze the current page

## How it works

The extension extracts the page HTML, sends it to the local distill server, and displays the quality scores in the popup. All processing happens locally — no data is sent to external servers.

## Server API

- `POST /score` — Score content. Body: `{"html": "...", "url": "..."}` or `{"text": "...", "url": "..."}`
- `GET /health` — Health check. Returns `{"status": "ok", "version": "..."}`
