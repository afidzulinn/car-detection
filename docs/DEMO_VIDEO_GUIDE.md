# Demo Video Guide

Use this guide to record the submission video so it matches the required deliverable and shows the strongest parts of the project clearly.

## Target Length

- Recommended: 2 to 4 minutes

## Recording Goal

Show the model running in real time with the web interface and clearly demonstrate state reporting for different car component states.

## Before You Record

- Start the backend
- Confirm `http://localhost:8000/health` responds
- Serve the frontend or start Docker
- Load the Chrome extension
- Open the 3D car page
- Open the dashboard page
- Make sure the extension popup is easy to access during the recording

## Recommended Recording Flow

### Part 1: Quick introduction

Show:

- project folder or README
- the running backend terminal
- the dashboard page

Narration idea:

- "This project detects whether car components are open or closed from a live browser view."

### Part 2: Show the live system pieces

Show:

- the 3D car page
- the extension popup
- the dashboard with five tracked components

Narration idea:

- "The extension captures the active tab and streams frames to a FastAPI backend, which runs the CNN and pushes predictions to the dashboard."

### Part 3: Start detection

Actions:

1. Click the extension popup
2. Start the detection stream
3. Switch between the 3D page and the dashboard if needed

What viewers should see:

- connection becomes active
- component states begin updating

### Part 4: Demonstrate component changes

Show at least several state transitions:

1. Open one front door
2. Close it again
3. Open another door or the hood
4. Show the dashboard update and the changed-component alert

Best practice:

- Pause briefly after each click so viewers can see the dashboard update
- Keep the active captured tab on the 3D car page when detection is running

### Part 5: Show status reporting

Make sure the recording clearly shows:

- each component label
- its open or closed badge
- confidence values
- changed-component highlight or alert

### Part 6: Optional bonus section

If Objectives 2 and 3 are working, show one quick example each:

- VLM description page
- visual grounding page

Keep this short so the mandatory real-time detector stays the focus.

## Suggested Voiceover Points

- Dataset was collected automatically across all component combinations
- Model is a custom CNN trained from scratch
- Backend uses WebSocket streaming for live updates
- Temporal smoothing helps reduce unstable frame-by-frame predictions
- Current system is end-to-end functional, with accuracy improvement as the next priority

## Common Recording Mistakes To Avoid

- Do not spend too long on code before showing the live system
- Do not keep the wrong browser tab active when the extension is capturing
- Do not move too quickly between state changes
- Do not hide the dashboard while demonstrating the 3D interactions

## Final Checklist

- The backend terminal is visible at least once
- The dashboard is clearly readable
- At least two different component states are demonstrated
- The video shows real-time interaction, not just static screenshots
- The recording ends cleanly with the working system visible
