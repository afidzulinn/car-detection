# Submission Checklist

Use this checklist before submitting the project.

## 1. GitHub Project Repository

- Source code is complete and organized
- `README.md` explains setup, usage, and troubleshooting
- `RUN_STEP_BY_STEP.md` matches the actual runtime flow
- Sensitive files are not included
- Large generated artifacts are excluded or handled appropriately

## 2. Demonstration Video

- Video shows the system running live
- Extension, backend, and dashboard are all visible at least briefly
- The 3D car page is shown
- Multiple component changes are demonstrated
- The dashboard shows current state reporting

## 3. Presentation Slide Deck

- Model architecture is explained
- Data acquisition and preprocessing are explained
- Training approach is explained
- Challenges and fixes are explained honestly
- Results and future work are included

## 4. Evaluation Criteria Coverage

### Model Architecture Efficiency

- Explain why a custom CNN was chosen
- Mention parameter count and lightweight real-time design goals

### Model Accuracy

- Include the measured validation results
- Explain which labels are strong and which are weak

### Real-Time Integration

- Show the extension to backend to dashboard workflow
- Demonstrate live state updates in the video

### Presentation And Reporting

- Use the slide outline in `docs/PRESENTATION_SLIDE_DECK.md`
- Keep the explanation clear, visual, and honest

## 5. Final Sanity Checks

- `http://localhost:8000/health` works
- Dashboard connects successfully
- Extension is loaded and active
- The 3D car page is reachable
- The checkpoint file exists if you plan to demo the trained model
