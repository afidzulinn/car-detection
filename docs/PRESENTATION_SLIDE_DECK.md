# Presentation Slide Deck Outline

Use this file as the content source for your PowerPoint deck.

## Slide 1: Title

- Project title: Car Component State Detector
- Team or participant name
- Competition or challenge name
- One-sentence summary: real-time detection of open and closed car components from a web-based 3D car view

## Slide 2: Problem Statement

- Goal: detect physical state changes of car components in real time
- Required states: open or closed
- Required delivery: working model plus web integration
- Main challenge: distinguish subtle visual differences across angles, elevations, and multiple component combinations

## Slide 3: System Overview

- Chrome extension captures the active browser tab
- FastAPI backend receives frames through WebSocket
- CNN predicts 5 component states
- Temporal smoothing reduces flicker
- Dashboard reports current states and changed components

Suggested visual:

- Use a simple pipeline diagram from browser to extension to backend to dashboard

## Slide 4: Model Architecture And Design Choices

- Custom CNN trained from scratch
- Five convolutional blocks with batch normalization, GELU, max pooling, and dropout
- Squeeze-and-excitation attention after feature extraction
- Multi-label output for 5 components
- Design choice: from-scratch model keeps the system lightweight and aligned with the challenge requirement

Concrete facts to include:

- 4,928,421 parameters
- `BCEWithLogitsLoss`
- `AdamW`
- `CosineAnnealingLR`

## Slide 5: Data Acquisition

- Automated data collection with Selenium
- Source: browser-based 3D car interface
- Exhaustive state coverage: 32 combinations of 5 binary components
- View diversity: 8 horizontal angles and 3 elevation levels
- Total raw dataset size: 768 annotated images

Suggested visual:

- Grid of sample screenshots showing angle and elevation variation

## Slide 6: Data Preprocessing And Augmentation

- Images resized to model input scale
- Combo-stratified validation split prevents augmented leakage into validation
- Augmentation types:
- brightness high
- brightness low
- motion blur
- gaussian noise
- hue shift
- perspective transform
- jpeg compression

Concrete facts to include:

- 5,376 augmented images added
- 6,144 annotated images total after augmentation
- Training split on augmented set: 4,992 train / 144 validation

## Slide 7: Training Approach

- CNN trained for 80 epochs
- Learning rate: `3e-4`
- Batch size: `32`
- Weight decay: `1e-4`
- Gradient clipping used for stability
- Validation monitors both per-label accuracy and exact-match accuracy

Talking point:

- The model was optimized for an end-to-end real-time workflow, not just offline classification

## Slide 8: Real-Time Integration

- FastAPI backend exposes WebSocket endpoints
- Extension streams frames from the active tab
- Dashboard displays 5 component states
- Change detector uses a short rolling buffer and hysteresis thresholds to reduce unstable predictions

Mention the endpoints:

- `WS /ws/extension`
- `WS /ws/dashboard`
- `GET /health`

## Slide 9: Results

Use a simple table:

| Label | Accuracy | ROC-AUC |
|------|----------|---------|
| Front left door | 40.97% | 0.5125 |
| Front right door | 40.28% | 0.5899 |
| Rear left door | 43.06% | 0.4067 |
| Rear right door | 40.97% | 0.3518 |
| Hood | 86.81% | 0.9167 |

Also include:

- Exact match accuracy across all 5 labels: `0.00%`
- The hood state is learned much better than the door states
- The end-to-end system works, but classification quality is still the main bottleneck

## Slide 10: Challenges Encountered

- Fine-grained visual differences between open and closed doors are subtle from some viewpoints
- Strong viewpoint changes make certain door states ambiguous
- Multi-label exact-match evaluation is strict and exposes compounding errors
- Real-time stability requires temporal smoothing in addition to frame-level classification

## Slide 11: How Challenges Were Addressed

- Automated collection to cover all component combinations systematically
- Multiple camera angles and elevations added to reduce overfitting to a single view
- Augmentation added lighting and quality variation
- Change detector added smoothing for live inference
- Separate optional modules explored richer outputs: VLM description and visual grounding

## Slide 12: Future Improvements

- Improve labeling and viewpoint diversity around door transitions
- Add tighter crops or region-focused modeling per component
- Explore transfer learning baselines for stronger feature extraction
- Calibrate thresholds per label instead of using one shared default
- Benchmark latency formally and optimize deployment path

## Slide 13: Demo Slide

- Show screenshots of:
- 3D car page
- extension popup
- dashboard with live state changes
- Optionally include one VLM or grounding screenshot as bonus work

## Slide 14: Conclusion

- Built a full end-to-end prototype from data collection to live inference
- Demonstrated browser integration and real-time status reporting
- Established a clear baseline and a practical roadmap for improving accuracy

## Presenter Notes

- Be honest about the current accuracy gap
- Emphasize engineering completeness: data collection, training, evaluation, backend, extension, and dashboard all exist
- Frame the current checkpoint as a baseline rather than a finished high-accuracy model
