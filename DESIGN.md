# DESIGN.md: PA-SCAN Automotive Telemetry & 3D Inspection Workstation

## Design Read
> Reading this as: Automotive computer-vision inspection and telemetry workstation for ML researchers and vehicle engineers, in an Industrial Automotive Telemetry style, dial `ENERGY 2 / RHYTHM 2 / MOTION 1`.

## Three Dials
- **ENERGY: 2 (Balanced)**
  High-contrast technical instrumentation with clear visual hierarchy between the live 3D vehicle stage and tabular telemetry readouts.
- **RHYTHM: 2 (Consistent with deliberate asymmetric breaks)**
  Full-width compact top instrument header, followed by an asymmetric two-column inspection workspace (`1.75fr` primary 3D Vehicle Stage vs. `1fr` Component State & Diagnostic Readout), and a grounded status/metrics bar.
- **MOTION: 1 (Hover and focus states only)**
  Zero decorative infinite animations (no pulsing status dots, no floating cards). Transitions are restricted to 120ms interactive state feedback, guarded by `@media (prefers-reduced-motion: reduce)`, and the live auto-stream simulator includes an explicit Stop/Pause control.

## Five Levers
1. **One focal point per screen**: The large **Live 3D Vehicle Inspection Viewport** (`64%` desktop width, with `1.85x` Studio Close-Up default zoom) is the dominant visual anchor on the page.
2. **Hierarchical contrast**: Primary vehicle viewport, secondary component state rows with confidence bars, and tertiary tabular metadata (`angle`, `elevation`, `sample file`).
3. **Whitespace as structure**: Hairline structural dividers (`1px solid`) and disciplined `16px`/`24px` grid gaps separate the viewport controls, optical canvas, and component telemetry table without floating card clutter.
4. **One deliberate accent**: **Telemetry Amber** (`#F59E0B` / `#FBBF24` in dark mode; `#B45309` / `#92400E` in light mode) is used selectively for active mode selection, keyboard focus rings, and state-transition alerts.
5. **Identity motif**: Tabular numeric telemetry (`font-variant-numeric: tabular-nums`) paired with structured `[TAG]` status indicators (`OPEN`, `CLOSED`, `MATCH`, `MISMATCH`) across all three research objectives (PA-SCAN, VLM Description, Visual Grounding).

## One-Line Decision Log (Rule R-31)
- **Color Palette**: Carbon obsidian (`#0B0F14`) and machined graphite (`#121820`) match the studio backdrop of the 3D car renders so the white vehicle chassis and door apertures stand out with maximum contrast, while a complete Daylight Lab Light Mode (`#F4F6F8` / `#FFFFFF`) supports bright-room inspection (`R-21`, `R-25`, `R-29`, `R-34`).
- **Status Semantics**: Signal Emerald (`#065F46` / `#ECFDF5`) marks physically `OPEN` closures, Recessed Slate (`#1E293B` / `#E2E8F0`) marks normal `CLOSED` (latched) closures, and Crimson (`#7F1D1D` / `#FEE2E2`) is reserved strictly for Ground-Truth mismatches (`MISMATCH`) or connection faults so normal closed doors never look like error alarms (`R-29`, `R-31`).
- **Layout**: An asymmetric `1.75fr : 1fr` desktop grid (`max-width: 1440px`) expands the 3D Vehicle Viewport to over `840px` wide (up from `420px`), placing the 3D car stage first and the 5-component telemetry readout alongside it for single-glance verification (`R-03`, `R-05`, `R-14`).
- **3D Viewport Optical Zoom**: Because raw `1264x625` captures contain 58% empty studio margin around the centered vehicle, a built-in optical view switcher (`1.85x Studio Close-Up`, `2.4x Detail Zoom`, `1.0x Full Frame`) and toggleable `PA-SCAN ROI Box` enlarge the car by more than `3x` in screen area (`R-31`, `R-40`).
- **Typography**: Geometric sans-serif (`Inter`, `Segoe UI`, `system-ui`) provides clean legibility for component names and controls, while tabular monospace (`JetBrains Mono`, `Cascadia Mono`, `Consolas`) is used exclusively on live numbers (confidence, azimuth angle, FPS, latency) to prevent horizontal layout jitter during streaming (`R-06`, `R-31`).
- **Border Radius & Elevation**: Technical radii (`8px` panels, `6px` controls, `4px` status badges) and `1px` structural borders replace generic `999px` pills and heavy floating drop-shadows (`R-11`, `R-12`, `R-31`).
- **Icons & Affordances**: Decorative emojis (`▶`, `🔁`, `✅`, `❌`, `🔍`, `🎯`) and em dashes are eliminated; every control uses native `<button>`/`<a href>`/<select>`/`<input>` elements with `cursor: pointer`, visible `<label>`, `:focus-visible` rings (`>= 3:1`), and text-first status indicators (`R-02`, `R-04`, `R-32`, `R-39`).
