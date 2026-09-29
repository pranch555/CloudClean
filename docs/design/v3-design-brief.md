# CloudClean v3 — design brief

## What was wrong (critique of v2)

- **No starting point.** The app opened in the middle of the pipeline (Process › Clean) on whatever asset was newest,
  with a parameter form (σ, voxel, "share of main") as the first thing on screen.
- **Three kinds of selection** — ticked (numbered checkbox), active (highlighted row), visible (eye) — plus "targets"
  derived from ticks-or-active. Nobody can predict what an operation will run on.
- **Two tab levels mixing unlike things**: pipeline steps (clean, merge, mesh), tools (edit, measure, colour) and
  metadata (info) side by side; modes (Capture/Process/Inspect/Autopilot) on top.
- **The model is not the hero.** A small grey cloud in navy chrome; panels take more than half the width.
- **Misleading headline numbers.** The size readout showed the axis-aligned box in scanner coordinates (a tilted bolt
  read 94.6 × 40.0 × 95.5 mm) and the real part size only in small print.
- **Generic look.** Navy surfaces, Inter, a sky-blue accent, uppercase micro-labels, 11–12 px text.

## Idea: "technical drawing, alive"

CloudClean turns a raw scan into a part you can trust the numbers of. The visual language borrows from engineering
drawings and precision instruments: warm drawing-paper surfaces, ink-black type, hairline rules, dimension lines with
arrowheads and value tags, and **one signal colour — vermilion** — for what matters right now (the current step, live
scanning, the measurement you are looking at, focus). Numbers are set like an instrument readout (monospace,
tabular). Everything else stays quiet so the model and its dimensions read first.

It must be friendly for anyone: plain words, large readable type (15 px body), big hit targets (≥ 36 px), one obvious
next action on every screen, and settings that start as sensible presets with details one click away.

## Information architecture

```
Home ──────────────── projects (one per physical part), big entry points, ask box, device status
 └─ Project workspace
     ├─ Journey (top bar):  1 Scan · 2 Clean · 3 Align · 4 Mesh · 5 Measure · 6 Export
     │     each step: status (done ✓ / current / to do / optional), click to go
     ├─ Left:  Models of this project (thumbnails, type, size; eye to show/hide) + "Add"
     ├─ Centre: 3D view (hero) — tool dock, part-size tag, view cube, Ask bar, live HUD
     └─ Right: [Step] | [Assistant] tabs
            Step panel = what this step does (one sentence), the main action with recommended
            settings, "Fine-tune" details, the result card ("size unchanged ±0.002 mm"), Next step →
Settings (dialog): Appearance (theme, units, text size) · Devices (scanner, turntable) · Assistant (LLM)
                   · Automations (autopilot) · About
```

- **One current model** (the one you clicked) is what every step works on. Multi-choice exists only where a step
  needs several models (Align lists scans with checkboxes). Visibility is separate and obvious (eye).
- Steps are a map, not a wizard: any step can be opened any time; the journey only shows progress and suggests the
  next one.
- The assistant is a copilot: an Ask bar floats in the viewport; answers and tool progress live in the Assistant tab.
  It can drive everything the user can (see docs/v3-plan.md Contract 7).

## Visual system

| Token | Paper (light, default) | Carbon (dark) |
|---|---|---|
| app background | `#F4F2ED` | `#0F0F10` |
| panel surface | `#FFFFFF` | `#18181A` |
| sunken / hover | `#F1EEE8` / `#ECE8E0` | `#202023` / `#29292D` |
| hairline / strong line | `#E6E2D9` / `#D5D0C4` | `#2C2C30` / `#3B3B41` |
| ink 1 / 2 / 3 | `#17161A` / `#4F4C45` / `#6E6A61` | `#F3F1EC` / `#C3BFB6` / `#908C83` |
| signal (vermilion) | `#EE4B1F`, text `#C0390F` | `#FF6A3D`, text `#FF8A63` |
| primary button | ink `#17161A` on white text | paper `#F3F1EC` with ink text |
| ok / warn / danger / info | `#1C8C4E` / `#9A6212` / `#C8302F` / `#2E5FD8` | `#4CC38A` / `#E6B04A` / `#FF7A70` / `#7BA2FF` |
| viewport | radial `#F7F5F1 → #E3DFD6`, graphite points `#45423C`, clay mesh `#B3AC9F` | radial `#1C1C1F → #0B0B0C`, points `#DCD7CD`, mesh `#8E897F` |

- Text contrast: every text token ≥ 4.5 : 1 on its surface (ink-3 on white 5.4 : 1). Signal text uses the darker
  `signal-ink`; the bright vermilion is for fills, strokes and glyphs.
- Status is never colour alone: always icon + word.
- Data colours (deviation maps, density) keep the validated dataviz ramps; categorical model colours are a fixed
  8-hue set tuned for both themes.

**Type.** Instrument Sans (UI; 15 px body, 13 px secondary, 12 px captions minimum, 600 for titles).
Instrument Serif (display only: Home greeting, empty states, step numbers). Geist Mono (every measured number,
tabular figures). Sentence case everywhere; no uppercase micro-labels except the tiny axis letters X/Y/Z.

**Shape.** 4 px grid. Radius 8 (controls) · 12 (cards) · 18 (panels, dialogs) · pill (primary buttons, chips,
journey). Soft warm shadows instead of borders for floating things; hairlines for structure.

**Motion.** 120/200/320 ms, ease-out; panels slide 12 px; nothing moves when the user prefers reduced motion.

**Signature elements.** Journey rail with numbered pills; dimension tags on the model (ink leader, vermilion value
chip, arrowheads); a part-size readout like a caliper display ("L 108.06 · W 38.28 · H 38.05 mm"); custom step
glyphs; the live-scanning "tally" dot in vermilion.

## Copy rules

Say what happens and what it means for the part, not the algorithm: "Removes stray points and scanner noise. Your
part keeps its size." Name the result: "Removed 1.9 % of points · size unchanged (≤ 0.002 mm)". Buttons are verbs
("Clean scan", "Check alignment", "Build mesh"). Numbers always carry units.
