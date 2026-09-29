# Frontend v3 — how it is put together

React 19 + TypeScript + zustand + three.js, built by Vite into `cloudclean/web/static`. Design language and copy
rules: `docs/design/v3-design-brief.md`. Backend contracts: `docs/v3-plan.md`.

## Screens and layout

| Area | Files |
|---|---|
| App shell, theme, init | `App.tsx` (Home ↔ Workspace; the workspace stays mounted behind Home so the WebGL context and loaded models survive), `lib/theme.ts` (`data-theme` = `paper`/`carbon`, `data-text` = `normal`/`large` on `<html>`) |
| Top bar | `shell/TopBar.tsx` (brand → Home, project switcher, `shell/Journey.tsx`, search, jobs, theme, settings, Ask) |
| Home | `home/HomeScreen.tsx` (greeting + ask box, three entry points, project cards with thumbnails and journey progress, device/assistant status strip, the dimensioned bolt illustration) |
| Workspace | left `shell/ModelsPanel.tsx`, centre `viewport/Viewport.tsx`, right `shell/SidePanel.tsx` (tabs: current step / Assistant), resizable `shell/Gutter.tsx` |
| Steps | `steps/StepFrame.tsx` (shared anatomy: StepFrame, TargetCard, ResultCard, ChoiceCards, Block, ToolTile, NextStepButton), `steps/CaptureStep` (in `steps/capture/`), `CleanStep`, `AlignStep`, `MeshStep`, `steps/measure/MeasureStep`, `ExportStep`; shared `EditTools.tsx` (edit recipe), `SmoothSection.tsx`, `DriftLine.tsx` + `useReport.ts` (accuracy drift of the operation that made a model), `Assessment.tsx`, `ReportView.tsx` |
| Viewport overlays | `viewport/SizeTag.tsx` (part size L×W×H; draws dimension lines), `Rails.tsx` (tool rail left, view rail right: fit, views, rotation style, section, display, grid, screenshot, video recording), `AskBar.tsx`, `Banners.tsx` (follow-scanner chip, loading, selection actions incl. assistant regions), `Labels.tsx` (measurements, dimension lines, notes, thread crests, M and P tools), `Legend.tsx`, selection/brush/pair/pane overlays, `features/capture/GuidanceHud.tsx` |
| Dialogs & helpers | `shell/SettingsDialog.tsx` (appearance, devices, assistant LLM, automations, shortcuts), `shell/CommandPalette.tsx` (Ctrl K; falls back to "Ask CloudClean: …"), `shell/JobsDrawer.tsx`, `shell/Tour.tsx` (first-run coach marks; Settings → "Show the guided tour again"), `shell/shortcuts.ts` |

## State

`store.ts` (zustand):

* `screen` (home/workspace), `projects`, `projectId`, `step` (capture/clean/align/mesh/measure/export),
  `rightTab` (step/assistant), `theme`, `textSize`, `layout` — persisted in `localStorage` (`cloudclean.*`).
* `activeId` = **the current model**; every step works on it. `selected` is only the multi-pick of the Align step.
  `visible` = what the 3D view shows.
* `dims` (dimension lines from measuring tools and the assistant), `annotations`, `measurements` (two-point M tool),
  `selection` (+ `selectionCounts`; a lasso/box *or* an assistant-highlighted region with `region`/`label`).
* `projectAssets()` / `useProjectAssets()` filter by project; `lib/projects.ts` falls back to one client-side
  "My scans" project when the server has no `/api/projects`.

Other stores: `features/assistant/assistantStore.ts` (conversation, streaming, attachments — shared by the Ask bar
and the Assistant tab), `features/capture/captureStore.ts` (live stream, `follow`), `lib/summary.ts` (part
summaries), `lib/thumbnails.ts` (client-rendered thumbnails, uploaded with `PUT /api/assets/{id}/thumbnail`).

## Assistant ↔ UI

`lib/applyUi.ts` implements every `ui` event of Contract 7 (show, focus, select, display, camera incl. orbit/zoom/
look_at, navigate, tool, section, highlight → viewer region mask + selection, measure_overlay → `store.dims`,
annotate, follow_scanner, layout, project). `uiContext()` builds the context sent with each chat message.

## Viewer (`viewer/Viewer.ts`)

Theme colours come from CSS tokens (`--vp-points`, `--vp-mesh`, `--vp-highlight`, `--vp-grid`) via
`Viewer.setTheme`. Additions in v3: `highlightRegion(assetId, resolvedRegion)` (tests in `lib/regions.ts`),
`orbit`, `zoomBy`, `lookAtPoint`, `cameraState`, `renderThumbnail`, `startRecording`/`stopRecording`, and
`fit()` that waits for models still loading. Every model gets a zeroed `selected` attribute at load (a shader
attribute without a buffer reads a stale generic value — that painted whole meshes in the highlight colour).

Live capture: any drag, wheel or double-click calls `viewer.onInteract` → `userTookTheView()` turns scanner-follow
off, so the camera is never pulled back while the user looks around; the "Follow scanner" chip turns it back on.

## Checks

* `npx tsc --noEmit`, `npm run build`.
* Accessibility: axe-core WCAG 2.1 AA scans of Home, Clean, Mesh, Assistant and Settings in both themes report no
  violations (2026-09-24). Every text token was contrast-checked (see the header of `styles/tokens.css`).
* Dev against another backend: `CLOUDCLEAN_API=http://127.0.0.1:18765 npx vite` (e.g. an SSH tunnel to the Spark).
