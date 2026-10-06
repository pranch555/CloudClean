"""The app guide: every feature of the web app, by the name it has on screen, where it is and how to get there.

The assistant answers "where is ...", "how do I ..." and "what does ... do" from this catalogue (tool app_guide),
and the web app opens the place and rings the control with the same `data-guide` id (lib/guide.ts). Keep names and
paths identical to the labels in the UI: the user reads them in the answer and then looks for them on screen.

nav (what the web app does to get there; all optional):
  screen: home | workspace       step: capture | clean | align | mesh | measure | export
  right: step | assistant        left: true (show the model list)
  measure_tab: dimensions | thread | cad | accuracy        measure_tool: p2p | across | extent | steps | diameter |
  angle | flatness | sphere | section                     settings: appearance | devices | assistant | automations |
  about                          jobs: true
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    id: str
    name: str          # the label on screen
    where: str         # how to get there, in the words of the UI ("Measure → Dimensions → Heights & steps")
    what: str          # what it is for, one or two sentences
    keywords: str = ""  # other words people use for it
    nav: dict = field(default_factory=dict)

    def brief(self) -> dict:
        return {"id": self.id, "name": self.name, "where": self.where, "what": self.what}


def _m(tab: str, tool: str | None = None) -> dict:
    nav = {"step": "measure", "right": "step", "measure_tab": tab}
    if tool:
        nav["measure_tool"] = tool
    return nav


def _s(step: str) -> dict:
    return {"step": step, "right": "step"}


FEATURES: list[Feature] = [
    # ---- getting around
    Feature("home", "Home", "the CloudClean logo at the top left",
            "All projects, recent work and quick starts: scan a part, open scan files, check against CAD.",
            "start page projects list main menu", {"screen": "home"}),
    Feature("projects", "Project switcher", "the project name at the top, next to the logo",
            "Switch to another project (one project per physical part) or create a new one.",
            "project folder switch change new create part", {"screen": "workspace"}),
    Feature("steps", "The six steps", "the step bar at the top: Scan · Clean · Align · Mesh · Measure · Export",
            "The journey through one part. Any step can be opened at any time; ticks show what is done.",
            "workflow journey stages tabs progress", {"screen": "workspace"}),
    Feature("assistant", "Ask CloudClean", "Ask CloudClean at the top right, the Ask bar under the 3D view, or Ctrl K",
            "Ask anything in plain words: it can do the work (scan, clean, merge, measure, export, turntable) or "
            "show you where things are.", "chat ai help question bot", {"right": "assistant"}),
    Feature("photos", "Photos", "the Photos button in the assistant's message box (or the photo icon in the Ask bar)",
            "Add photos of the real part from this computer, a picture of the 3D view, or photos saved in the "
            "project. New photos are kept in the project as reference photos the assistant can look at again.",
            "image picture reference attach upload camera phone", {"right": "assistant"}),
    Feature("chat.reply", "Latest reply", "the Latest reply box above the Ask bar under the 3D view (it shows while "
            "the side panel is not showing the chat)",
            "The assistant's last answer in short while you work: your question, the step it is working on and the "
            "answer. Show more makes it taller; its buttons open the whole chat in the side panel, pop the chat out, "
            "or hide the box until the next reply (the assistant sign at the start of the Ask bar shows it again).",
            "answer response reply preview last message snippet result box", {"screen": "workspace"}),
    Feature("chat.popout", "Pop out the chat", "the Pop out the chat button at the top of the Assistant tab, or on the "
            "Latest reply box above the Ask bar",
            "Moves the conversation into a small window over the app that you can drag anywhere and resize, so you "
            "can keep talking to the assistant while you work in Clean, Mesh or any other step with its tools in the "
            "side panel. It stays open when you change steps.",
            "floating chat window detach undock drag move separate keep open while working picture in picture",
            {"right": "assistant"}),
    Feature("chat.minimize", "Minimize the chat", "the – button at the top right of the floating chat (or Esc in it; "
            "Ctrl J opens it again)",
            "Shrinks the floating chat to a small bubble you can drag anywhere. The bubble shows when the assistant "
            "is working or has a new reply; click it to open the chat again.",
            "minimise collapse hide bubble small out of the way", {}),
    Feature("chat.size", "Make the chat tall", "the Make the chat tall / Make the chat compact button at the top of "
            "the floating chat, or drag its bottom-right corner",
            "Switches the floating chat between compact and the full height of the 3D view; drag the corner for any "
            "size.", "resize bigger smaller larger taller compact chat window size", {}),
    Feature("chat.dock", "Dock the chat", "the Dock the chat button at the top right of the floating chat, or Dock it "
            "here in the Assistant tab", "Puts the floating chat back into the side panel's Assistant tab.",
            "dock attach put back side panel restore", {"right": "assistant"}),
    Feature("settings", "Settings", "the gear button at the top right",
            "Theme, text size, units, scanner and turntable, the assistant's LLM server, automations, shortcuts.",
            "preferences options configuration", {}),
    Feature("settings.appearance", "Appearance", "Settings → Appearance",
            "Light (Paper) or dark (Carbon) theme, larger text, units shown and the part size readout.",
            "theme dark mode light mode colour color text size font units mm inch", {"settings": "appearance"}),
    Feature("settings.devices", "Scanner & turntable settings", "Settings → Scanner & turntable",
            "Default scanner, turntable device and scan defaults.", "scanner turntable bluetooth device defaults",
            {"settings": "devices"}),
    Feature("settings.assistant", "Assistant settings", "Settings → Assistant",
            "Which LLM server and model the assistant uses, and whether it can see images (vision).",
            "llm model server vision qwen ai settings", {"settings": "assistant"}),
    Feature("settings.automations", "Automations", "Settings → Automations",
            "Hands-free processing: every new scan is cleaned, merged, meshed and exported automatically, "
            "optionally from a watched folder.", "autopilot automatic watch folder hands free pipeline",
            {"settings": "automations"}),
    Feature("settings.about", "Shortcuts & about", "Settings → Shortcuts & about",
            "Every keyboard shortcut, and the guided tour of the app.", "keyboard shortcuts keys hotkeys tour help",
            {"settings": "about"}),
    Feature("tour", "Guided tour", "Settings → Shortcuts & about → Show the guided tour again",
            "A short walk through the app for new users.", "tutorial walkthrough introduction onboarding",
            {"settings": "about"}),
    Feature("theme", "Light / dark theme", "the moon / sun button at the top right (or Settings → Appearance)",
            "Switch between the light Paper and the dark Carbon theme.", "dark mode light mode night theme", {}),
    Feature("jobs", "Jobs and logs", "the jobs button at the top right (it shows while something runs)",
            "Everything running or finished, with progress, logs and errors.", "progress log running tasks queue",
            {"jobs": True}),
    # ---- models
    Feature("models", "Model list", "the Models panel on the left (in a narrow window: the button at the top left of "
            "the 3D view)", "Every scan, merge, mesh and result of this project in coloured groups (Scans, Made from "
            "scans, Checks, CAD & golden models, Photos); click one to work on it, the eye shows or hides it.", "assets files scans list tree left panel",
            {"screen": "workspace", "left": True}),
    Feature("add", "Add", "Models → + Add",
            "Scan a part, open scan files from this computer (PLY, STL, OBJ, STEP, photos) or import big files from "
            "the server's disk.", "import open upload load file new scan", {"screen": "workspace", "left": True}),
    Feature("model.menu", "Model menu", "the … button on a model in the Models list",
            "Rename, download, move to another project, delete.", "rename download delete move model options",
            {"screen": "workspace", "left": True}),
    Feature("models.scans", "Scans", "Models → Scans (the blue group at the top of the Models list)",
            "Scans straight from the scanner, or opened from files. Every group of the Models list has its own "
            "colour: Scans, Made from scans, Checks, CAD & golden models, Photos.",
            "scanned raw captures imported files point clouds group", {"screen": "workspace", "left": True}),
    Feature("models.made", "Made from scans", "Models → Made from scans (the violet group)",
            "Cleaned, merged and meshed versions of your scans; each says what it was made from. The original "
            "scans stay as they are.", "results versions cleaned merged mesh derived group",
            {"screen": "workspace", "left": True}),
    Feature("model.made-from", "Made from", "Models → the From … line under a model",
            "What a model was made from; click a name to go to that model.", "lineage parent source history origin",
            {"screen": "workspace", "left": True}),
    Feature("models.checks", "Checks", "Models → Checks (the red group)",
            "Golden checks and deviation maps: your scans compared with the golden model, newest first, each with "
            "its verdict (Matches, Mostly matches, Does not match, Scan more) and its deviation map under it.",
            "inspection results golden check deviation map compare verdict group", {"screen": "workspace", "left": True}),
    Feature("model.check-result", "Golden check verdict", "Models → Checks → the verdict on a golden check (Matches, "
            "Mostly matches, Does not match or Scan more)", "Click the verdict to open the full result in Measure → "
            "Golden model.", "verdict pass fail result open outcome", {"screen": "workspace", "left": True}),
    Feature("models.cad", "CAD & golden models", "Models → CAD & golden models (the green group)",
            "CAD models (STEP) and the golden model: the part as it should be. The ★ Golden model tag marks the one "
            "golden checks compare your scans with.", "cad step reference design nominal group",
            {"screen": "workspace", "left": True}),
    Feature("model.golden", "Golden model tag", "Models → CAD & golden models → the ★ Golden model tag",
            "Marks the golden model of this project, the one golden checks compare scans with.",
            "star golden reference nominal tag", {"screen": "workspace", "left": True}),
    Feature("model.make-golden", "Make it the golden model", "the … button on a mesh or CAD model in the Models "
            "list → Make it the golden model", "Makes this model the golden model of the project: Measure → Golden "
            "model then checks scans against it.", "set golden reference nominal choose", {"screen": "workspace", "left": True}),
    Feature("model.show", "Show or hide in 3D", "the eye button on each model in the Models list",
            "A red eye means the model is shown in the 3D view, a closed eye means hidden; click to switch. "
            "Double-click a model to show only that one.", "visible visibility hidden eye model",
            {"screen": "workspace", "left": True}),
    Feature("models.show-all", "Show all / hide all", "the eye in the title bar of a group in the Models list",
            "Shows or hides every model of that group in the 3D view at once.", "show all hide all group visibility eye",
            {"screen": "workspace", "left": True}),
    Feature("models.photos", "Photos", "Models → Photos (the pink group at the bottom of the Models list)",
            "The photos of the real part in this project. Click one to see it full size; the × on a photo deletes it.",
            "pictures images gallery list", {"screen": "workspace", "left": True}),
    Feature("photo.viewer", "Photo viewer", "Models → Photos → click a photo (or Scan → Make a 3D model from photos → "
            "click a photo)", "A photo full size; arrow keys step through them. Download it, delete it, or use the "
            "photos to colour a model.", "view enlarge bigger big see zoom full size lightbox preview picture open",
            {"screen": "workspace", "left": True}),
    Feature("photo.delete", "Delete photo", "the × on a photo (Models → Photos, or Scan → Make a 3D model from "
            "photos), or Delete in the photo viewer", "Removes a photo from the project after you confirm. Models "
            "already made or coloured from it stay.", "delete remove erase picture image trash",
            {"screen": "workspace", "left": True}),
    Feature("photos.delete-all", "Delete all photos", "Models → Photos → the … button → Delete all photos",
            "Removes every photo of this project after you confirm.", "delete remove all clear pictures images",
            {"screen": "workspace", "left": True}),
    # ---- 3D view
    Feature("view.tools", "Tool bar", "the bar on the left of the 3D view",
            "Move the view, select with a box or by drawing around, measure between two points, smoothing brush, "
            "pick the point to turn around.", "tools toolbar select box lasso", {"screen": "workspace"}),
    Feature("tool.box", "Select with a box", "the dashed-square button on the left of the 3D view (key B)",
            "Drag a rectangle to select points: then delete them, keep only them, or measure just that part.",
            "select rectangle marquee selection", {"screen": "workspace"}),
    Feature("tool.lasso", "Select by drawing around", "the lasso button on the left of the 3D view (key L)",
            "Draw around points to select them.", "lasso freehand select selection", {"screen": "workspace"}),
    Feature("tool.brush", "Smoothing brush", "the brush button on the left of the 3D view (key S), or Clean → Smooth "
            "rough spots", "Paint over rough patches to smooth them; the rest of the part is untouched.",
            "smooth brush rough noise bumps paint", _s("clean")),
    Feature("tool.pivot", "Pick the point to turn around", "the crosshair button on the left of the 3D view (key P)",
            "Click a spot on the model: the view then rotates around it.", "pivot rotate centre center orbit",
            {"screen": "workspace"}),
    Feature("view.fit", "Fit the view", "the top button on the right of the 3D view (key F, or double-click)",
            "Frames the models so all of them are visible.", "zoom fit frame reset view", {"screen": "workspace"}),
    Feature("view.standard", "Standard views", "the grid button on the right of the 3D view",
            "Front, right, top, back, left, bottom and isometric views; perspective or orthographic.",
            "front top side iso view cube orthographic perspective", {"screen": "workspace"}),
    Feature("view.section", "Section view", "the scissors button on the right of the 3D view",
            "Cuts the view open along X, Y or Z to look inside (the data is not changed).",
            "clip cut away inside clipping plane", {"screen": "workspace"}),
    Feature("view.display", "Display", "the sliders button on the right of the 3D view",
            "Colours (scan colours, plain, per model, surface direction, deviation / data), point size, wireframe, "
            "box, floor grid, up axis, turning mode, units shown.", "colour color point size wireframe grid display "
            "options heatmap", {"screen": "workspace"}),
    Feature("view.fullscreen", "Full screen 3D view", "the expand button on the right of the 3D view",
            "Shows only the 3D view (Esc to leave).", "fullscreen maximize big", {"screen": "workspace"}),
    Feature("view.picture", "Save a picture of the view", "the camera button on the right of the 3D view",
            "Saves (and copies) a picture of the 3D view.", "screenshot image snapshot capture picture",
            {"screen": "workspace"}),
    Feature("view.record", "Record a video of the view", "the red dot button on the right of the 3D view",
            "Records the 3D view as a video while you turn it.", "video recording movie", {"screen": "workspace"}),
    Feature("size-tag", "Part size", "the card at the top left of the 3D view",
            "Length × width × height along the part itself (Part) or along the scanner's axes (XYZ); the ruler "
            "draws them on the model.", "dimensions size length width height bounding box", {"screen": "workspace"}),
    # ---- Scan
    Feature("capture.scanner", "Scanner", "Scan → Scanner",
            "Choose the scanner: MetroY by USB (native on the DGX), the Revo Metro bridge (scan in Revo Metro on "
            "the PC, exports arrive here) or the simulated scanner for practice; then start scanning.",
            "scanner metroy revo metro usb connect start scanning capture bridge", _s("capture")),
    Feature("capture.turntable", "Turntable", "Scan → Turntable",
            "Connect the turntable over Bluetooth, turn and tilt it, set the speed, and run stop-and-go scan "
            "programs synced with the scanner.", "turntable rotate turn tilt spin bluetooth table program",
            _s("capture")),
    Feature("capture.guidance", "Guidance", "Scan → Guidance (while scanning), plus the gauges on the 3D view",
            "Live coverage, distance, speed, tracking and density, and which areas still need scanning.",
            "coverage holes missing distance speed tracking live help", _s("capture")),
    Feature("capture.save", "Save this scan", "Scan → Save this scan (after scanning)",
            "Saves the scan into the project, optionally processing it automatically.", "save keep finish scan",
            _s("capture")),
    Feature("capture.markers", "Marker map", "Scan → Marker map",
            "Maps the marker stickers on or around the part first, so later scans line up on them.",
            "markers stickers targets map global markers", _s("capture")),
    Feature("photos-to-3d", "Make a 3D model from photos", "Scan → Make a 3D model from photos (or ask the "
            "assistant)", "Turns photos of the part into a coloured 3D model in about a minute (COLMAP places the "
            "photos, MapAnything builds the surface). Photos needed: 12 or more, about every 30 degrees all the way "
            "round at the same height; 24-36 at 2-3 heights rebuild the most; 3-4 give only a rough model. Good for "
            "the shape, not for measuring holes or diameters (surface about 0.5-1 mm off). Photograph the part on "
            "the printed scale sheet and the model comes out at true size; without it, set the size from one known "
            "length (Clean → Set the true size).",
            "photos pictures images photogrammetry reconstruct 3d model from photos camera phone no scanner",
            _s("capture")),
    Feature("photos-to-3d.sheet", "Print the scale sheet", "Scan → Make a 3D model from photos → Print the scale "
            "sheet", "Photograph the part on this printed sheet and the photo model comes out at true size. Print "
            "at 100 % (actual size), measure the 100 mm bar with a caliper and type it in, tape the sheet flat and "
            "keep some of its black squares in every photo.",
            "scale sheet ruler true size markers aruco print calibration", _s("capture")),
    Feature("capture.files", "Or bring in files", "Scan → Or bring in files (or Models → + Add → Open files)",
            "Open scans exported from Revo Metro or any PLY / STL / OBJ / STEP file.", "import open files upload",
            _s("capture")),
    # ---- Clean
    Feature("clean.presets", "How thorough?", "Clean → How thorough?",
            "Removes stray points, noise and floating bits (light, standard or thorough). Points are only removed, "
            "never moved, so the part keeps its size.", "clean noise outliers stray points denoise filter",
            _s("clean")),
    Feature("clean.table", "Remove the table or turntable", "Clean → How thorough? → Remove the table or turntable",
            "Also removes the flat surface the part stood on.", "table floor turntable plane remove base",
            _s("clean")),
    Feature("clean.by-hand", "Remove things by hand", "Clean → Remove things by hand",
            "Select with a box or lasso, then Delete or Keep only.", "delete remove manual select erase",
            _s("clean")),
    Feature("clean.position", "Position & crop", "Clean → Position & crop",
            "Sit flat on the floor, move to the origin, line up with X/Y/Z, crop to a box, cut with a plane, "
            "remove loose bits.", "crop cut plane orient align axes floor origin rotate position", _s("clean")),
    Feature("clean.scale", "Set the true size", "Clean → Set the true size (at the top of Clean when the model "
            "was made from photos; for any other model: Clean → Position & crop → More edits → Position · Scale)",
            "Pick a length you know on the model (overall length/width/height or a measured distance), type its real "
            "length, and the whole model is scaled to match. For models made from photos, whose size is only "
            "estimated (models photographed on the printed scale sheet are already true size); scanner data is "
            "already true to size, so never scale it without a reason.",
            "scale model resize true size real dimension factor units calibrate ruler", _s("clean")),
    Feature("clean.smooth", "Smooth rough spots", "Clean → Smooth rough spots",
            "The smoothing brush and whole-model smoothing for rough patches.", "smooth rough noise brush",
            _s("clean")),
    # ---- Align
    Feature("align.scans", "Scans to combine", "Align → Scans to combine (tick scans in the Models list)",
            "Pick the scans of the same part to merge; the first is the reference. The crosshair button next to a "
            "scan lets you pick matching points by hand.", "merge combine stitch scans reference point pairs",
            _s("align")),
    Feature("align.check", "Check before merging", "Align → Check scans (the main button)",
            "Checks whether merging helps: each scan must add surface and line up exactly; warns about symmetric "
            "parts and doubled surfaces, then offers Merge / Use the best scan.",
            "merge check assess alignment quality symmetric doubled", _s("align")),
    Feature("align.stickers", "Line up on stickers", "Align → Line up on stickers",
            "Aligns scans on the marker stickers they share (3 or more): exact even when the part looks the same "
            "in several positions.", "stickers markers targets dots holes line up align exactly", _s("align")),
    Feature("align.photos", "Which way do the scans fit?", "Align → Which way do the scans fit? → Compare with my "
            "photos (with two scans ticked)", "When scans can line up in more than one way, the assistant draws "
            "each option and compares it with your reference photos of the part to pick the right one.",
            "photos reference images merge options candidates which way orientation compare", _s("align")),
    Feature("align.preview", "Preview before merging", "Align → Preview before merging",
            "Each scan in its own window plus one with them together, sharing the camera.",
            "preview side by side compare look", _s("align")),
    Feature("align.settings", "Alignment settings", "Align → Fine-tune (at the bottom)",
            "Advanced alignment parameters.", "advanced parameters icp settings", _s("align")),
    # ---- Mesh
    Feature("mesh.kind", "What kind of surface?", "Mesh → What kind of surface?",
            "True to the scan (only surface backed by points, best for measuring) or closed and watertight (for 3D "
            "printing); detail and smoothing.", "mesh surface stl watertight printing build triangulate", _s("mesh")),
    Feature("mesh.repair", "Repair & finish", "Mesh → Repair & finish",
            "Fill holes, repair the surface, remove loose bits, smooth, use fewer triangles, sit flat.",
            "repair fix fill holes simplify decimate smooth", _s("mesh")),
    Feature("mesh.holes", "Holes", "Mesh → Holes (with a mesh selected)",
            "Lists every hole in the mesh with its size; fill the ones you pick.", "holes gaps fill openings",
            _s("mesh")),
    Feature("mesh.colour", "Colour from photos", "Mesh → Colour from photos → Colour it from all photos (or line up "
            "one photo by hand)", "Gives your scan or mesh the real colours of the part from photos of it: CloudClean "
            "works out where each photo was taken, lines the photos up with your model by itself and paints them on "
            "(a mesh also gets a texture). The shape and size do not change. Photos needed: 12 or more, about every "
            "30 degrees all the way round at the same height; 6 every 60 degrees is the fewest that worked; 3-4 "
            "photos, big jumps in angle or height, or photos of one side only are refused.",
            "texture textured colour color colours colors paint photos pictures images real colours true colours",
            _s("mesh")),
    Feature("mesh.colour-by-hand", "Line up one photo by hand", "Mesh → Colour from photos → Line up one photo by "
            "hand", "The manual way to colour a mesh: line each photo up with the mesh in the 3D view, save the view, "
            "then apply. For when the automatic colouring cannot place the photos.",
            "manual align photo overlay camera view by hand texture", _s("mesh")),
    # ---- Measure
    Feature("measure.part-size", "Part size", "Measure → Dimensions → Part size",
            "Length × width × height along the part itself, with stray points ignored; Draw on model shows them.",
            "dimensions size length width height overall", _m("dimensions")),
    Feature("measure.p2p", "Point to point", "Measure → Dimensions → Point to point (or key M)",
            "Click two points for the distance between them, optionally along X/Y/Z or the thread axis.",
            "distance two points ruler click", _m("dimensions", "p2p")),
    Feature("measure.caliper", "Caliper", "Measure → Dimensions → Caliper",
            "The distance between two opposite faces, like calipers: planes are fitted to both faces.",
            "caliper across width thickness face to face jaws", _m("dimensions", "across")),
    Feature("measure.overall", "Overall size", "Measure → Dimensions → Overall size",
            "End to end in one direction, stray points ignored.", "extent overall length end to end longest",
            _m("dimensions", "extent")),
    Feature("measure.heights", "Heights & steps", "Measure → Dimensions → Heights & steps",
            "Finds every flat face along the part (head top, shoulder, tip, floor of a socket) and the face-to-face "
            "distances: head height, length under the head, recess depth, overall.",
            "head height shoulder step height gauge length under head recess depth socket", _m("dimensions", "steps")),
    Feature("measure.diameter", "Diameter", "Measure → Dimensions → Diameter",
            "Fits a cylinder to a round surface you select (shank, pin, hole) and gives its diameter.",
            "diameter radius round cylinder hole pin shank bore", _m("dimensions", "diameter")),
    Feature("measure.angle", "Angle", "Measure → Dimensions → Angle",
            "The angle between two flat faces you select.", "angle degrees chamfer taper", _m("dimensions", "angle")),
    Feature("measure.flatness", "Flatness", "Measure → Dimensions → Flatness",
            "How flat a selected face is.", "flatness flat plane warp", _m("dimensions", "flatness")),
    Feature("measure.sphere", "Ball / sphere", "Measure → Dimensions → Ball / sphere",
            "Fits a ball to a round, dome-shaped surface you select.", "sphere ball dome radius",
            _m("dimensions", "sphere")),
    Feature("measure.section", "Cross-section", "Measure → Dimensions → Cross-section",
            "Cuts across the part and draws the outline with its width and height.",
            "section cut profile slice outline", _m("dimensions", "section")),
    Feature("measure.results", "Measurement list", "Measure → Dimensions → the list under the tools",
            "Every measurement with copy, CSV and show-on-model.", "results list copy csv table measurements",
            _m("dimensions")),
    Feature("measure.thread", "Thread", "Measure → Thread",
            "Pitch, major / minor / pitch diameter and the nearest standard size (ISO metric, UNC, UNF, BSP).",
            "thread pitch tpi screw bolt standard unc metric", _m("thread")),
    Feature("measure.cad", "Golden model", "Measure → Golden model",
            "Checks a scan against the golden model (the part as it should be: its CAD file or a trusted mesh): "
            "what was not scanned or is off, what to scan again and how, and every size of the golden model "
            "measured on the scan with ok / off.",
            "golden model master reference cad compare deviation inspection tolerance heatmap step stl rescan "
            "missing not scanned coverage measurements match pass fail", _m("cad")),
    Feature("golden.verdict", "Golden check verdict", "Measure → Golden model → the box at the top of a result",
            "Matches, Mostly matches (95 % or more of the surface within the tolerance, a few things to look at), "
            "Does not match, or Scan more; the ring shows how much of the surface matches, the three numbers under "
            "it how much was scanned, how many sizes match and how many areas to look at.",
            "verdict result pass fail percent match score summary how good", _m("cad")),
    Feature("golden.areas", "Areas to look at", "Measure → Golden model → Areas to look at",
            "Every problem area of the check, numbered like the pins on the 3D view: Different from the golden "
            "model (less or more material than designed, with a bar against the tolerance) and Scan these again "
            "(not scanned, too few points, rough), each with what it means and what to do.",
            "problem areas list regions off different rescan scan again pins numbers", _m("cad")),
    Feature("golden.show-me", "Show me", "Measure → Golden model → Areas to look at → Show me on an area",
            "Turns the 3D view to that area, greys out the rest of the part so only the area keeps its colour, and "
            "marks its pin. Whole part goes back to the full view. Clicking a numbered pin on the model does the same.",
            "show me zoom focus where locate find area highlight pin", _m("cad")),
    Feature("golden.colour-by", "Colour the model by", "Measure → Golden model → The surface → Colour the model by",
            "What was found (the check's colours), Distance (blue less material, red more) or Scan points (every scan "
            "point coloured by its distance).", "colour color heatmap deviation distance view map", _m("cad")),
    Feature("measure.fill-photos", "Fill from photos", "Measure → Golden model → Scan these again → Fill from photos",
            "Fills the areas a scan missed with points from photos of the part, when it cannot be scanned again. "
            "Photos are only good to about 1-2 mm, so filled areas complete the model but are not measured.",
            "fill gaps holes missing photos images complete model rescan cannot scan", _m("cad")),
    Feature("measure.accuracy", "Accuracy", "Measure → Accuracy",
            "Tells the scanner apart from the software when a size looks off: did processing change the size, is "
            "the scanner repeatable, and a check against a known size.", "accuracy error off wrong calibration "
            "scanner software trust", _m("accuracy")),
    Feature("measure.known-size", "Check against a known size", "Measure → Accuracy → Check against a known size",
            "Scan a gauge block, ball bar or anything of certified size and see the scanner's error.",
            "gauge block ball bar calibration reference known size", _m("accuracy")),
    Feature("measure.repeatability", "Is the scanner repeatable?",
            "Measure → Accuracy → Is the scanner repeatable?",
            "Compares two scans of the same part.", "repeatability compare two scans consistency", _m("accuracy")),
    # ---- Export
    Feature("export.download", "Download to this computer", "Export → Download to this computer",
            "PLY, STL, OBJ, GLB or 3MF of the current model.", "download save stl obj ply glb 3mf file export",
            _s("export")),
    Feature("export.reports", "Reports", "Export → Reports",
            "The measurement report (print or save as PDF), the processing report and the measurements as CSV.",
            "report pdf print csv spreadsheet", _s("export")),
    Feature("export.server", "Save on the server", "Export → Save on the server",
            "Writes the file straight to a folder or network share on the machine running CloudClean.",
            "server folder network share nas disk", _s("export")),
]

BY_ID = {f.id: f for f in FEATURES}
_WORD = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "to", "of", "and", "or", "in", "on", "for", "is", "it", "i", "my", "me", "how", "do",
         "where", "what", "can", "find", "show", "open", "go", "get", "use", "with", "this", "that", "want", "need"}


def _words(text: str) -> list[str]:
    out = []
    for w in _WORD.findall(text.lower()):
        if w in _STOP:
            continue
        out.append(w[:-1] if len(w) > 4 and w.endswith("s") else w)  # plurals: holes -> hole
    return out


def search(query: str, limit: int = 5) -> list[tuple[float, Feature]]:
    """Features ranked for a question in plain words: name words count most, then keywords, then the rest."""
    q = _words(query)
    if not q:
        return []
    ranked = []
    for f in FEATURES:
        name, keys, rest = set(_words(f.name)), set(_words(f.keywords)), set(_words(f.what + " " + f.where))
        score = sum(3.0 if w in name else 2.0 if w in keys else 1.0 if w in rest else 0.0 for w in q)
        if f.name.lower() in query.lower():
            score += 5.0
        if score > 0:
            ranked.append((score, f))
    ranked.sort(key=lambda x: -x[0])
    return ranked[:limit]


def feature_index() -> str:
    """Compact "id (name)" list for the tool description, so the model can name a feature exactly."""
    return ", ".join(f"{f.id} ({f.name})" for f in FEATURES)
