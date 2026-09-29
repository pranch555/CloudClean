# Assistant: app guide, reference photos, photo-guided merges

Three assistant features added on 2026-09-24. They share one idea: the assistant is where people ask, so it must
know the app as the user sees it, see what the user shows it, and prove its choices with pictures.

## App guide (tool `app_guide`)

"Where do I measure the head height?" → the assistant answers with the path in words
("Measure → Dimensions → Heights & steps"), and the app opens that place and rings the control.
There is no separate search: the old command palette was removed on request, and Ctrl K now puts the cursor in the
nearest Ask box (Home prompt, the conversation, or the Ask bar).

- **Catalogue:** `cloudclean/assistant/guide.py`, `FEATURES`. Each entry has:
  - `id`
  - `name`: exactly the label on screen
  - `where`: the path, in the words of the UI
  - `what`: one or two sentences
  - `keywords`: the words people use for it
  - `nav`: how to get there; screen, step, right panel tab, left panel, measure tab and tool, settings section, jobs
- **Search:** `search(question)` ranks features by words in the name first, then keywords, then the rest.
- **The tool:** `app_guide {question | feature, open?}`
  - returns the matches (id, name, where, what)
  - when one match is clear, it sends the UI event `{"action": "guide", feature, label, where, nav}`
  - the tool description lists every `id (name)`, so the model can name a feature exactly
- **In the web app:**
  - `lib/guide.ts` `guideTo()` follows `nav`.
  - It then waits up to 2.5 s for the element with `data-guide~="<id>"`.
  - `ui/GuideSpotlight.tsx` rings that element for 10 s, with its name and path. The ring goes away on click or
    Esc.
  - If the control is not on screen (e.g. Guidance only exists while scanning), a note with the path shows instead.
- **Adding a feature:**
  1. Add a `Feature(...)` with the on-screen name.
  2. Put `data-guide="<id>"` on its control. `Block`, `ToolTile` and `Segmented` options take a `guide` prop; any
     other element takes the attribute directly.
  3. `tests/test_assistant.py::test_app_guide_*` checks ids are unique and common questions resolve.
- **Naming rule:** a feature's label says what it does in the user's words ("Heights & steps", "Caliper",
  "Overall size", "Cross-section", "Ball / sphere", "Which way do the scans fit?"). The guide, the README and the
  prompt use the same names.

## Reference photos (Photos button, tool `look_at_photos`)

- **Where you add photos:** every chat box has a **Photos** button: the assistant panel, the Ask bar and the Home
  prompt. Paste and drop work too. The menu offers photos from this computer, a picture of the 3D view, and photos
  already saved in the project.
- **Keeping them:** photos from this computer are saved in the project at full resolution, as `kind: image` assets
  (switch "Keep new photos in the project", default on). Their caption then names the photo asset. The upload is
  `POST /api/upload` with the project id, before the message is sent.
- **Seeing them again:** the model sees attached images only in their own message. `look_at_photos {asset_ids? |
  project?}` shows saved photos again: the newest 4 of the current project by default, at most 1280 px, EXIF-upright
  JPEG.
- **How the images reach the model:**
  - Tools return images through `ToolResult.images`.
  - The agent adds them as a user message with image parts after the tool results, for the rest of that turn only.
  - They are never stored in the conversation file.
- **The state block** says how many reference photos the project has.

## Photo-guided merge options (tool `compare_merge_options`)

A symmetric part can line up in more than one pose that fits the surface almost equally well. On the bolt, scan 06
fits the best pose, one turned 92°, and two turned 180° that put a head at both ends.

- **Finding the options:** `cloudclean/merge_views.py` `merge_options(ref, moving)` collects:
  - the best pose from `register.align_pair`
  - its `alternatives`
  - the stickers' pose, when both scans share 3+ stickers (it comes through `align_pair`, where it wins when it fits
    the surface)

  Each is refined with ICP; duplicates within 2° and 0.5 mm are dropped. Every option comes with overlap, gap, and
  how far it is turned and shifted from the best pose.
- **Drawing them:** `render_option()` draws one 1440×834 picture per option:
  - top row: three grey views (side, end, angle, along the part's own axes), to compare with photos
  - bottom row: the same views coloured by scan (orange = reference, blue = moved scan)

  It is a numpy splat renderer (no GPU or display needed on the DGX) and takes about 0.6 s per option.
- **Job and routes:**
  - The job `merge_options` saves `workspace/renders/<key>/<A..D>.jpg`.
  - `POST /api/merge/options` starts the job.
  - `GET /api/merge/options/<key>/<A>.jpg` serves the pictures.
- **The assistant flow:**
  1. `compare_merge_options {asset_ids: [reference, moving], photo_ids?, use_photos?}` runs the job. It shows the
     model the option pictures plus up to 2 reference photos, and shows the same pictures in the chat (UI event
     `{"action": "images", title, images}`).
  2. The model says which option matches the real part and why.
  3. Once the user agrees, `merge {options_job_id, option}` merges with exactly that pose. The merged asset's report
     records `chosen_option`.
- **Where in the app:** Align → Which way do the scans fit? → **Compare with my photos**, shown when two scans are
  ticked.
- **Limits:** photos rule out options that look different: flipped, the wrong end joined, parts that do not join.
  They cannot resolve sub-millimetre ambiguities such as one head flat of a 12-sided head plus a twelfth of a thread
  pitch. Stickers or a caliper check settle those, and the assistant is told to say so.
- **Tests:** `tests/test_merge_views.py`: the true pose is among the options, options are distinct, and every panel
  is drawn.
