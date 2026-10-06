# Running CloudClean on an NVIDIA DGX Spark

The Spark (GB10, ARM64, DGX OS / Ubuntu, 128 GB unified memory) runs three things:

1. **CloudClean** – the processing engine and web app, opened from any browser on the LAN.
2. **An LLM server** for the assistant – vLLM, SGLang or Ollama, OpenAI-compatible API.
3. (optional) **Autopilot** watching a shared scan folder.

The scanner is either plugged into the Spark (native MetroY USB driver) or into a scanning PC running Revo Metro,
where the small **bridge** pushes every export to the Spark the moment it is written.

```
 scanning PC                                  DGX Spark
 ┌──────────────────────────┐   HTTP upload   ┌───────────────────────────────────────────┐
 │ Revo Metro → export PLY  │ ──────────────▶ │ cloudclean serve  :8765  (UI + engine)     │
 │ cloudclean bridge        │                 │   ├─ capture session, autopilot, inspect   │
 └──────────────────────────┘                 │   └─ assistant ──▶ LLM server (vLLM …)     │
 any browser ─────────────────────────────▶   └───────────────────────────────────────────┘
```

## 1. Install CloudClean (DGX OS / Ubuntu 24.04)

```bash
git clone https://github.com/pranch555/CloudClean.git ~/code/CloudClean
cd ~/code/CloudClean
deploy/install-service.sh          # sets up .venv (~5 min the first time), then runs CloudClean as a service on :8765
```

Open `http://<spark-name-or-ip>:8765` from any machine on your network (with Tailscale, the Spark's tailnet name).
The first account you create is the admin (`docs/accounts.md`). To try it without a service, `./cloudclean.sh --host 0.0.0.0`.

`install-service.sh` writes `~/.config/systemd/user/cloudclean.service` (see `--print`), enables lingering so it runs
without a login, and takes `--port` and `--workspace` (default `~/cloudclean-workspace`). Day to day:

```bash
systemctl --user status cloudclean
journalctl --user -u cloudclean -f
deploy/update.sh               # update to the newest code on GitHub now (only while no capture, job or chat is running)
deploy/update.sh --check       # is an update waiting, and is CloudClean busy?
```

To have the Spark follow GitHub by itself, run `deploy/install-autoupdate.sh` once: a timer runs `update.sh` every
2 minutes, so a `git push` from the PC you develop on reaches the Spark within a few minutes. It waits while a
capture, a job or an assistant reply is running, saves the old code to `~/cc-backups/` first, installs new
requirements when `pyproject.toml` changed, and puts the old code back if CloudClean does not start. It never
overwrites files edited by hand on the Spark. Log: `journalctl --user -u cloudclean-update`; off: `--off`.

The browser UI is prebuilt into `cloudclean/web/static`, so Node is **not** needed on the Spark.

ARM64 differences CloudClean handles automatically:

- **Open3D needs `libgfortran.so.5`**, which DGX OS does not ship. `cloudclean.sh setup` copies the one bundled with
  numpy into Open3D's folder (no sudo). `sudo apt install libgfortran5` fixes it system-wide instead.
- **Poisson meshing runs single-threaded.** Open3D's multi-threaded Poisson solver fails ("Failed to close loop") or
  segfaults in the aarch64 build. One thread is still quick: depth 11 on 1 M points takes ~30 s. Override with
  `CLOUDCLEAN_POISSON_THREADS`.
- **UV texture baking uses a numpy baker.** `ProjectImagesToAlbedo` is x86-only in Open3D; vertex colouring was never
  affected. Force either path with `CLOUDCLEAN_TEXTURE_BAKER=native|numpy`.
- Open3D from pip is CPU-only on ARM64; the Grace CPU's 20 cores handle scans of several million points.

## 2. LLM server

The assistant talks to any OpenAI-compatible `/v1` endpoint. Set it in **Settings → Assistant** (server address, model,
key), or with `CLOUDCLEAN_LLM_BASE_URL`, `CLOUDCLEAN_LLM_MODEL` and `CLOUDCLEAN_LLM_API_KEY` in the service's
environment (`install-service.sh` copies them from your shell). Without a server, everything except the chat works.

Tested on a Spark:

- **NVIDIA Nemotron 3.5 Lightning 30B-A3B (NVFP4)** with the **DSpark** speculative-decoding drafter, NVIDIA's
  "1x DGX Spark" recipe, GPU memory capped at 50 % so CloudClean keeps ~55 GB for large scans. Fastest; text only.

  ```bash
  docker run -d --name cloudclean-llm --restart unless-stopped --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:v0.27.1 \
    --model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 --served-model-name nemotron-3.5-lightning-30b \
    --moe-backend marlin --kv-cache-dtype fp8 --enable-prefix-caching \
    --gpu-memory-utilization 0.5 --max-model-len 262144 \
    --speculative_config.num_speculative_tokens 3 \
    --speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark \
    --mamba-backend flashinfer --mamba-cache-mode align \
    --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder --enable-auto-tool-choice
  ```

- **Qwen3.8-27B NVFP4** on SGLang: vision-capable (the assistant can look at your photos of the part). It also works
  with a 16k context window and without a tool-call parser (see `docs/assistant-small-models.md`).

Measured with CloudClean's assistant on a Spark (same requests):

| Model / server | "Clean scan X and show it" (whole turn incl. the clean job) | Notes |
|---|---|---|
| Qwen3.8-27B BF16, vLLM, thinking on | first action after 162 s | 4.6 tokens/s |
| Qwen3.8-27B NVFP4, SGLang + n-gram speculation, thinking off | 33 s | vision |
| **Nemotron 3.5 Lightning 30B-A3B NVFP4, vLLM + DSpark, thinking off** | **10.6 s** | text only |

The assistant sends `enable_thinking: false` by default (Assistant → gear → *Deep thinking* turns reasoning on).

Models checked and not used (Sep 2026): **Qwen3.8-Flash-Next NVFP4** needs ~133 GB of weights (does not fit 121 GB);
**GLM-5.3-Flash** (169 B) and **DeepSeek-V4-Flash** (304 B) are too large to share one Spark with scan processing.
If other model servers already use the GPU, lower `--gpu-memory-utilization` so everything fits.

## 3. Photos → 3D (optional)

Photos → 3D and COLMAP camera placement for *Colour from photos* run in a GPU container that CloudClean starts
itself. Build it once on the Spark (see `docs/photos-to-3d.md`):

```bash
mkdir -p ~/cloudclean-deploy/recon && cd ~/cloudclean-deploy/recon
git clone https://github.com/facebookresearch/map-anything
cp ~/code/CloudClean/tools/recon/Dockerfile . && docker build -t cloudclean-recon:latest .
```

## 4. Scanner → Spark with no manual upload

Revopoint publishes no SDK for the MetroY Ultra, so CloudClean drives it itself: plug the scanner into the Spark's
USB and use **Scan → MetroY by USB** (`docs/metroy-protocol.md`); no Revo Metro, no PC. Once per machine, let
CloudClean open the scanner even when nobody is logged in at the Spark's own screen:

```bash
sudo deploy/install-scanner-access.sh     # a udev rule (plugdev) for the scanner's command channel and camera
```

Without it Linux hands the scanner only to the user logged in at the screen (on a headless Spark that is the login
screen), and the Scan step says so. Native capture is Linux-only for now.

The fallback: keep Revo Metro on a Windows/macOS PC and let the bridge upload every export. On the scanning PC (only
needs Python + `httpx`):

```powershell
pip install httpx
python -m cloudclean.capture.bridge --server http://<spark>:8765 --watch "D:\RevoMetro\Exports" --key <key>
# or, with CloudClean installed: cloudclean bridge --server http://<spark>:8765 --watch "D:\RevoMetro\Exports" --key <key>
# the key: CloudClean -> Settings -> Users (admin), or Scan -> Or bring in files shows the whole command
```

Set Revo Metro's export folder to the watched folder. Every exported scan appears live in **Capture** on the Spark,
gets coverage/hole guidance ("flip the part and scan the underside"), and – with *Run when a capture is saved* on in
Autopilot – comes out as a cleaned, merged, meshed STL.

If Revopoint grants you SDK access (email customer@revopoint3d.com with the scanner serial number), set
`REVOPOINT_SDK_PATH` and the `revopoint_sdk` driver in `cloudclean/capture/drivers/revopoint_sdk.py` is the place to
wire the real calls; the live guidance, fusion and UI already work frame-by-frame (see the `simulated` driver).

Alternatively, share the export folder over SMB/NFS and point **Autopilot → Watch folder** at it.

## 5. Security

Everyone signs in with their own account. Keep port 8765 on your LAN or tailnet (Tailscale limits it to your own
devices); do not expose it to the internet.

## 6. Developing the UI

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173, proxies /api to cloudclean serve on :8765
npm run build      # type-checks and writes cloudclean/web/static
```
