# cloudclean-recon: photos -> cameras (COLMAP, CUDA) + dense points (MapAnything) on the DGX Spark (aarch64, GB10).
# See docs/photos-to-3d.md. Build on the Spark, next to a clone of MapAnything:
#   cd ~/cloudclean-deploy/recon && git clone https://github.com/facebookresearch/map-anything
#   cp <repo>/tools/recon/Dockerfile . && docker build -t cloudclean-recon:latest .
# The vLLM image is used as the base because it already has a CUDA 13 PyTorch that runs on the GB10.
FROM vllm/vllm-openai:cu130-nightly

# MapAnything (Apache-2.0 checkpoint facebook/map-anything-apache, downloaded to /hf on first use)
COPY map-anything /opt/map-anything
RUN python3 -m pip install --no-cache-dir -e /opt/map-anything && python3 -c "import mapanything, torch; print(torch.__version__, torch.cuda.is_available())"

# COLMAP 3.9.1 with CUDA (sm_121) for GPU SIFT and matching. No aarch64 pycolmap wheel exists, and Ubuntu's own
# colmap package is CPU only (~20x slower). Its dense PatchMatch stereo gave empty depth maps on the GB10: NVCC
# miscompiles the PatchMatch kernels for sm_100+ (Blackwell), so, as upstream does since colmap PR #4213, the
# PatchMatch library alone is built as sm_90 PTX, which the GB10 compiles when it first runs. Tested 2026-09-25 in
# the tag cloudclean-recon:pm-test (this build, plus the line below): PatchMatch then works (docs/photos-to-3d.md,
# Accuracy with the scale sheet). The pipeline still takes its dense points from MapAnything.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git cmake ninja-build \
      build-essential libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev \
      libeigen3-dev libflann-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgtest-dev libsqlite3-dev \
      libglew-dev qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch 3.9.1 https://github.com/colmap/colmap /opt/colmap-src \
 && printf '\nif(CUDA_ENABLED)\n    set_target_properties(colmap_mvs_cuda PROPERTIES CUDA_ARCHITECTURES "90-virtual")\nendif()\n' \
      >> /opt/colmap-src/src/colmap/mvs/CMakeLists.txt \
 && mkdir /opt/colmap-src/build && cd /opt/colmap-src/build \
 && cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCUDA_ENABLED=ON -DCMAKE_CUDA_ARCHITECTURES=121 -DGUI_ENABLED=OFF \
      -DOPENGL_ENABLED=OFF -DTESTS_ENABLED=OFF -DCMAKE_INSTALL_PREFIX=/opt/colmap \
 && ninja -j 18 && ninja install && rm -rf /opt/colmap-src

ENV PATH=/opt/colmap/bin:$PATH QT_QPA_PLATFORM=offscreen PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/hf
ENTRYPOINT []
