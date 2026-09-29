# CloudClean web app in a container (Linux x86-64 or ARM64, e.g. a DGX Spark). See "Docker" in README.md:
#   docker compose up -d        then open http://localhost:8765
# Not in the container: live USB scanner capture, Bluetooth turntables and Photos -> 3D (it starts its own GPU
# container); run CloudClean natively (cloudclean.sh) for those.
FROM python:3.12-slim

# Open3D's native libraries (libgfortran5, libidn2-0 on ARM64); libgl1 / libegl1 / libglib2.0-0: Open3D and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgfortran5 libidn2-0 libgl1 libegl1 \
      libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY cloudclean ./cloudclean
RUN pip install --no-cache-dir --disable-pip-version-check ".[all]" && python -c "import open3d, cloudclean.mesh"

ENV PYTHONUNBUFFERED=1
EXPOSE 8765
VOLUME /workspace
CMD ["cloudclean", "serve", "--host", "0.0.0.0", "--port", "8765", "--workspace", "/workspace", "--no-browser"]
