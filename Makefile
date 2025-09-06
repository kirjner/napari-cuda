.PHONY: typestubs pre watch dist settings-schema
.PHONY: install-cuda setup-gl install-gl verify-gl clean-gl test-nvenc hpc-setup help

typestubs:
	python -m napari.utils.stubgen

# note: much faster to run mypy as daemon,
# dmypy run -- ...
# https://mypy.readthedocs.io/en/stable/mypy_daemon.html
typecheck:
	tox -e mypy

check-manifest:
	pip install -U check-manifest
	check-manifest

dist: typestubs check-manifest
	pip install -U build
	python -m build

settings-schema:
	python -m napari.settings._napari_settings

pre:
	pre-commit run -a

# If the first argument is "watch"...
ifeq (watch,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "watch"
  WATCH_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(WATCH_ARGS):;@:)
endif

# examples:
# make watch ~/Desktop/Untitled.png
# make watch -- -w animation  # -- is required for passing flags to napari

watch:
	@echo "running: napari $(WATCH_ARGS)"
	@echo "Save any file to restart napari\nCtrl-C to stop..\n" && \
		watchmedo auto-restart -R \
			--ignore-patterns="*.pyc*" -D \
			--signal SIGKILL \
			napari -- $(WATCH_ARGS) || \
		echo "please run 'pip install watchdog[watchmedo]'"

# =============================================================================
# HPC: CUDA-OpenGL interop setup (PyCUDA with GL support)
# =============================================================================

# Configuration
GL_PREFIX := ./vendor/gl
SSL_CERT_FILE := /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
GL_HEADERS := $(GL_PREFIX)/include/GL/gl.h
GL_LIBGL := $(GL_PREFIX)/lib/libGL.so
CUDA_ENABLE_GL := True
# Help target
help:
	@echo "napari-cuda Makefile targets:"
	@echo ""
	@echo "Standard napari targets:"
	@echo "  make typestubs     - Generate type stubs"
	@echo "  make typecheck     - Run mypy type checking"
	@echo "  make pre           - Run pre-commit hooks"
	@echo "  make dist          - Build distribution"
	@echo ""
	@echo "HPC/CUDA targets:"
	@echo "  make install-cuda  - Install Python deps with CUDA support (fast)"
	@echo "  make install-gl    - Install with PyCUDA GL support (rebuilds from source)"
	@echo "  make verify-gl     - Verify PyCUDA GL and EGL are working"
	@echo "  make setup-gl      - One-time GL headers/libs setup"
	@echo "  make test-nvenc    - Test PyNvVideoCodec NVENC functionality"
	@echo "  make clean-gl      - Remove GL setup and PyCUDA"
	@echo "  make hpc-setup     - Complete HPC setup with CUDA-GL interop"
	@echo ""
	@echo "Prerequisites for HPC targets:"
	@echo "  module load cuda/12.4.0"
	@echo "  module load miniforge/24.3.0-0"

# Fast path: install Python deps (server + cuda extras)
install-cuda:
	uv sync --extra server --extra cuda

# One-time: install GL dev headers/libs into user prefix via conda-forge
setup-gl:
	@command -v nvcc >/dev/null 2>&1 || { \
		echo "ERROR: 'nvcc' not found. Load CUDA module first."; \
		echo "       e.g., 'module load cuda/12.4.0'"; \
		exit 1; \
	}
	@command -v mamba >/dev/null 2>&1 || { \
		echo "ERROR: 'mamba' not found. Load your miniforge/mambaforge module first."; \
		echo "       e.g., 'module load miniforge/24.3.0-0'"; \
		exit 1; \
	}
	@if [ -f "$(GL_HEADERS)" ] && [ -e "$(GL_LIBGL)" ]; then \
		echo "GL dev already present at $(GL_PREFIX)"; \
	else \
		echo "Installing GL dev files to $(GL_PREFIX)..."; \
		mamba create -y -p $(GL_PREFIX) -c conda-forge \
		  mesalib libglu xorg-libx11 xorg-libxext xorg-libxfixes xorg-libxau xorg-libxdmcp; \
	fi

# Rebuild PyCUDA from source with GL support detected via headers/libs above
install-gl: setup-gl 
	@echo "Cleaning environment variables that might interfere..."
	SSL_CERT_FILE=$(SSL_CERT_FILE) \
	REQUESTS_CA_BUNDLE=$(SSL_CERT_FILE) \
	GL_PREFIX=$(GL_PREFIX) \
	CPATH=$(GL_PREFIX)/include \
	LIBRARY_PATH=$(GL_PREFIX)/lib \
	LD_LIBRARY_PATH=$(GL_PREFIX)/lib:$$LD_LIBRARY_PATH \
	uv pip install --project . --force-reinstall --no-binary=pycuda --no-cache pycuda
	uv sync --extra server --extra cuda --no-cache
	$(MAKE) verify-gl

# Verify PyCUDA GL and EGL functionality
verify-gl:
	@echo "Verifying pycuda.gl and EGL..."
	uv run python -c "\
import os; \
os.environ.setdefault('QT_QPA_PLATFORM','offscreen'); \
os.environ.setdefault('PYOPENGL_PLATFORM','egl'); \
import pycuda.driver as cuda; \
import pycuda.gl as gl; \
cuda.init(); \
print(f'CUDA device: {cuda.Device(0).name()}'); \
print('pycuda.gl import: OK'); \
from OpenGL.EGL import *; \
import ctypes; \
display = eglGetDisplay(EGL_DEFAULT_DISPLAY); \
major = ctypes.c_int(); \
minor = ctypes.c_int(); \
eglInitialize(display, major, minor) and print(f'EGL {major.value}.{minor.value} initialized: OK') and eglTerminate(display)"
	@echo "✓ GL/EGL verification passed"

# Test PyNvVideoCodec NVENC functionality
test-nvenc:
	@echo "Testing PyNvVideoCodec NVENC..."
	uv run python scripts/test_pynv_simple.py

# Clean GL setup
clean-gl:
	@echo "Removing GL development files..."
	rm -rf $(GL_PREFIX)
	@echo "Uninstalling PyCUDA..."
	uv pip uninstall --project . pycuda || true
	@echo "Clearing uv cache..."
	uv cache clean pycuda || true
	@echo "✓ GL cleanup complete"

# Complete HPC setup with verification
hpc-setup: setup-gl install-gl verify-gl test-nvenc
	@echo ""
	@echo "✅ HPC setup complete with CUDA-GL interop!"
	@echo ""
	@echo "Environment configured with:"
	@echo "  - CUDA 12.4 support"
	@echo "  - PyCUDA with OpenGL interop"
	@echo "  - PyNvVideoCodec for NVENC"
	@echo "  - Headless EGL rendering"
	@echo ""
	@echo "Ready to run: uv run napari-cuda-server data.npy"
