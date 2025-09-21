# Multi-stage build: Start with PyTorch image, then add OpenHands components
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel as pytorch-base

# Stage 2: Get OpenHands runtime components
FROM docker.all-hands.dev/all-hands-ai/runtime:0.56.0-nikolaik as openhands-runtime

# Final stage: Combine PyTorch with OpenHands runtime
FROM pytorch-base

# Copy essential tools and configurations from OpenHands runtime
COPY --from=openhands-runtime /usr/local/bin/ /usr/local/bin/
COPY --from=openhands-runtime /root/.bashrc /root/.bashrc
COPY --from=openhands-runtime /etc/apt/sources.list* /etc/apt/

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SDL_VIDEODRIVER=dummy
ENV XDG_RUNTIME_DIR=/tmp/runtime-docker
ENV CUDA_LAUNCH_BLOCKING=0

# Create runtime directory for headless operation
RUN mkdir -p /tmp/runtime-docker && chmod 700 /tmp/runtime-docker

# Install system dependencies for both npp-rl and nclone
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Basic system tools
    ca-certificates \
    curl \
    wget \
    gnupg \
    lsb-release \
    git \
    build-essential \
    pkg-config \
    netcat-openbsd \
    zip \
    # Python development dependencies
    python3-dev \
    python3-pip \
    python3-venv \
    # Graphics and multimedia dependencies (for pygame, opencv, pycairo)
    libcairo2-dev \
    libgirepository1.0-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev \
    libjpeg-dev \
    libpng-dev \
    # OpenCV dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Additional media libraries
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools (no --break-system-packages needed with PyTorch base)
RUN python -m pip install --upgrade pip setuptools wheel setuptools_scm[toml]

# Set pip configuration to avoid backtracking issues
ENV PIP_NO_BUILD_ISOLATION=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /workspace

# Clone and install nclone first (npp-rl depends on it)
RUN git clone https://github.com/tetramputechture/nclone.git /workspace/nclone
WORKDIR /workspace/nclone

# PyTorch base uses Python 3.12, so no need to modify requirements
# Install shared dependencies with resolved versions to avoid conflicts
RUN pip install --no-cache-dir \
    "opencv-python>=4.12.0" \
    "pillow>=11.3.0" \
    "gymnasium>=0.29.0" \
    "albumentations>=2.0.8" \
    "pytest>=8.4.1"

# Install nclone-specific dependencies
RUN pip install --no-cache-dir \
    pygame>=2.1.0 \
    fastmcp>=2.12.0 \
    pycairo>=1.28.0

# Install npp-rl-specific dependencies  
RUN pip install --no-cache-dir \
    stable-baselines3>=2.1.0 \
    sb3-contrib>=2.0.0 \
    optuna>=3.3.0 \
    tensorboard>=2.14.0 \
    imageio>=2.31.0

# Install nclone core dependencies first (without optional deps to avoid conflicts)
RUN pip install --no-cache-dir -e . --no-deps

# Install nclone optional dev/test dependencies separately
RUN pip install --no-cache-dir \
    pytest-cov>=3.0.0 \
    pytest-xdist>=2.5.0 \
    black>=22.0.0 \
    isort>=5.10.0 \
    flake8>=4.0.0 \
    mypy>=0.950 \
    ruff>=0.13.0

# Clone and install npp-rl
WORKDIR /workspace
RUN git clone https://github.com/tetramputechture/npp-rl.git /workspace/npp-rl
WORKDIR /workspace/npp-rl
COPY requirements.txt .

# Install remaining npp-rl requirements (most should already be satisfied)
RUN pip install --no-cache-dir -r requirements.txt

# Install npp-rl in development mode (if it has setup.py/pyproject.toml)
RUN if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then pip install --no-cache-dir -e .; fi

# Create necessary directories for both projects
RUN mkdir -p /workspace/nclone/nclone/maps/official \
             /workspace/nclone/nclone/maps/eval \
             /workspace/npp-rl/training_logs \
             /workspace/npp-rl/datasets/processed \
             /workspace/npp-rl/datasets/raw \
             /workspace/npp-rl/models

# PyTorch is already installed from the base image - no need to reinstall

# Verify installations (separate commands for better debugging)
RUN python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
RUN python -c "import numpy; print(f'NumPy {numpy.__version__} installed')"
RUN python -c "import nclone; print('nclone imported successfully')"
RUN python -c "import stable_baselines3; print(f'Stable Baselines3 {stable_baselines3.__version__} installed')"
RUN python -c "import gymnasium; print(f'Gymnasium {gymnasium.__version__} installed')"
RUN python -c "import pygame; print(f'Pygame {pygame.__version__} installed')"
RUN python -c "import cv2; print(f'OpenCV {cv2.__version__} installed')"

# Set up environment configuration
RUN echo "# NPP-RL + nclone Environment Configuration\n\
PYTHONPATH=/workspace/nclone:/workspace/npp-rl\n\
CUDA_LAUNCH_BLOCKING=0\n\
NPP_RL_NUM_ENVS=16\n\
NPP_RL_TOTAL_TIMESTEPS=1000000\n\
NPP_RL_LOG_LEVEL=INFO\n\
NPP_RL_LOG_DIR=/workspace/npp-rl/training_logs\n\
SDL_VIDEODRIVER=dummy" > /workspace/.env

# Set PYTHONPATH for both projects
ENV PYTHONPATH="/workspace/nclone:/workspace/npp-rl"

WORKDIR /workspace/

# Set default command
CMD ["bash"]