# ... (Your header remains the same)
FROM nvcr.io/nvidia/deepstream:8.0-triton-multiarch

# 1. Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. Install system dependencies
# Added libgirepository1.0-dev which is often required and uncommented
RUN apt-get update && apt-get install -y \
    python3-pip \
    libpython3-dev \
    python3-dev \
    python3-gi \
    python3-gst-1.0 \
    python-gi-dev \
    libcairo2-dev \
    libgirepository1.0-dev \
    libgstreamer1.0-dev \
    pkg-config \
    git \
    wget \
    meson \
    cmake \
    g++ \
    build-essential \
    libglib2.0-dev \
    libglib2.0-dev-bin \
    libtool \
    m4 \
    autoconf \
    automake \
    && rm -rf /var/lib/apt/lists/*

# 3. Build DeepStream Python Bindings
RUN git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git && \
    cd deepstream_python_apps && \
    git checkout master && \
    git submodule update --init && \
    # FIX 1: Install build tools 'wheel' and 'setuptools' explicitly
    cd bindings && \
    mkdir build && cd build && \
    export DS_PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')") && \
    cmake .. -DDS_PYTHON_VERSION=${DS_PYTHON_VER} && \
    make -j$(nproc) && \
    # FIX 2: Instead of looking for a missing .whl file, install the bindings 
    # directly using pip from the bindings source directory.
    cd .. && \
    pip3 install .

# 4. Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 5. Copy Application Code
COPY src/app.py .

# 6. Entrypoint
CMD ["python3", "app.py"]
