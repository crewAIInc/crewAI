FROM mcr.microsoft.com/devcontainers/python:3.12-bullseye

# Install additional OS packages
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    build-essential \
    cmake \
    gdb \
    valgrind \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libffi-dev \
    swig \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for development
RUN pip3 install --no-cache-dir \
    cython \
    cffi \
    pybind11 \
    setuptools \
    wheel \
    build

# Set the working directory
WORKDIR /workspaces/Soln.ai

# Copy the project files
COPY . .

# Set environment variables
ENV PYTHONPATH=/workspaces/Soln.ai
ENV PYTHONUNBUFFERED=1

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
