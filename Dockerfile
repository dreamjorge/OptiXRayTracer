# Use a compatible NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive  
# Install essential development tools and libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create the target directory for OptiX
RUN mkdir -p /usr/local/optix

# Copy and install OptiX
COPY third_party/NVIDIA-OptiX-SDK-8.1.0-linux64-x86_64-35015278.sh /optix.sh
RUN chmod +x /optix.sh && ./optix.sh --skip-license --prefix=/usr/local/optix > /optix_install.log 2>&1 || (cat /optix_install.log && exit 1) && \
    rm /optix.sh

# Set up environment variables for CUDA and OptiX
ENV CUDA_PATH=/usr/local/cuda
ENV OPTIX_ROOT=/usr/local/optix

# Set up user to run as non-root in the container
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create a non-root user to use the container
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    chown -R $USERNAME:$USERNAME /usr/local/optix

# Set the working directory to /workspace and switch to the user
WORKDIR /workspace
USER $USERNAME