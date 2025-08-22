# Base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-pyqt5 \
    python3-pyqt5.qtwebkit \
    python3-pyqt5.qtsvg \
    build-essential \
    libgl1-mesa-glx \
    libxrender1 \
    libsm6 \
    libxext6 \
    libxcomposite-dev \
    libxcursor-dev \
    libxdamage-dev \
    libxtst-dev \
    libnss3 \
    libasound2 \
    libxrandr-dev \
    libxi-dev \
    libdbus-1-dev \
    libfontconfig1 \
    libfreetype6 \
    libx11-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*


# Uninstall system numpy if installed (optional, but recommended for consistency)
RUN apt-get remove -y python3-numpy || true
RUN pip install --upgrade pip \
    && pip install pandas scikit-learn scikit-image \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

RUN apt-get update && apt-get install -y\
    python3-matplotlib\
    python3-opencv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure correct numpy version
RUN pip install --force-reinstall numpy==1.22.4

# #############################################################################################
# # Install additional dependencies for X11 forwarding

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    xauth \
    sudo \
    screen \
    lsof \
    rsync \
    htop \
    iputils-ping \
    x11-apps \
    parallel \
    git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Dynamically create a user matching the host's username, UID, and GID
ARG USER_NAME=defaultuser
ARG USER_ID=1000
ARG GROUP_ID=1000

# Create the user, add to sudo group, and configure sudo privileges
RUN groupadd -g ${GROUP_ID} ${USER_NAME} && \
    useradd -l -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash ${USER_NAME} && \
    usermod -aG sudo ${USER_NAME} && \
    mkdir -p /etc/sudoers.d && \
    echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME} && \
    chmod 0440 /etc/sudoers.d/${USER_NAME} && \
    chmod 755 /etc/sudoers.d

# Fix permissions for apt directories
RUN chmod -R a+rwx /var/lib/apt/lists /var/cache/apt /var/log

# Verify sudo works before switching users
RUN echo "Testing sudo access" && \
    sudo -u ${USER_NAME} bash -c "echo Hello from user: $(whoami)"

# Switch to the dynamically created user
USER ${USER_NAME}
WORKDIR /home/${USER_NAME}
