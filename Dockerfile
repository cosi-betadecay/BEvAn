# ------------------------------------------------------------
# Base OS + Official Packages
# ------------------------------------------------------------
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -yq \
    gosu vim nano less gzip git git-lfs gawk dpkg-dev make g++ gcc \
    gfortran gdb valgrind binutils libx11-dev libxpm-dev libxft-dev \
    libxext-dev libssl-dev libpcre3-dev libglu1-mesa-dev libglew-dev \
    libftgl-dev libmysqlclient-dev libfftw3-dev libgraphviz-dev \
    libavahi-compat-libdnssd-dev libldap2-dev python3 python3-pip \
    python3-dev python3-tk python3-venv libxml2-dev libkrb5-dev \
    libgsl-dev cmake libxmu-dev curl doxygen libblas-dev liblapack-dev \
    expect dos2unix libncurses5-dev libboost-all-dev libcfitsio-dev \
    libxerces-c-dev libhealpix-cxx-dev bc libhdf5-dev python3-matplotlib \
    libbz2-dev libtbb-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Install Python 3.11 from PPA and set it as default
# ------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-dev python3.11-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 2 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Add COSI user (required for correct install & permissions)
# ------------------------------------------------------------
RUN groupadd -g 1111 cosi && \
    useradd -u 1111 -g 1111 -ms /bin/bash cosi

# ------------------------------------------------------------
# Switch to non-root user and install COSItools
# ------------------------------------------------------------
USER cosi
WORKDIR /home/cosi

#RUN /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/cositools/cosi-setup/main/setup.sh)" && \
#    echo ". /home/cosi/COSItools/source.sh" >> ~/.bashrc

# ------------------------------------------------------------
# Install UV (still as user cosi)
# ------------------------------------------------------------
RUN curl -fsSL https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

ENV PATH="/home/cosi/.local/bin:${PATH}"

# ------------------------------------------------------------
# Copy project into image
# ------------------------------------------------------------
# IMPORTANT: This now puts everything into /home/cosi/code/cosi-initial
COPY --chown=1111:1111 . /home/cosi/code/cosi-initial

WORKDIR /home/cosi/code/cosi-initial

# ------------------------------------------------------------
# Create UV env + install deps
# ------------------------------------------------------------
RUN rm -f .python-version
RUN uv venv && uv sync

# ------------------------------------------------------------
# Install torch (GPU or CPU depending on env)
# ------------------------------------------------------------
RUN chmod +x /home/cosi/code/cosi-initial/install_torch.sh && \
    /bin/bash /home/cosi/code/cosi-initial/install_torch.sh

# ------------------------------------------------------------
# Create external exchange directory
# ------------------------------------------------------------
WORKDIR /home/cosi
RUN mkdir /home/cosi/COSIDockerData

# ------------------------------------------------------------
# Switch back to root to fix UID/GID mapping script
# ------------------------------------------------------------
USER root

RUN cd /usr/local/bin \
    && echo '#!/bin/bash' > entrypoint.sh \
    && echo 'if [ "${USERID}" != "" ]; then usermod -u ${USERID} cosi; fi' >> entrypoint.sh \
    && echo 'if [ "${GROUPID}" != "" ]; then groupmod -g ${GROUPID} cosi; fi' >> entrypoint.sh \
    && echo 'if [ "${USERID}" != "" ] || [ "${GROUPID}" != "" ]; then chown -R cosi:cosi /home/cosi; fi' >> entrypoint.sh \
    && echo 'gosu cosi bash' >> entrypoint.sh \
    && chmod a+rx /usr/local/bin/entrypoint.sh

# Default working directory
WORKDIR /home/cosi/COSIDockerData

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]