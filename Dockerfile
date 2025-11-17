# Base OS
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    bash ca-certificates curl wget git build-essential cmake \
    python3 python3-pip python3-dev gfortran xorg-dev libx11-dev \
    libxpm-dev libxft-dev libxext-dev libssl-dev libffi-dev \
    libglu1-mesa-dev libgl1-mesa-glx libgl1-mesa-dev \
    software-properties-common git-lfs gawk gdb valgrind \
    libpcre3-dev libglew-dev libftgl-dev libmysqlclient-dev \
    libfftw3-dev libgraphviz-dev libavahi-compat-libdnssd-dev \
    libldap2-dev python3-tk python3-venv python3-matplotlib \
    libxml2-dev libkrb5-dev libgsl-dev doxygen libblas-dev \
    liblapack-dev expect dos2unix libncurses5-dev libboost-all-dev \
    libcfitsio-dev libxerces-c-dev libhealpix-cxx-dev bc libhdf5-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create directory for COSItools
RUN mkdir -p /opt/COSItools

# Copy install script into container
COPY install_cosi.sh /opt/COSItools/install_cosi.sh
RUN chmod +x /opt/COSItools/install_cosi.sh

# Install COSItools
RUN /opt/COSItools/install_cosi.sh

SHELL ["/bin/bash", "-c"]
CMD ["/bin/bash"]