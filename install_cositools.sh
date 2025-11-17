#!/bin/bash
set -e

cd /opt/COSItools

# Clone cosi-setup if not exists
if [ ! -d "cosi-setup" ]; then
    git clone https://github.com/cositools/cosi-setup cosi-setup
fi

cd cosi-setup

# Run setup (non-interactive, resume-safe)
bash setup.sh --ignore-missing-packages --maxthreads=8