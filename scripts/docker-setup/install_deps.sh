#!/usr/bin/env bash

set -eu

if [ $# -eq 0 ]; then
    echo "Usage: $0 <package1> <package2> ..."
    exit 1
fi

apt update
apt install -y --no-install-recommends $@
