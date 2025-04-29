#!/usr/bin/env bash

set -eu

if [ $# -ne 1 ]; then
    echo "Usage: $0 <volumes_root>"
    exit 1
fi

volumes_root=$(realpath $1)

for volume in big-clone-bench dolly-models; do
    if ! docker volume inspect $volume >/dev/null 2>&1; then
        docker volume create $volume --opt type=none --opt device=$volumes_root/$volume --opt o=bind
    else
        echo "Volume $volume already exists"
    fi
done