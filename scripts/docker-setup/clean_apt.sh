#!/usr/bin/env bash

set -eu

if [ $# -eq 0 ]; then
    echo "Usage: $0 <package1> <package2> ..."
    exit 1
fi

must_haves="git tar build-essential"
to_remove="$@"
for must_have in $must_haves; do
    to_remove=$(echo "$to_remove" | sed "s/$must_have//g")
done

apt remove -y --purge $to_remove
apt autoremove -y
apt clean
rm -rf /var/lib/apt/lists/*
