#!/usr/bin/env bash

set -eu

if [ $# -ne 2 ]; then
    echo "Usage: $0 <joern_version> <joern_home>"
    exit 1
fi

joern_version=$1
joern_home=$2

mkdir joern
cd joern
curl -L "https://github.com/joernio/joern/releases/download/${joern_version}/joern-install.sh" -o joern-install.sh
chmod u+x joern-install.sh
./joern-install.sh --version=${joern_version} --install-dir=${joern_home}
cd ..
rm -rf joern

