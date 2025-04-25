#!/usr/bin/env bash

set -eu

if [ $# -ne 1 ]; then
    echo "Usage: $0 <python_version>"
    exit 1
fi

python_version=$1

wget https://www.python.org/ftp/python/${python_version}/Python-${python_version}.tar.xz
tar -xf Python-${python_version}.tar.xz
cd Python-${python_version}
./configure --enable-optimizations
make -j 8
make install
ln -s /usr/local/bin/python3 /usr/bin/python
ln -s /usr/local/bin/pip3 /usr/bin/pip
cd ..
rm -rf Python-${python_version}
rm Python-${python_version}.tar.xz
