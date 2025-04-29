#!/usr/bin/env bash

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <java_version>"
    exit 1
fi

if [ ! curl ]; then
    echo "curl is not installed"
    exit 1
fi

java_version=$1

curl -s "https://get.sdkman.io" | bash
source "/root/.sdkman/bin/sdkman-init.sh"
sdk install java ${java_version}
sdk install sbt
