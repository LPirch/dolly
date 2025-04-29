#!/usr/bin/env bash

set -eu

if [ $# -ne 5 ]; then
    echo "Usage: $0 <memory> <cores> <script_file> <in_file> <out_file>"
    exit 1
fi

memory=$1; shift
cores=$1; shift
script_file=$1; shift
in_file=$1; shift
out_file=$1; shift

echo -e "main(\"$in_file\", \"$out_file\")" | joern -J-Xmx$memory -J-XX:ActiveProcessorCount=$cores --import $script_file
