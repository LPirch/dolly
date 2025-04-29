#!/usr/bin/env bash

set -eu

JOERN_PARSE="/opt/joern/joern-cli/joern-parse"

if [ $# -ne 4 ]; then
    echo "Usage: $0 <input-path> <output-path> <joern-memory> <joern-cores>"
    exit 1
fi

INPUT_PATH=$1
OUTPUT_PATH=$2
JOERN_MEMORY=$3
JOERN_CORES=$4

$JOERN_PARSE -J-Xmx$JOERN_MEMORY -J-XX:ActiveProcessorCount=$JOERN_CORES "$INPUT_PATH" -o "$OUTPUT_PATH"
