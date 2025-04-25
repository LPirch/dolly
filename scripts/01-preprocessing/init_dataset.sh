#!/usr/bin/env bash

set -eu

RAW_DIR="/app/dolly/data/big-clone-bench/raw"
HF_DIR="/app/dolly/data/big-clone-bench/hf"
BASE_MODEL="microsoft/unixcoder-base-nine"

dolly init-dataset $RAW_DIR $HF_DIR $BASE_MODEL
