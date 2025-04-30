#!/bin/bash

image_id=$(docker images dolly-dolly:latest -q --no-trunc)

if [ $# -ne 1 ]; then
    echo "Usage: $0 <output_file>"
    exit 1
fi
output_file=$1

TAR_TMP_FILE=/tmp/dolly-dolly-$(id -u).tar
echo "[+] Saving $image_id to tar"
docker save $image_id  -o $TAR_TMP_FILE
echo "[+] Building Apptainer container"
apptainer build --force $output_file docker-archive:$TAR_TMP_FILE
echo "[+] Cleanup of docker tar image"
rm -f $TAR_TMP_FILE
echo "[+] All done!"