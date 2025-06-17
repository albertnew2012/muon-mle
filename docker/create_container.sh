#!/usr/bin/env bash

LUCID_DATA_DIR=/perception_data/
# WORK_DIR=/integration-dds
USER_NAME=$(id -un)

# Get the full path to the script
SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
DIR_NAME="$(basename "$(dirname "$SCRIPT_DIR")")"
IMG_NAME="$(echo "$DIR_NAME" | tr '[:upper:]' '[:lower:]')" # Convert to lowercase
REPO_DIR="$(dirname "$SCRIPT_DIR")"

docker run \
    -it \
    --env DISPLAY=$DISPLAY \
    --gpus all \
    --network=host \
    --shm-size=128G \
    --privileged=true \
    -v $REPO_DIR:/$DIR_NAME \
    -v /home/$USER:/home/$USER_NAME \
    -v /etc/localtime:/etc/localtime:ro \
    -v /etc/timezone:/etc/timezone:ro \
    -w /$DIR_NAME \
    $IMG_NAME:latest bash
