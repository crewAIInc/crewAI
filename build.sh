#!/bin/bash

# Remove all files in the 'dist' directory
rm -f /mnt/d/Infra/crewAI/dist/*

# Remove all files in the '../ctim/crewAI/dist' directory
rm -f /mnt/d/agentia/agentia_ctim/ctim/crewAI/dist/*

# Build the project using Poetry
poetry build

# Copy all files from the 'dist' directory to '../ctim/crewAI/dist'
cp -r /mnt/d/Infra/crewAI/dist/* /mnt/d/agentia/agentia_ctim/ctim/crewAI/dist/

cd /mnt/d/agentia/agentia_ctim/ctim/

make build && make up

cd /mnt/d/agentia/agentia_ctim/crewAI
