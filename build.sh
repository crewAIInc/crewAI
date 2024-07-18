#!/bin/bash

# Remove all files in the 'dist' directory
rm -f dist/*

# Remove all files in the '../ctim/crewAI/dist' directory
rm -f ../ctim/crewAI/dist/*

# Build the project using Poetry
poetry build

# Copy all files from the 'dist' directory to '../ctim/crewAI/dist'
cp -r dist/* ../ctim/crewAI/dist/

cd ../ctim/

make build && make up

cd ../crewAI
