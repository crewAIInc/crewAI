# Use an official Ubuntu as a parent image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install common utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    software-properties-common

# Install Python
RUN apt-get install -y python3 python3-pip

# Clean up
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace
