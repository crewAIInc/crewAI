#!/bin/bash

# Exit on error
set -e

echo "Starting post-creation setup..."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found, skipping Python dependencies"
fi

# Install Node.js dependencies
echo "Installing Node.js dependencies..."
if [ -d "frontend/nextjs" ]; then
    cd frontend/nextjs
    
    # Install npm packages
    echo "Installing npm packages..."
    npm install

    # Create necessary directories if they don't exist
    mkdir -p docs/components

    cd ../..
else
    echo "No frontend/nextjs directory found, skipping Node.js dependencies"
fi

# Install global npm packages
echo "Installing global npm packages..."
npm install -g prettier
npm install -g jest

# Set up Git hooks (if needed)
if [ -d ".git" ]; then
    echo "Setting up Git hooks..."
    # Add any git hook setup here
fi

echo "Post-creation setup completed successfully!" 