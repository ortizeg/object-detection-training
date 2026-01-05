#!/bin/bash
set -e

echo "Setting up development environment..."

# Check if pixi is installed
if ! command -v pixi &> /dev/null; then
    echo "Pixi not found. Installing..."
    curl -fsSL https://pixi.sh/install.sh | bash

    # Add pixi to PATH for current session
    export PATH="$HOME/.pixi/bin:$PATH"

    echo "Pixi installed successfully."
else
    echo "Pixi is already installed."
fi

# Install dependencies using pixi
echo "Installing dependencies..."
pixi install

echo "Development environment setup complete."
echo "Run 'pixi run train' to start training or 'pixi shell' to activate the environment."
