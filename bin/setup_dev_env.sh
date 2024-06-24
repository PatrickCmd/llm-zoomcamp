#!/bin/bash

set -ex

# Move to the specified folder
cd "$CODESPACE_VSCODE_FOLDER"

# Download the Anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

# Install Anaconda
bash Anaconda3-2024.02-1-Linux-x86_64.sh

# Initialize Conda for the current shell session
source ~/anaconda3/bin/activate

# Install necessary Python packages
pip install tqdm openai elasticsearch groq

# Confirm successful setup
echo "Development environment setup complete!"
