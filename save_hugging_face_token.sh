#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Error: Please provide exactly one argument - your Hugging Face access token."
  exit 1
fi

# Create the .secrets directory if it doesn't exist
mkdir -p ~/.secrets

# Securely store the token (using gpg for encryption)
echo "$1" | gpg -c --output ~/.secrets/huggingface_token

# Determine the user's shell
if [[ $SHELL == *bash* ]]; then
  config_file=~/.bashrc
elif [[ $SHELL == *zsh* ]]; then
  config_file=~/.zshrc
else
  echo "Unsupported shell. Please use bash or zsh."
  exit 1
fi

# Check if the export line already exists
if ! grep -q "export HF_HOME=/shared/huggingface" "$config_file"; then
  # Append the export line to the configuration file
  echo "export HF_HOME=/shared/huggingface" >> "$config_file"
fi

if ! grep -q "export HF_TOKEN=\$(cat ~/.secrets/huggingface_token | gpg -d)" "$config_file"; then
  # Append the export line to the configuration file
  echo "export HF_TOKEN=\$(cat ~/.secrets/huggingface_token | gpg -d)" >> "$config_file"
fi

echo "Hugging Face token stored securely in ~/.secrets/huggingface_token."
echo "Configuration updated. Source your shell configuration file (e.g., source ~/.bashrc) and open a new terminal."
