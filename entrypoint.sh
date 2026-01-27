#!/bin/bash
set -e

# Load environment variables from .env if present (for local development only)
# In production, variables come from GitHub Secrets/Variables or Hugging Face Space settings
if [ -f .env ]; then
  echo "Loading environment from .env file..."
  set -o allexport
  source .env
  set +o allexport
fi

# Set default provider if not set
export MODEL_PROVIDER="${MODEL_PROVIDER:-openai}"

echo "Model Provider: $MODEL_PROVIDER"

# Execute the main command
exec "$@"
