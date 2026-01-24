#!/bin/bash
set -e

# Load environment variables from .env if present
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi

# Set default provider if not set
if [ -z "$MODEL_PROVIDER" ]; then
  MODEL_PROVIDER="openai"
fi

echo "Model Provider: $MODEL_PROVIDER"

# Execute the main command
exec "$@"
