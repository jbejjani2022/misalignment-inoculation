#!/bin/bash

# Run evaluation script and save filtered output to eval/test_output.txt
# This replicates the behavior of the previous GitHub Actions autograder

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables from .env file in project root
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment from .env file..."
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
    # Export variables from .env, skipping comments and empty lines
    # export $(grep -v '^#' "$PROJECT_ROOT/.env" | grep -v '^$' | xargs)
else
    echo "Warning: No .env file found at $PROJECT_ROOT/.env"
    echo "Make sure OPENAI_API_KEY is set in your environment."
fi

# Verify API key is available
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY is not set. Please add it to your .env file."
    exit 1
fi

echo "Running test script..."

# Run the test script and save output
cd "$PROJECT_ROOT"
python eval/run_tests.py 2>&1 | tee "$SCRIPT_DIR/test_output.txt"

echo ""
echo "Results saved to: $SCRIPT_DIR/test_output.txt"
