#!/usr/bin/env bash
# Starts the Pokerdice trainer (opens the browser). Options: ./start.sh --help
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    echo "No .venv yet - run ./setup.sh first."
    exit 1
fi
exec .venv/bin/python trainer/assistant/app.py "$@"
