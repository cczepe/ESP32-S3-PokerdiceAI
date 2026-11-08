#!/usr/bin/env bash
# Pokerdice trainer - creates the Python environment .venv (Python 3.11) and installs everything.
# Run in the repository folder:  ./setup.sh
# Running it again only updates the packages; an existing .venv is kept.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    PY=""
    for c in python3.11 python3; do
        if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))'; then
            PY="$c"; break
        fi
    done
    if [ -z "$PY" ]; then
        echo "Python 3.11 not found (needed for esp-ppq). Install it, e.g. 'sudo apt install python3.11 python3.11-venv'"
        echo "or with pyenv / Homebrew, then run ./setup.sh again."
        exit 1
    fi
    echo "Creating .venv with $PY ..."
    "$PY" -m venv .venv
else
    echo ".venv exists already - updating the packages."
fi

.venv/bin/python -m pip install --upgrade pip
echo "Installing torch (CPU version) ..."
if [ "$(uname)" = "Darwin" ]; then
    .venv/bin/python -m pip install torch==2.15.0
else
    .venv/bin/python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
fi
echo "Installing the trainer packages ..."
.venv/bin/python -m pip install -r trainer/requirements.txt
echo
echo "Done. Start the trainer with ./start.sh (or: .venv/bin/python trainer/assistant/app.py)"
