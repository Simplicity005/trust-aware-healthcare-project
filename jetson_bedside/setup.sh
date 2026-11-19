#!/usr/bin/env bash
set -e
exec python3 "$(dirname "${BASH_SOURCE[0]}")/setup.py" "$@"
