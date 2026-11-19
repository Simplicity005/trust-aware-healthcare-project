#!/usr/bin/env bash
set -e
export PATH="$PATH:/opt/homebrew/sbin:/opt/homebrew/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin"
exec python3 "$(dirname "${BASH_SOURCE[0]}")/setup.py" "$@"
