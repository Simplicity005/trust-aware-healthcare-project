#!/usr/bin/env bash
export PATH="$PATH:/opt/homebrew/sbin:/opt/homebrew/bin:/usr/local/sbin:/usr/local/bin"
exec python3 "$(dirname "${BASH_SOURCE[0]}")/nurse_station/launch.py" --demo "$@"
