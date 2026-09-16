#!/usr/bin/env bash
set -euo pipefail
ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
command -v python3 >/dev/null 2>&1 || { printf "%s\n" "Python 3.11+ is required; install it yourself, then retry." >&2; exit 1; }
exec python3 -S -B "$ROOT/scripts/bootstrap.py" "$@"
