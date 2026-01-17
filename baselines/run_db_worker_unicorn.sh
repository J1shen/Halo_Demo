#!/bin/sh
# POSIX-friendly strict mode (pipefail not in POSIX sh)
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PORT_DEFAULT=9104
WORKERS_DEFAULT=1
LOGFILE_DEFAULT=./baselines/logs/db_worker.log

PORT="${PORT:-$PORT_DEFAULT}"
WORKERS="${WORKERS:-$WORKERS_DEFAULT}"
BIND="0.0.0.0:${PORT}"
FOREGROUND="${FOREGROUND:-1}"
LOGFILE="${LOGFILE:-$LOGFILE_DEFAULT}"
TIMEOUT="${TIMEOUT:-1200}"
POOL_SIZE="${POOL_SIZE:-32}"
# write access log to file by default (disable with ACCESS_LOG="")
ACCESS_LOG="${ACCESS_LOG:-$LOGFILE.access}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Default DB connection (matches halo defaults); can override by exporting before running.
: "${HALO_PG_HOST:=/var/run/postgresql}"
: "${HALO_PG_PORT:=4032}"
: "${HALO_PG_DBNAME:=imdb}"
: "${HALO_PG_USER:=${USER:-postgres}}"
: "${HALO_PG_POOL_SIZE:=$POOL_SIZE}"
# HALO_PG_PASSWORD left unset by default.

# Ensure DB env vars are visible to gunicorn workers.
export HALO_PG_HOST HALO_PG_PORT HALO_PG_DBNAME HALO_PG_USER HALO_PG_POOL_SIZE HALO_PG_PASSWORD

if ! command -v gunicorn >/dev/null 2>&1; then
  echo "gunicorn not found in PATH. Install with: pip install gunicorn" >&2
  exit 127
fi

cd "$REPO_ROOT"
GUNICORN_OPTS="
  -k uvicorn.workers.UvicornWorker
  --bind $BIND
  --workers $WORKERS
  --timeout $TIMEOUT
  --error-logfile $LOGFILE
  --access-logfile ${ACCESS_LOG:--}
"

echo "Starting DB worker: bind=$BIND workers=$WORKERS foreground=$FOREGROUND log=$LOGFILE"

if [ "$FOREGROUND" = "1" ]; then
  # Run in foreground to show logs.
  # shellcheck disable=SC2086
  gunicorn baselines.db_worker.unicorn_worker:app $GUNICORN_OPTS
else
  # shellcheck disable=SC2086
  gunicorn baselines.db_worker.unicorn_worker:app $GUNICORN_OPTS --daemon
  echo "DB worker started in background on $BIND (error log: $LOGFILE)"
fi
