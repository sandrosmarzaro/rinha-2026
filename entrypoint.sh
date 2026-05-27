#!/bin/sh
set -e

mkdir -p "$(dirname "$RINHA_SOCKET")"
rm -f "$RINHA_SOCKET"

# Grant haproxy (different user inside its container) access to the socket.
umask 000

exec granian \
    --interface asgi \
    --uds "$RINHA_SOCKET" \
    --workers 1 \
    --runtime-mode st \
    --log-level warning \
    fraud_api.app:app
