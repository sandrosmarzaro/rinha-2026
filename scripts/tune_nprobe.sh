#!/bin/bash
# Run k6 full test for several nprobe values; save results per value.
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p test/tuning

echo "==================== building image (once) ===================="
docker compose build 2>&1 | tail -3

for nprobe in 8 16 32 64; do
    echo "==================== nprobe=$nprobe ===================="
    docker compose down 2>&1 | tail -2

    RINHA_NPROBE=$nprobe docker compose up -d 2>&1 | tail -3

    echo "waiting ready..."
    until curl -fsS -o /dev/null --max-time 1 http://127.0.0.1:9999/ready 2>/dev/null; do sleep 1; done

    rm -f test/test/results.json

    (cd test && docker compose --profile test up --abort-on-container-exit 2>&1 \
        | grep -vE 'Request Failed|level=warning|Downloading|Extracting|Pull|Verifying' \
        | tail -10)

    cp test/test/results.json test/tuning/results-nprobe-$nprobe.json

    final_score=$(python3 -c "import json; d=json.load(open('test/tuning/results-nprobe-$nprobe.json')); print(d['scoring']['final_score'])")
    p99=$(python3 -c "import json; d=json.load(open('test/tuning/results-nprobe-$nprobe.json')); print(d['p99'])")
    failure_rate=$(python3 -c "import json; d=json.load(open('test/tuning/results-nprobe-$nprobe.json')); print(d['scoring']['failure_rate'])")

    echo "  >>> nprobe=$nprobe: score=$final_score  p99=$p99  failure_rate=$failure_rate"
done

echo
echo "==================== SUMMARY ===================="
for f in test/tuning/results-nprobe-*.json; do
    n=$(echo $f | grep -oP 'nprobe-\K\d+')
    python3 -c "
import json
d = json.load(open('$f'))
print(f'nprobe=$n  score={d[\"scoring\"][\"final_score\"]:>8}  p99={d[\"p99\"]:>10}  fail={d[\"scoring\"][\"failure_rate\"]:>7}  p99_score={d[\"scoring\"][\"p99_score\"][\"value\"]:>6}  det_score={d[\"scoring\"][\"detection_score\"][\"value\"]:>6}'
)"
done
