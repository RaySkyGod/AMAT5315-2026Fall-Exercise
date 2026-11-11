#!/usr/bin/env bash
# TRG sweeps for the periodic-square peak study (week 10, part 3).
#
#   sweep_trg.sh scan    L = 64, 256, 1024 x K = 0.40..0.50 step 0.0025, chi = 16
#   sweep_trg.sh refine  9 points spaced 0.00125 around each scan peak,
#                        chi = 16 and 32 (needs runs/torus/centers.json)
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:?usage: sweep_trg.sh scan|refine}"
JOBS="${JOBS:-8}"

case "$STAGE" in
  scan)
    python3 - > /tmp/trg_scan_jobs.txt <<'PY'
for L in (64, 256, 1024):
    for i in range(41):
        K = 0.40 + i * 0.0025
        print(f"{L} {K!r} 16 runs/torus/L{L}/chi16/scan/K{K:.4f}")
PY
    ;;
  refine)
    python3 - > /tmp/trg_refine_jobs.txt <<'PY'
import json
centers = json.load(open("runs/torus/centers.json"))
for L, c in sorted(centers.items(), key=lambda kv: int(kv[0])):
    for chi in (16, 32):
        for j in range(-4, 5):
            K = c + j * 0.00125
            print(f"{L} {K!r} {chi} runs/torus/L{L}/chi{chi}/refine/K{K:.5f}")
PY
    ;;
  *)
    echo "usage: sweep_trg.sh scan|refine" >&2
    exit 2
    ;;
esac

JOBSFILE="/tmp/trg_${STAGE}_jobs.txt"
TOTAL=$(wc -l < "$JOBSFILE")
PENDING=0
while read -r L K CHI OUT; do
  [ -f "$OUT/meta.json" ] || { echo "$L $K $CHI $OUT"; PENDING=$((PENDING + 1)); }
done < "$JOBSFILE" > /tmp/trg_${STAGE}_pending.txt

xargs -P "$JOBS" -n 4 sh -c 'thermo --method trg --l "$1" --k "$2" --chi "$3" --out "$4" > /dev/null' _ < /tmp/trg_${STAGE}_pending.txt
echo "sweep_trg.sh $STAGE: $TOTAL runs listed, $PENDING executed"
