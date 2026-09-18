#!/usr/bin/env bash
# End-to-end check against a running service.
#   ./scripts/smoke_test.sh                          # local
#   ./scripts/smoke_test.sh https://your.onrender.com # deployed
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"
CASES="$(dirname "$0")/../data/public_sample_cases.json"

echo "== GET $BASE/health"
curl -fsS --max-time 60 "$BASE/health"; echo

echo "== POST $BASE/optimize-energy  (SAMPLE-01)"
python3 -c "
import json,sys
pack=json.load(open('$CASES'))
json.dump(pack['cases'][0]['input'],sys.stdout)
" > /tmp/gridwise_case.json

START=$(python3 -c 'import time;print(time.time())')
curl -fsS --max-time 30 -X POST "$BASE/optimize-energy" \
  -H 'Content-Type: application/json' \
  --data @/tmp/gridwise_case.json > /tmp/gridwise_out.json
python3 -c "import time;print(f'   latency: {time.time()-$START:.2f}s')"

python3 - <<'PY'
import json
body = json.load(open('/tmp/gridwise_out.json'))
print(f"   scenario_id : {body['scenario_id']}")
print(f"   plan hours  : {len(body['hourly_plan'])}")
print(f"   total cost  : {body['total_cost_bdt']:,.2f} BDT")
print(f"   peak grid   : {body['peak_grid_kwh']:,.2f} kWh")
for entry in body["directive_interpretation"]:
    print(f"   note {entry['note_index']}: {entry['directive_type']} -> {entry['structured_adjustment']}")
PY

echo "== malformed body should return 400"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/optimize-energy" \
  -H 'Content-Type: application/json' --data '{not json')
echo "   got HTTP $code"; [ "$code" = "400" ] || { echo "   EXPECTED 400"; exit 1; }

echo "== schema-invalid body should return 422"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/optimize-energy" \
  -H 'Content-Type: application/json' --data '{"scenario_id":"X"}')
echo "   got HTTP $code"; [ "$code" = "422" ] || { echo "   EXPECTED 422"; exit 1; }

echo
echo "All smoke checks passed against $BASE"
