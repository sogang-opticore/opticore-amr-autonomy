#!/bin/sh
# Run at repository root with the M3 Python environment active.
set -eu
if [ "$#" -ne 3 ]; then
    echo "usage: $0 M3_BUNDLE TRAINING_RUNS NEW_SCORE_DIRECTORY" >&2
    exit 2
fi
m3_bundle=$1
runs=$2
scores=$3
if [ -e "$scores" ] || [ -e "$scores-summary.json" ]; then
    echo "Choose a new score directory and summary path." >&2
    exit 2
fi
export PYTHONPATH="$PWD/packages/amr_lite"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for suite in primary secondary; do
    python scripts/m4/evaluate.py --suite "$suite" --variant m3 --bundle "$m3_bundle" \
        --out "$scores/$suite-m3-7.json"
    for variant in full no_history no_randomization; do
        for seed in 7 19 31; do
            bundle="$runs/$variant/$seed/bundle"
            python scripts/m4/evaluate.py --suite "$suite" --variant "$variant" --bundle "$bundle" \
                --out "$scores/$suite-$variant-$seed.json"
            if [ "$variant" = full ]; then
                python scripts/m4/evaluate.py --suite "$suite" --variant full --shield-off --bundle "$bundle" \
                    --out "$scores/$suite-full-shield-off-$seed.json"
            fi
        done
    done
done
python scripts/m4/summarize.py "$scores"/*.json --out "$scores-summary.json"
