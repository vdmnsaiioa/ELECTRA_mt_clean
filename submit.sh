#!/bin/bash
# Queue one training run on every GPU partition by submitting two copies of the
# same sbatch script: a 7-day copy for the week-long partitions and a 2-day copy
# for the partitions capped at 2-02:00:00. Whichever copy starts first cancels
# the other (see the sibling block in test.sh).
#
# Usage: ./submit.sh [script]   (default: test.sh)
set -euo pipefail

SBATCH_SCRIPT=${1:-test.sh}
LONG_PARTITIONS=${LONG_PARTITIONS:-sm3090el8,sm3090el9,sm3090el9_768,a100_week}
SHORT_PARTITIONS=${SHORT_PARTITIONS:-a100,a100el9,h200el9}

# Both copies share this tag as job name; it is how they find each other.
PAIR_TAG=electra_MM_$(date +%Y%m%d%H%M%S)_$RANDOM

# Copy the script once so both submissions get identical settings.
SNAPSHOT=$(mktemp)
trap 'rm -f "$SNAPSHOT"' EXIT
cp "$SBATCH_SCRIPT" "$SNAPSHOT"

LONG_ID=$(sbatch --parsable --job-name="$PAIR_TAG" --export=ALL,PAIR_TAG="$PAIR_TAG" \
    --partition="$LONG_PARTITIONS" --time=7-00:00:00 "$SNAPSHOT")
SHORT_ID=$(sbatch --parsable --job-name="$PAIR_TAG" --export=ALL,PAIR_TAG="$PAIR_TAG" \
    --partition="$SHORT_PARTITIONS" --time=2-02:00:00 "$SNAPSHOT")

echo "$PAIR_TAG: 7-day job $LONG_ID ($LONG_PARTITIONS), 2-day job $SHORT_ID ($SHORT_PARTITIONS)"
