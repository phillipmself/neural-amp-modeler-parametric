#!/usr/bin/env bash
#
# Run one GPU's half of the knob-move arm comparison: two FiLMWaveNet arms, then two
# ConcatLSTM arms. Launch one instance per GPU; the two halves are independent.
#
#   ./tools/run_knob_move_arms.sh 0 &
#   ./tools/run_knob_move_arms.sh 1 &
#
# Overridable:
#   DATA_DIR   where the configs and captures live   (default /workspace/captures/SD1)
#   OUT_DIR    where run directories are written     (default /workspace/runs)
#
# Why these pairings: the arms are not equal cost, so each GPU takes one cheap and one
# expensive run of each architecture rather than all the cheap ones landing together.
#   FiLM  - the quasi-static anchor adds two forwards per step (its moving render and its
#           frozen reference), so c_quasi and d_both cost noticeably more than a_baseline
#           and b_landed. Landed-move is ~free; it only changes the control tensor's shape.
#   LSTM  - the silence anchors add one forward of 8192 sequential steps per training step,
#           roughly +25%, so b_silence and d_both cost more than a_baseline and c_landed.
# FiLM runs first on both GPUs so those results land before the LSTM work starts.

set -uo pipefail

GPU="${1:-}"
if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 <gpu-index>" >&2
    exit 2
fi

DATA_DIR="${DATA_DIR:-/workspace/captures/SD1}"
OUT_DIR="${OUT_DIR:-/workspace/runs}"

case "$GPU" in
    0) ARMS=(film:a_baseline film:d_both lstm:a_baseline lstm:d_both) ;;
    1) ARMS=(film:b_landed   film:c_quasi lstm:c_landed  lstm:b_silence) ;;
    *) echo "No arm list defined for GPU $GPU (expected 0 or 1)." >&2; exit 2 ;;
esac

config_paths() {
    # arch arm -> data, model, learning
    local arch="$1" arm="$2"
    case "$arch" in
        film) echo "$DATA_DIR/data.json $DATA_DIR/model_film_$arm.json $DATA_DIR/learning_film.json" ;;
        lstm) echo "$DATA_DIR/data_lstm.json $DATA_DIR/model_lstm_$arm.json $DATA_DIR/learning_lstm.json" ;;
        *)    return 1 ;;
    esac
}

# Preflight: a missing config three hours in is the worst way to find out.
missing=0
for entry in "${ARMS[@]}"; do
    read -r data model learning <<<"$(config_paths "${entry%%:*}" "${entry#*:}")"
    for f in "$data" "$model" "$learning"; do
        [[ -f "$f" ]] || { echo "missing config: $f" >&2; missing=1; }
    done
done
[[ "$missing" -eq 0 ]] || exit 1

mkdir -p "$OUT_DIR"
echo "GPU $GPU  |  arms: ${ARMS[*]}"
echo "data $DATA_DIR  ->  out $OUT_DIR"

declare -a RESULTS=()
overall=0

for entry in "${ARMS[@]}"; do
    arch="${entry%%:*}"
    arm="${entry#*:}"
    name="${arch}_${arm}"
    read -r data model learning <<<"$(config_paths "$arch" "$arm")"

    run_out="$OUT_DIR/$name"
    log="$OUT_DIR/${name}.log"
    mkdir -p "$run_out"

    echo
    echo "=== $name  ($(date '+%H:%M:%S')) ==="
    start=$SECONDS
    CUDA_VISIBLE_DEVICES="$GPU" nam-full-parametric \
        "$data" "$model" "$learning" "$run_out" --no-show 2>&1 | tee "$log"
    status="${PIPESTATUS[0]}"
    elapsed=$(( SECONDS - start ))

    if [[ "$status" -eq 0 ]]; then
        RESULTS+=("ok      $name  $((elapsed / 60))m")
    else
        RESULTS+=("FAILED  $name  $((elapsed / 60))m  (exit $status, see $log)")
        overall=1
    fi
    # Keep going on failure: one bad arm should not idle the GPU for the rest.
done

echo
echo "=== GPU $GPU summary ==="
printf '%s\n' "${RESULTS[@]}"

# The .nam each run exports is what tools/score_knob_move.py reads.
echo
echo "score with:"
echo "  python tools/score_knob_move.py $OUT_DIR/film_*/*/model_parametric.nam \\"
echo "      --input $DATA_DIR/input_train.wav"

exit "$overall"
