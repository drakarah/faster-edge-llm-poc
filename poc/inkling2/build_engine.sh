#!/bin/bash
# Builds the Inkling 2 Mini engine with the settings of poc/config/inkling2-mini.sh and the INT8 drafter file.
# usage: [DRAFTER=dflash2|dspark] [DFLASH2_DIR=<download dir of incoai/Inkling-2-Mini-DFlash2>] build_engine.sh CHECKPOINT_DIR ENGINE_DIR
#        (default DRAFTER=dflash2: the checkpoint is downloaded next to ENGINE_DIR when DFLASH2_DIR is not set; dspark = the drafter that ships in the Inkling checkpoint)   (run inside the builder image; CHECKPOINT_DIR = converted Inkling 2 checkpoint, also holding the original hf files for the drafter)
set -euo pipefail
CK=${1:?checkpoint dir}; OUT=${2:?engine dir}
HF=${HF_DIR:-$CK}                                    # directory with mtp.safetensors and the target safetensors (embedding norm, unembedding)
POC=$(cd "$(dirname "$0")/.." && pwd)
set +e; die() { echo "$*"; }; . "$POC/config/inkling2-mini.sh"; set -e   # the config's die line only warns here
for kv in $BUILD_ENV; do export "$kv"; done
python3 - "$CK" "$OUT" "$MAXIN" "$CAP" <<'PY'
import sys
ck, out, maxin, cap = sys.argv[1:5]
sys.argv = ['build', '--model-dir', ck, '--engine-dir', out, '--max-input-len', maxin, '--max-kv-cache-capacity', cap, '--max-batch-size', '1']
from experimental.builder.cli import main
main()
PY
# Drafter: the community DFlash2 drafter is the default (METRICS 209: hybrid K=3 40.67 tok/s against 37.64 with the checkpoint's DSpark drafter on the same engine); DRAFTER=dspark builds the DSpark file instead
if [ "${DRAFTER:-$DRAFTER_DEFAULT}" = dflash2 ]; then
  DFLASH2_DIR=${DFLASH2_DIR:-$(dirname "$OUT")/Inkling-2-Mini-DFlash2}
  if [ ! -f "$DFLASH2_DIR/model.safetensors" ]; then
    echo "downloading $DRAFTER_REPO to $DFLASH2_DIR"
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$DRAFTER_REPO', local_dir='$DFLASH2_DIR', allow_patterns=['config.json', 'model.safetensors'])"
  fi
  python3 "$POC/tools/inkling2_dflash2_int8.py" "$DFLASH2_DIR/model.safetensors" "$HF" "$OUT/drafter_int8.safetensors"
else
  python3 "$POC/tools/inkling2_drafter_int8.py" "$HF" "$OUT/drafter_int8.safetensors"
fi
echo "engine and drafter written to $OUT"
