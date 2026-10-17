# Pipeline stages, sourced by run.sh after config/common.sh and config/<model>.sh.
ASSIST=$MODELS_DIR/${ASSIST_REPO##*/}
BASE=$MODELS_DIR/${BASE_REPO##*/}
CKPT=$CKPT_DIR/ckpt
FAST=$ENGINE_DIR/fast
STOCK_CACHE=$ENGINE_DIR/stock-cache
WORK=$DATA/work/$MODEL

# Fail early, with the fix, when RAM + swap cannot hold the conversion (earlyoom kills the job otherwise, hours in).
need_memory_gb() {
  local have; have=$(awk '/^(MemTotal|SwapTotal):/ {s+=$2} END {printf "%d", s/1048576}' /proc/meminfo)
  [ "$have" -ge "$1" ] || die "this step needs about $1 GB of RAM + swap, found $have GB. Add swap on the host:
  sudo fallocate -l $(( $1 - have + 4 ))G /swapfile.poc && sudo chmod 600 /swapfile.poc && sudo mkswap /swapfile.poc && sudo swapon /swapfile.poc"
}

hf_download() {  # hf_download REPO DIR  (resumable; Gemma needs HF_TOKEN and an accepted licence)
  local repo=$1 dir=$2
  [ -f "$dir/.complete" ] && return 0
  python - "$repo" "$dir" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
repo, d = sys.argv[1:3]
snapshot_download(repo_id=repo, local_dir=d, token=os.environ.get("HF_TOKEN") or None,
                  ignore_patterns=["*.gguf", "*.msgpack", "*.h5", "original/*"])
PY
  touch "$dir/.complete"
}

prepare_text() {  # generic calibration text for the Gemma GPTQ steps: Wikipedia (nl, en, fr) + source code
  local T=$DATA/text; mkdir -p $T
  [ -f $T/.complete ] && return 0
  python - "$T" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download
t = sys.argv[1]
for lang, n in (("nl", 6), ("en", 41), ("fr", 17)):
    dst = f"{t}/20231101.{lang}-00000-of-{n:05d}.parquet"
    if not os.path.exists(dst):
        p = hf_hub_download("wikimedia/wikipedia", f"20231101.{lang}/train-00000-of-{n:05d}.parquet", repo_type="dataset")
        shutil.copy(p, dst)
PY
  find /usr/local/lib/python3*/dist-packages/transformers /opt/fast -name '*.py' -size +2k 2>/dev/null | sort | head -4000 \
    | xargs cat | head -c 60000000 > $T/code_json.txt
  touch $T/.complete
}

quantise_gemma() {
  need_memory_gb 40
  prepare_text
  local T=$DATA/text PY=$DATA/venv-llmc/bin/python TOOLS=$POC/tools
  [ -x $PY ] || { python -m venv --system-site-packages $DATA/venv-llmc && $DATA/venv-llmc/bin/pip install -q "llmcompressor==0.14.*" pyarrow; }
  mkdir -p $WORK
  # Google's QAT checkpoint is group 32; the Edge-LLM INT4 kernels take group 128. GPTQ re-quantises the exact QAT values
  # on activations (layer by layer), then the output head gets its own GPTQ pass against the quantised body.
  [ -d $WORK/text-qat ]   || { step strip multimodal towers;  python $TOOLS/strip_multimodal.py $BASE $WORK/text-qat; }
  [ -d $WORK/dense ]      || { step dequantise QAT to bf16;   python $TOOLS/ct_dequantize.py $WORK/text-qat $WORK/dense; }
  [ -d $WORK/ct-g128 ]    || { step "GPTQ g128 (slow)";       $PY $TOOLS/gptq_g128.py $WORK/dense $WORK/ct-g128 --text-dir $T --samples $GPTQ_SAMPLES --seq-len 1024; }
  rm -rf $WORK/dense $WORK/text-qat
  [ -d $WORK/gptq-g128 ]  || { step repack for Edge-LLM;      python $TOOLS/ct_to_gptq.py $WORK/ct-g128 $WORK/gptq-g128 --group-size 128; }
  [ -d $WORK/dense-g128 ] || { step dequantise g128;          python $TOOLS/ct_dequantize.py $WORK/ct-g128 $WORK/dense-g128; }
  [ -d $WORK/gptq-head ]  || { step "GPTQ output head";       $PY $TOOLS/gptq_head.py $WORK/dense-g128 $WORK/gptq-g128 $WORK/gptq-head --text-dir $T --samples 128 --seq-len 1024; }
  rm -rf $CKPT; mv $WORK/gptq-head $CKPT
  rm -rf $WORK
}

quantise_qwen() {
  need_memory_gb 40
  step "INT4-AWQ with Edge-LLM's own quantiser"
  use_stack fast
  local DEV=cuda SAMPLES=256
  [ "$PROFILE" = dgpu ] && { DEV=cpu; SAMPLES=16; }          # the 4B model does not fit calibration on an 8 GB card
  rm -rf $CKPT
  EDGELLM_QUANT_MAX_LENGTH=256 python -m tensorrt_edgellm.scripts.quantize llm --model_dir $BASE --output_dir $CKPT \
    --quantization int4_awq --lm_head_quantization int4_awq --device $DEV --num_samples $SAMPLES
}

builder() {
  step "model $MODEL, profile $PROFILE"
  [ -f $CKPT_DIR/.done ] || {
    step "download"
    hf_download $BASE_REPO $BASE
    [ -n "$ASSIST_REPO" ] && hf_download $ASSIST_REPO $ASSIST
    quantise_$KIND
    touch $CKPT_DIR/.done
  }
  [ -f $FAST/.done ] || build_fast
  [ -f $STOCK_CACHE/.done ] || build_stock
  step "builder finished"
}

build_fast() {
  step "optimised stack: export"
  use_stack fast
  need_memory_gb 28
  rm -rf $FAST; mkdir -p $FAST
  local DRAFT=""; [ -n "$ASSIST_REPO" ] && DRAFT="--mtp-draft-dir $ASSIST"
  env $EXPORT_ENV python -m tensorrt_edgellm.scripts.export $CKPT $FAST/onnx $EXPORT_ARGS $DRAFT 2>&1 | tee $LOG_DIR/$MODEL-$PROFILE-export.log | tail -n 5
  step "optimised stack: engines"
  export EDGELLM_XQA_MULTI_BLOCK=auto                       # split-KV decode attention, sized from the GPU's SM count
  local COMMON="--maxInputLen $MAXIN --maxKVCacheCapacity $CAP --maxBatchSize 1 --maxKVPoolPages $POOL"
  local OUT=$FAST/engine; [ "$INT8_EMBEDDINGS" = 1 ] && OUT=$FAST/engine_fp
  $LLM_BUILD --onnxDir $FAST/onnx/llm --engineDir $OUT $COMMON --maxVerifyTreeSize $TREE --specBase > $LOG_DIR/$MODEL-$PROFILE-build-base.log 2>&1
  $LLM_BUILD --onnxDir $FAST/onnx/mtp_draft --engineDir $OUT $COMMON --maxDraftTreeSize $TREE --specDraft > $LOG_DIR/$MODEL-$PROFILE-build-draft.log 2>&1
  [ "$INT8_EMBEDDINGS" = 1 ] && { python $POC/tools/quantize_ple.py $FAST/engine_fp $FAST/engine --token-embedding; rm -rf $FAST/engine_fp; }
  rm -rf $FAST/onnx
  touch $FAST/.done
}

# Wait until the server on $1 answers /health, or its process $2 dies, or $3 seconds pass.
wait_ready() {
  local port=$1 pid=$2 limit=$3 t=0
  until curl -sf localhost:$port/health >/dev/null; do
    kill -0 $pid 2>/dev/null || return 1
    sleep 3; t=$((t+3)); [ $t -lt $limit ] || return 1
  done
}

start_stock() {  # the stock server builds (first time) or loads (afterwards) its own engine bundle from the checkpoint
  use_stack stock
  local spec=${STOCK_SPEC//@ASSIST@/$ASSIST}
  (cd $STOCK_ROOT && exec python -m experimental.server $CKPT --cache-dir $STOCK_CACHE --host 0.0.0.0 --port $1 \
     --served-model-name local $STOCK_ARGS --speculative-config "$spec") > $LOG_DIR/$MODEL-$PROFILE-stock-server.log 2>&1 &
  SERVER_PID=$!
}

start_fast() {
  use_stack fast
  (cd $STACK_ROOT && exec env $SERVE_ENV python -m experimental.server $FAST/engine --host 0.0.0.0 --port $1 \
     --served-model-name local --enable-auto-tool-choice $SERVE_ARGS) > $LOG_DIR/$MODEL-$PROFILE-fast-server.log 2>&1 &
  SERVER_PID=$!
}

stop_server() { kill $SERVER_PID 2>/dev/null || true; wait $SERVER_PID 2>/dev/null || true; sleep 5; }

build_stock() {
  step "stock stack: its server builds its own engine on first start (can take a long time)"
  start_stock $BENCH_PORT
  if wait_ready $BENCH_PORT $SERVER_PID 14400; then
    touch $STOCK_CACHE/.done
  else
    echo "stock stack failed to start; last log lines:"; tail -n 20 $LOG_DIR/$MODEL-$PROFILE-stock-server.log
    echo "STOCK_FAILED" > $STOCK_CACHE/.failed; touch $STOCK_CACHE/.done
  fi
  stop_server
}

bench() {
  [ -f $FAST/.done ] || die "no optimised engine yet: run the builder first"
  report_clocks
  rm -f $RESULT_DIR/$MODEL-$PROFILE-*-round*.json
  local r
  for r in $(seq 1 $BENCH_ROUNDS); do
    if [ ! -f $STOCK_CACHE/.failed ]; then
      step "round $r: stock"
      start_stock $BENCH_PORT
      if wait_ready $BENCH_PORT $SERVER_PID 1800; then
        python $POC/bench.py run --url http://localhost:$BENCH_PORT --label stock --calls $BENCH_CALLS \
          --out $RESULT_DIR/$MODEL-$PROFILE-stock-round$r.json
      else echo "stock server did not start; skipping"; fi
      stop_server
    fi
    step "round $r: optimised"
    start_fast $BENCH_PORT
    wait_ready $BENCH_PORT $SERVER_PID 1800 || { tail -n 30 $LOG_DIR/$MODEL-$PROFILE-fast-server.log; die "optimised server did not start"; }
    python $POC/bench.py run --url http://localhost:$BENCH_PORT --label fast --calls $BENCH_CALLS \
      --out $RESULT_DIR/$MODEL-$PROFILE-fast-round$r.json
    stop_server
  done
  step "results"
  python $POC/bench.py report --results $RESULT_DIR --model $MODEL --profile $PROFILE
}

report_clocks() {  # an unpinned Jetson GPU ramps with load and skews short benchmarks: say so
  local d
  for d in /sys/class/devfreq/*gpu* /sys/devices/platform/*gpu*/devfreq/*; do
    [ -r $d/cur_freq ] || continue
    echo "GPU clock $(( $(cat $d/cur_freq) / 1000000 )) MHz (min $(( $(cat $d/min_freq) / 1000000 )), max $(( $(cat $d/max_freq) / 1000000 )))"
    [ "$(cat $d/min_freq)" = "$(cat $d/max_freq)" ] || echo "WARNING: GPU clock is not pinned. On a Jetson run 'sudo nvpmodel -m 0 && sudo jetson_clocks' on the host first."
    return
  done
}

serve() {
  step "serving the optimised stack on port $PORT (OpenAI API: http://localhost:$PORT/v1/chat/completions)"
  start_fast $PORT
  trap 'kill $SERVER_PID 2>/dev/null' TERM INT
  wait $SERVER_PID
}
