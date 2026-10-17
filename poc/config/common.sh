# Shared defaults; sourced by run.sh before config/<model>.sh. Every value can be overridden from the environment (.env).
POC=/poc
DATA=${DATA:-/data}
PROFILE=${PROFILE:-orin}                 # orin | dgpu
MODEL=${MODEL:-gemma4-e4b}               # gemma4-e4b | gemma4-12b | qwen35-4b
PORT=${PORT:-8000}                       # port of the optimised server that stays up at the end (`serve`)
BENCH_PORT=${BENCH_PORT:-18080}
BENCH_CALLS=${BENCH_CALLS:-60}           # recorded requests replayed per run
BENCH_ROUNDS=${BENCH_ROUNDS:-2}          # stock, optimised, stock, optimised, ...: the order cancels drift (thermal, clocks)
MODELS_DIR=$DATA/models
CKPT_DIR=$DATA/checkpoints/$MODEL
ENGINE_DIR=$DATA/engines/$MODEL/$PROFILE
LOG_DIR=$DATA/logs
RESULT_DIR=$DATA/results
mkdir -p "$MODELS_DIR" "$CKPT_DIR" "$ENGINE_DIR" "$LOG_DIR" "$RESULT_DIR"

step()  { echo; echo "== $* ($(date +%T))"; }
die()   { echo "ERROR: $*" >&2; exit 1; }

# Point the python path, the plugin and the binaries at one of the two builds (stock | fast).
use_stack() {
  STACK_ROOT=/opt/$1
  export PYTHONPATH=$STACK_ROOT:$STACK_ROOT/build/pybind
  export EDGELLM_PLUGIN_PATH=$STACK_ROOT/build/libNvInfer_edgellm_plugin.so
  LLM_BUILD=$STACK_ROOT/build/examples/llm/llm_build
}
