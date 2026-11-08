#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_cp_queue.sh — three experiment lanes on the circle-packing tasks, each a
# tmux session working through its own queue one job at a time. The lanes run
# side by side; inside a lane the jobs are strictly sequential.
#
#   ./scripts/server/run_cp_queue.sh start   [t1 t2 t3]   # launch lanes (default: all), detached
#   ./scripts/server/run_cp_queue.sh status  [t1 t2 t3]   # done / running (live numbers) / left
#   ./scripts/server/run_cp_queue.sh compare [t1|t2|t3]   # tokens / reasoning / latency / score (default t3)
#   ./scripts/server/run_cp_queue.sh stop    [t1 t2 t3]   # kill lanes and their runs
#   DRY_RUN=1 ./scripts/server/run_cp_queue.sh start      # print all 19 commands, no API calls
#
# Lane t1 — tmux cp_t1_nav — swap the Navigator's model. 1h30 per job, 6 jobs.
#   nav_gpt5_low     Navigator = gpt-5, reasoning_effort=low
#   nav_gpt5_medium  Navigator = gpt-5, reasoning_effort=medium (gpt-5's default, i.e. stock SpecEvo)
#   nav_qwen3_30b    Navigator = qwen3-30b-a3b-instruct-2507 (the mutation model)
#   Only the Navigator's paradigm-shift calls change: the bootstrap's diverse
#   seeds stay on gpt-5 at its default effort, so every arm starts alike.
#
# Lane t2 — tmux cp_t2_budget — $3 per job, 8 jobs.
#   full             full SpecEvo
#   no_nav           w/o Navigator                        (--ablation no_paradigm)
#   no_nav_adv_freq  w/o Navigator, Advisor every $T2_ADVISOR_INTERVAL evals instead of 50
#   no_nav_adv_gpt5  w/o Navigator, Advisor on gpt-5 instead of the mutation model
#   $3 is the stop condition; T2_SECONDS (8h) only ends a run that cannot
#   spend $3 in that time, and still ends it gracefully with a summary.
#
# Lane t3 — tmux cp_t3_base — circle_packing only, 3h per job, one LLM call
# in flight at a time, 5 jobs in this order:
#   evox, adaevolve, gepa_native, openevolve_native  (baselines, gpt-5, 100 iterations
#                                                     max — baseline.yml's setting)
#   specevo          (qwen3-30b + gpt-5, --workers 1, 1 evaluator, bootstrap 1 at a time)
#   Then `compare t3` prints the token / latency / score table and writes
#   comparison.{md,csv,json} into the lane's directory.
#
# Tasks: circle_packing      (26 circles in the unit square)
#        circle_packing_rect (21 circles in a rectangle of perimeter 4)
#
# Outputs: outputs/server/$GROUP/<lane dir>/<task>__<arm>/
# Restarting is safe: a finished job is skipped; one that died half-way is
# moved aside to <dir>.incomplete_<ts> and run again.
#
# Knobs (environment variables, defaults in brackets):
#   GROUP [cp_exp]   WORKERS [4]   PE_INTERVAL [10]   EVAL_TIMEOUT [600]   THREADS_PER_EVAL [2]
#   MUTATION_MODEL [qwen3-30b-a3b-instruct-2507]   PARADIGM_MODEL / GPT5 [gpt-5]
#   T1_SECONDS [5400]  T1_HARD_TIMEOUT [6300]  T1_DOLLARS [15]   (spend is a runaway guard only)
#   T2_DOLLARS [3]     T2_SECONDS [28800]      T2_HARD_TIMEOUT [29700]   T2_ADVISOR_INTERVAL [10]
#   T2_ADVISOR_MAX_TOKENS [16000]   (gpt-5's hidden reasoning counts against it)
#   T3_SECONDS [10800] T3_HARD_TIMEOUT [11700] T3_DOLLARS [40]  T3_ITERATIONS [100]
#   BASELINE_MODEL [gpt-5]
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
SELF="$REPO_ROOT/scripts/server/run_cp_queue.sh"

GROUP="${GROUP:-cp_exp}"
WORKERS="${WORKERS:-4}"
PE_INTERVAL="${PE_INTERVAL:-10}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-600}"
THREADS_PER_EVAL="${THREADS_PER_EVAL:-2}"
MUTATION_MODEL="${MUTATION_MODEL:-openrouter/qwen/qwen3-30b-a3b-instruct-2507}"
PARADIGM_MODEL="${PARADIGM_MODEL:-openrouter/openai/gpt-5}"
GPT5="${GPT5:-openrouter/openai/gpt-5}"
QWEN="${QWEN:-openrouter/qwen/qwen3-30b-a3b-instruct-2507}"
BASELINE_MODEL="${BASELINE_MODEL:-openrouter/openai/gpt-5}"

T1_SECONDS="${T1_SECONDS:-5400}"
T1_HARD_TIMEOUT="${T1_HARD_TIMEOUT:-6300}"
T1_DOLLARS="${T1_DOLLARS-15}"
T2_DOLLARS="${T2_DOLLARS:-3}"
T2_SECONDS="${T2_SECONDS:-28800}"
T2_HARD_TIMEOUT="${T2_HARD_TIMEOUT:-29700}"
T2_ADVISOR_INTERVAL="${T2_ADVISOR_INTERVAL:-10}"
T2_ADVISOR_MAX_TOKENS="${T2_ADVISOR_MAX_TOKENS:-16000}"
T3_SECONDS="${T3_SECONDS:-10800}"
T3_HARD_TIMEOUT="${T3_HARD_TIMEOUT:-11700}"
T3_DOLLARS="${T3_DOLLARS-40}"
T3_ITERATIONS="${T3_ITERATIONS:-100}"

# Every knob above, so a lane started inside tmux sees the same values.
KNOBS=(GROUP WORKERS PE_INTERVAL EVAL_TIMEOUT THREADS_PER_EVAL MUTATION_MODEL PARADIGM_MODEL GPT5 QWEN
       BASELINE_MODEL T1_SECONDS T1_HARD_TIMEOUT T1_DOLLARS T2_DOLLARS T2_SECONDS T2_HARD_TIMEOUT
       T2_ADVISOR_INTERVAL T2_ADVISOR_MAX_TOKENS T3_SECONDS T3_HARD_TIMEOUT T3_DOLLARS T3_ITERATIONS)

DRY_RUN="${DRY_RUN:-0}"
GROUP_DIR="outputs/server/$GROUP"
# run_bench.sh --dry-run still creates the output dir and run.log; keep those
# out of the real group so a later `start` does not mistake them for crashes.
[[ "$DRY_RUN" == 1 ]] && GROUP_DIR="outputs/server/${GROUP}_dryrun"
VENV="$REPO_ROOT/.venv"

LANES=(t1 t2 t3)
CP_TASKS=(circle_packing circle_packing_rect)
T1_ARMS=(nav_gpt5_low nav_gpt5_medium nav_qwen3_30b)
T2_ARMS=(full no_nav no_nav_adv_freq no_nav_adv_gpt5)
T3_ARMS=(evox adaevolve gepa_native openevolve_native specevo)

die() { echo "ERROR: $*" >&2; exit 1; }
ts()  { date '+%Y-%m-%d %H:%M:%S'; }

lane_session() { case "$1" in t1) echo cp_t1_nav ;; t2) echo cp_t2_budget ;; t3) echo cp_t3_base ;; *) die "unknown lane '$1' (t1 | t2 | t3)" ;; esac; }
lane_dir()     { case "$1" in t1) echo "$GROUP_DIR/t1_navigator" ;; t2) echo "$GROUP_DIR/t2_budget3usd" ;; t3) echo "$GROUP_DIR/t3_baselines" ;; esac; }
lane_title()   { case "$1" in t1) echo "Navigator swap, ${T1_SECONDS}s each" ;; t2) echo "\$${T2_DOLLARS} per run" ;; t3) echo "baselines vs SpecEvo, ${T3_SECONDS}s each, 1 worker" ;; esac; }

lane_jobs() {   # lane -> "<lane> <task> <arm>" lines, in run order
    local lane="$1" arm task
    case "$lane" in
        t1) for arm in "${T1_ARMS[@]}"; do for task in "${CP_TASKS[@]}"; do echo "t1 $task $arm"; done; done ;;
        t2) for arm in "${T2_ARMS[@]}"; do for task in "${CP_TASKS[@]}"; do echo "t2 $task $arm"; done; done ;;
        t3) for arm in "${T3_ARMS[@]}"; do echo "t3 circle_packing $arm"; done ;;
    esac
}

job_kind() { case "$1:$2" in t3:specevo) echo blade ;; t3:*) echo baseline ;; *) echo blade ;; esac; }
job_dir()  { echo "$(lane_dir "$1")/$2__$3"; }

job_done() {    # lane task arm -> 0 when the run left its final result
    local d; d="$(job_dir "$1" "$2" "$3")"
    if [[ "$(job_kind "$1" "$3")" == blade ]]; then [[ -f "$d/summary.json" ]]
    else [[ -f "$d/run/best/best_program_info.json" ]]; fi
}

# ---------------------------------------------------------------------------
# The command for one job, into the global array CMD.
# ---------------------------------------------------------------------------
build_cmd() {   # lane task arm
    local lane="$1" task="$2" arm="$3" out run_id
    out="$(job_dir "$lane" "$task" "$arm")"
    run_id="${GROUP}_${lane}_${task}_${arm}"
    local bench="$REPO_ROOT/scripts/server/run_bench.sh"

    if [[ "$(job_kind "$lane" "$arm")" == baseline ]]; then
        CMD=("$bench" baseline --no-conda --no-install-deps
             --baseline "$arm" --benchmark-dir "benchmarks/math/$task"
             --model "$BASELINE_MODEL" --iterations "$T3_ITERATIONS"
             --seconds "$T3_SECONDS" --timeout "$T3_HARD_TIMEOUT" --dollars "$T3_DOLLARS"
             --run-id "$run_id" --output-dir "$out")
        [[ "$DRY_RUN" == 1 ]] && CMD+=(--dry-run)
        return 0
    fi

    local seconds hard dollars workers="$WORKERS" ablation=full adv=""
    case "$lane" in
        t1) seconds="$T1_SECONDS"; hard="$T1_HARD_TIMEOUT"; dollars="$T1_DOLLARS" ;;
        t2) seconds="$T2_SECONDS"; hard="$T2_HARD_TIMEOUT"; dollars="$T2_DOLLARS" ;;
        t3) seconds="$T3_SECONDS"; hard="$T3_HARD_TIMEOUT"; dollars="$T3_DOLLARS" ;;
    esac
    case "$lane:$arm" in
        t1:nav_gpt5_low)     adv="{\"navigator_model\":\"$GPT5\",\"navigator_reasoning_effort\":\"low\"}" ;;
        t1:nav_gpt5_medium)  adv="{\"navigator_model\":\"$GPT5\",\"navigator_reasoning_effort\":\"medium\"}" ;;
        t1:nav_qwen3_30b)    adv="{\"navigator_model\":\"$QWEN\"}" ;;
        t2:full)             ;;
        t2:no_nav)           ablation=no_paradigm ;;
        t2:no_nav_adv_freq)  ablation=no_paradigm; adv="{\"meta_advice_interval\":$T2_ADVISOR_INTERVAL}" ;;
        t2:no_nav_adv_gpt5)  ablation=no_paradigm
                             adv="{\"advisor_model\":\"$GPT5\",\"meta_advice_max_tokens\":$T2_ADVISOR_MAX_TOKENS}" ;;
        t3:specevo)          workers=1; adv='{"eval_processes":1,"init_concurrency":1}' ;;
        *) die "unknown arm '$arm' in lane $lane" ;;
    esac

    CMD=("$bench" blade --no-conda --no-install-deps
         --benchmark "levi/examples/$task" --ablation "$ablation"
         --seconds "$seconds" --timeout "$hard" --dollars "$dollars"
         --mutation-model "$MUTATION_MODEL" --paradigm-model "$PARADIGM_MODEL"
         --workers "$workers" --pe-interval "$PE_INTERVAL" --eval-timeout "$EVAL_TIMEOUT"
         --save-eval-code --run-id "$run_id" --output-dir "$out")
    [[ -n "$adv" ]] && CMD+=(--advanced-options "$adv")
    [[ "$DRY_RUN" == 1 ]] && CMD+=(--dry-run)
    return 0
}

run_job() {     # lane task arm
    local lane="$1" task="$2" arm="$3" out rc
    out="$(job_dir "$lane" "$task" "$arm")"
    if job_done "$lane" "$task" "$arm"; then
        echo "[$(ts)] skip   $task / $arm (already finished)"
        return 0
    fi
    if [[ -d "$out" && "$DRY_RUN" != 1 ]]; then
        local aside="$out.incomplete_$(date +%Y%m%d-%H%M%S)"
        mv "$out" "$aside"
        echo "[$(ts)] moved  half-finished $out -> $aside"
    fi
    build_cmd "$lane" "$task" "$arm"
    echo "[$(ts)] start  $task / $arm -> $out"
    set +e
    "${CMD[@]}" < /dev/null     # never let a run read the job list lane_main is looping over
    rc=$?
    set -e
    [[ "$DRY_RUN" == 1 ]] || echo "$rc" > "$out/.exit_code"
    echo "[$(ts)] end    $task / $arm  exit=$rc  $(job_done "$lane" "$task" "$arm" && echo finished || echo 'NO RESULT')"
    return 0    # one failed job must not stop the rest of the lane
}

lane_main() {   # lane
    local lane="$1" l task arm
    # The tmux server may predate this venv, so pin the interpreter here.
    export VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH"
    # Three lanes x up to 4 evaluator processes on 16 threads: cap the numeric
    # libraries so candidates cannot oversubscribe the box.
    export OMP_NUM_THREADS="$THREADS_PER_EVAL" OPENBLAS_NUM_THREADS="$THREADS_PER_EVAL" \
           MKL_NUM_THREADS="$THREADS_PER_EVAL" NUMEXPR_NUM_THREADS="$THREADS_PER_EVAL"
    mkdir -p "$(lane_dir "$lane")"
    echo "[$(ts)] lane $lane begins: $(lane_title "$lane")"
    while read -r l task arm; do
        run_job "$l" "$task" "$arm"
    done < <(lane_jobs "$lane")
    echo "[$(ts)] lane $lane finished."
}

# ---------------------------------------------------------------------------
pick_lanes() {  # args -> validated lane list (default: all)
    local l
    if [[ $# -eq 0 ]]; then echo "${LANES[*]}"; return; fi
    for l in "$@"; do lane_session "$l" >/dev/null; done
    echo "$*"
}

report_jobs() { # lanes... -> job lines for cp_report.py
    local lane l task arm
    for lane in "$@"; do
        while read -r l task arm; do
            echo "$l $task $arm $(job_kind "$l" "$arm") $REPO_ROOT/$(job_dir "$l" "$task" "$arm")"
        done < <(lane_jobs "$lane")
    done
}

py() { if [[ -x "$VENV/bin/python" ]]; then "$VENV/bin/python" "$@"; else python3 "$@"; fi; }

cmd_status() {
    local lanes s l
    # shellcheck disable=SC2046
    lanes=($(pick_lanes "$@"))
    report_jobs "${lanes[@]}" | py "$REPO_ROOT/scripts/server/cp_report.py" status
    echo
    for l in "${lanes[@]}"; do
        s="$(lane_session "$l")"
        tmux has-session -t "$s" 2>/dev/null && echo "tmux $s: alive" || echo "tmux $s: not running"
    done
}

cmd_compare() {
    local lane="${1:-t3}"
    lane_session "$lane" >/dev/null
    report_jobs "$lane" | py "$REPO_ROOT/scripts/server/cp_report.py" compare --out "$REPO_ROOT/$(lane_dir "$lane")"
}

cmd_stop() {
    local lanes l s task arm d p
    # shellcheck disable=SC2046
    lanes=($(pick_lanes "$@"))
    for l in "${lanes[@]}"; do
        s="$(lane_session "$l")"
        tmux kill-session -t "$s" 2>/dev/null && echo "killed tmux $s" || true
        # Belt and braces: any run that outlived its pane.
        while read -r _ task arm; do
            d="$(job_dir "$l" "$task" "$arm")"
            if [[ -s "$d/.run.pid" ]] && kill -0 "$(cat "$d/.run.pid")" 2>/dev/null; then
                p="$(cat "$d/.run.pid")"
                pkill -TERM -P "$p" 2>/dev/null || true; kill -TERM "$p" 2>/dev/null || true
                echo "sent SIGTERM to $l $task / $arm (pid $p)"
            fi
        done < <(lane_jobs "$l")
    done
}

cmd_start() {
    local lanes l s env_args k
    # shellcheck disable=SC2046
    lanes=($(pick_lanes "$@"))
    command -v tmux >/dev/null || die "tmux is not installed"
    [[ -x "$VENV/bin/python" ]] || die "no venv at $VENV"
    "$VENV/bin/python" -c 'import numpy, scipy, litellm, levi, skydiscover' 2>/dev/null \
        || die "$VENV cannot import numpy / scipy / litellm / levi / skydiscover"
    [[ -f .env ]] || die "no .env with OPENAI_API_KEY=sk-or-v1-..."

    if [[ "$DRY_RUN" == 1 ]]; then
        for l in "${lanes[@]}"; do lane_main "$l"; done
        return 0
    fi

    for l in "${lanes[@]}"; do
        s="$(lane_session "$l")"
        tmux has-session -t "$s" 2>/dev/null && die "tmux session '$s' already exists (status: $0 status, stop: $0 stop $l)"
    done

    local key
    key="$(sed -n 's/^\(OPENROUTER_API_KEY\|OPENAI_API_KEY\)=//p' .env | head -1)"
    if [[ -n "$key" ]] && command -v curl >/dev/null; then
        curl -s -m 15 https://openrouter.ai/api/v1/key -H "Authorization: Bearer $key" \
          | "$VENV/bin/python" -c '
import json, sys
try:
    d = json.load(sys.stdin)["data"]
except Exception:
    print("key check  : could not read /api/v1/key"); sys.exit(0)
lim, rem = d.get("limit"), d.get("limit_remaining")
print("key check  : used $%.2f, limit %s, remaining %s"
      % (d.get("usage", 0), "none" if lim is None else "$%.0f" % lim,
         "unlimited" if rem is None else "$%.2f" % rem))
' || true
    fi

    env_args="HOME='$HOME'"
    for k in "${KNOBS[@]}"; do env_args+=" $k='${!k}'"; done
    for l in "${lanes[@]}"; do
        mkdir -p "$(lane_dir "$l")"
        tmux new-session -d -s "$(lane_session "$l")" -c "$REPO_ROOT" \
            "env $env_args bash '$SELF' _lane $l 2>&1 | tee -a '$(lane_dir "$l")/lane.log'; \
echo; bash '$SELF' status $l; exec bash"
        printf '%-3s tmux %-13s %s  ->  %s/\n' "$l" "$(lane_session "$l")" "$(lane_title "$l")" "$(lane_dir "$l")"
    done
    cat <<EOF

watch      : tmux attach -t $(lane_session "${lanes[0]}")     (detach: Ctrl-b then d)
progress   : $0 status            (live: watch -n 60 $0 status)
t3 table   : $0 compare t3
stop       : $0 stop [t1 t2 t3]
EOF
}

case "${1:-status}" in
    start)   shift; cmd_start "$@" ;;
    status)  shift; cmd_status "$@" ;;
    compare|table) shift; cmd_compare "$@" ;;
    stop)    shift; cmd_stop "$@" ;;
    _lane)   lane_main "${2:?lane}" ;;
    -h|--help) sed -n '2,52p' "$0" ;;
    *) die "unknown command '$1' (start | status | compare | stop)" ;;
esac
