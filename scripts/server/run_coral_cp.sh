#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_coral_cp.sh — CORAL (github.com/Human-Agent-Society/CORAL) on the two
# circle-packing tasks, side by side in two tmux lanes, each until 100 graded
# attempts or 3 hours, whichever comes first, with every LLM call's tokens and
# cost recorded through CORAL's LiteLLM gateway.
#
#   ./scripts/server/setup_coral.sh                     # once: uv, CORAL, OpenCode, Frontier-CS
#   ./scripts/server/run_coral_cp.sh check              # offline pre-flight + `coral validate` (no LLM calls)
#   ./scripts/server/run_coral_cp.sh smoke              # optional: 4-minute live run of c1 into GROUP=coral_smoke
#   ./scripts/server/run_coral_cp.sh start   [c1 c2]    # launch the lanes (default: both), detached
#   ./scripts/server/run_coral_cp.sh status  [c1 c2]    # state, best, evals, cost, tokens, wall
#   ./scripts/server/run_coral_cp.sh compare [c1 c2]    # writes comparison.{md,csv,json}
#   ./scripts/server/run_coral_cp.sh stop    [c1 c2]    # SIGTERM -> CORAL shuts its agents down
#
# Lanes:
#   c1  tmux coral_cp   circle_packing       26 circles in the unit square      (gateway :PORT_BASE+1)
#   c2  tmux coral_cpr  circle_packing_rect  21 circles, rectangle perimeter 4  (gateway :PORT_BASE+2)
#
# The same machinery runs CORAL on run_pk_queue.sh's two tasks through
# scripts/server/run_coral_pk.sh (CORAL_SUITE=pk; same commands, lanes p1 p2,
# 3 hours each and no eval cap by default):
#   p1  tmux coral_poly polyominoes          CORAL examples/frontier_cs_algo/0, graded by our local
#                                            judge (benchmarks/coral_tasks/polyominoes/judge.py, the
#                                            baselines' and SpecEvo's; CORAL's needs Docker), with
#                                            Frontier-CS's statement     (gateway :PORT_BASE+3)
#   p2  tmux coral_kern kernel_builder       CORAL examples/kernel_builder, CORAL's grader (cycles,
#                                            lower is better); best = 147734 / cycles  (gateway :PORT_BASE+4)
#
# What runs: CORAL's own tasks (CORAL_DIR/examples/math/<task>: task text, seed,
# grader), with $CORAL_AGENTS OpenCode agent(s) on $CORAL_MODEL. Agent traffic
# goes OpenCode -> CORAL gateway (LiteLLM, logs every request with its usage)
# -> OpenRouter. CORAL has neither a clock nor an eval limit of its own, so
# the lane sends it SIGTERM after $CORAL_SECONDS, or as soon as
# $CORAL_MAX_EVALS attempts are graded (checked every $CORAL_EVAL_POLL s;
# crashes and timeouts count, as in the status table's "evals"). CORAL handles
# SIGTERM as a normal shutdown (agents, grader daemon and gateway stop;
# attempts stay on disk). An eval already running then may still land, so a
# capped run can end one or two over the cap.
#
# Outputs: outputs/server/$GROUP/<task>/
#     task.yaml, seed/, grader/, litellm_config.yaml   the job as CORAL sees it
#     coral.log                                        CORAL's own console
#     results/<slug>/<ts>/.coral/public/attempts/      every graded attempt
#     results/<slug>/<ts>/.coral/public/gateway/requests.jsonl   every LLM call
# Restarting is safe: a finished job is skipped; one that died is moved aside.
#
# Knobs (environment variables, defaults in brackets):
#   GROUP [coral_exp]   CORAL_SECONDS [10800]   CORAL_KILL_AFTER [600]
#   CORAL_MAX_EVALS [100; pk: 0] (0 = no cap)   CORAL_EVAL_POLL [10]
#   CORAL_MODEL [openrouter/openai/gpt-5]   CORAL_AGENTS [1]   PORT_BASE [4100]
#   CORAL_DIR [/home/ubuntu/projects/CORAL]   CORAL_SUITE [cp] (cp: c1 c2 | pk: p1 p2)
#   FRONTIER_CS_DIR [/home/ubuntu/projects/Frontier-CS]   JUDGE_WORKERS [3]   (polyominoes)
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
SELF="$REPO_ROOT/scripts/server/run_coral_cp.sh"
TOOLS="$REPO_ROOT/scripts/server/coral_tools.py"

CORAL_SUITE="${CORAL_SUITE:-cp}"
case "$CORAL_SUITE" in
    cp) LANES=(c1 c2); DEFAULT_MAX_EVALS=100 ;;
    pk) LANES=(p1 p2); DEFAULT_MAX_EVALS=0 ;;
    *)  echo "ERROR: CORAL_SUITE must be cp or pk, not '$CORAL_SUITE'" >&2; exit 1 ;;
esac
ENTRY="$REPO_ROOT/scripts/server/run_coral_$CORAL_SUITE.sh"   # what the user types, for messages

GROUP="${GROUP:-coral_exp}"
CORAL_SECONDS="${CORAL_SECONDS:-10800}"
CORAL_KILL_AFTER="${CORAL_KILL_AFTER:-600}"
CORAL_MAX_EVALS="${CORAL_MAX_EVALS:-$DEFAULT_MAX_EVALS}"
CORAL_EVAL_POLL="${CORAL_EVAL_POLL:-10}"
CORAL_MODEL="${CORAL_MODEL:-openrouter/openai/gpt-5}"
CORAL_AGENTS="${CORAL_AGENTS:-1}"
PORT_BASE="${PORT_BASE:-4100}"
CORAL_DIR="${CORAL_DIR:-/home/ubuntu/projects/CORAL}"
FRONTIER_CS_DIR="${FRONTIER_CS_DIR:-/home/ubuntu/projects/Frontier-CS}"
JUDGE_WORKERS="${JUDGE_WORKERS:-3}"
KNOBS=(CORAL_SUITE GROUP CORAL_SECONDS CORAL_KILL_AFTER CORAL_MAX_EVALS CORAL_EVAL_POLL CORAL_MODEL CORAL_AGENTS
       PORT_BASE CORAL_DIR FRONTIER_CS_DIR JUDGE_WORKERS)

GROUP_DIR="outputs/server/$GROUP"
VENV="$REPO_ROOT/.venv"
CORAL_BIN="$CORAL_DIR/.venv/bin/coral"

die() { echo "ERROR: $*" >&2; exit 1; }
ts()  { date '+%Y-%m-%d %H:%M:%S'; }

lane_session() {
    case " ${LANES[*]} " in *" $1 "*) ;; *) die "unknown lane '$1' for $(basename "$ENTRY") (${LANES[*]})" ;; esac
    case "$1" in c1) echo coral_cp ;; c2) echo coral_cpr ;; p1) echo coral_poly ;; p2) echo coral_kern ;; esac
}
lane_task() { case "$1" in c1) echo circle_packing ;; c2) echo circle_packing_rect ;; p1) echo polyominoes ;; p2) echo kernel_builder ;; esac; }
lane_port() { case "$1" in c1) echo $((PORT_BASE + 1)) ;; c2) echo $((PORT_BASE + 2)) ;; p1) echo $((PORT_BASE + 3)) ;; p2) echo $((PORT_BASE + 4)) ;; esac; }
job_dir()      { echo "$GROUP_DIR/$(lane_task "$1")"; }
# Finished = ran out its clock (timeout's 124; 0 if CORAL ended on its own) or
# hit the eval cap (SIGTERM: 0, or 143 if CORAL died of the signal), and not
# cut short by `stop`, after which CORAL also exits 0.
job_done()     { local d; d="$(job_dir "$1")"; [[ -f "$d/.exit_code" && ! -e "$d/.stopped_by_user" ]] \
                     && { grep -qx '0\|124' "$d/.exit_code" || [[ -e "$d/.eval_cap_reached" ]]; }; }

py() { if [[ -x "$VENV/bin/python" ]]; then "$VENV/bin/python" "$@"; else python3 "$@"; fi; }

coral_env() {
    # coral, opencode and uv on PATH -- also for the agents, which run `coral eval`.
    export PATH="$CORAL_DIR/.venv/bin:$HOME/.opencode/bin:$HOME/.local/bin:$PATH"
    local key
    key="$(sed -n 's/^\(OPENROUTER_API_KEY\|OPENAI_API_KEY\)=//p' "$REPO_ROOT/.env" | head -1)"
    [[ -n "$key" ]] || die "no OPENROUTER_API_KEY / OPENAI_API_KEY in $REPO_ROOT/.env"
    export OPENROUTER_API_KEY="$key"
    # The agents' OPENAI_API_KEY is set by CORAL to their gateway key; never leak ours.
    unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_API_BASE
}

prepare_job() {     # lane [dir]
    local lane="$1" d="${2:-$(job_dir "$1")}"
    py "$TOOLS" prepare --task "$(lane_task "$lane")" --job-dir "$d" --coral-dir "$CORAL_DIR" \
        --port "$(lane_port "$lane")" --model "$CORAL_MODEL" --agents "$CORAL_AGENTS" \
        --frontier-cs-dir "$FRONTIER_CS_DIR" --judge-workers "$JUDGE_WORKERS" >/dev/null
}

port_free() { ! (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

kill_leftovers() {  # job_dir -- processes of this run still alive (an orphaned agent keeps spending)
    # Every process of the run inherits CORAL_JOB_TAG (CORAL passes its
    # environment on to agents, tools and graders); a shell you opened in the
    # job dir to look at results does not, so it is never touched.
    local tag p pids=()
    tag="$(cat "$1/.job_tag" 2>/dev/null)" || return 0
    [[ -n "$tag" ]] || return 0
    for p in /proc/[0-9]*; do
        [[ "${p#/proc/}" == "$$" ]] && continue
        { tr '\0' '\n' < "$p/environ"; } 2>/dev/null | grep -qxF "CORAL_JOB_TAG=$tag" && pids+=("${p#/proc/}")
    done
    [[ ${#pids[@]} -eq 0 ]] && return 0
    echo "[$(ts)] killing ${#pids[@]} leftover process(es) of $1: ${pids[*]}"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 10
    kill -KILL "${pids[@]}" 2>/dev/null || true
}

save_pricing() {    # OpenRouter's per-token prices for CORAL_MODEL -> $GROUP_DIR/pricing.json
    mkdir -p "$GROUP_DIR"
    curl -s -m 20 https://openrouter.ai/api/v1/models | py -c '
import json, sys
model, out = sys.argv[1], sys.argv[2]
slug = model.split("openrouter/", 1)[-1]
try:
    data = json.load(sys.stdin)["data"]
except Exception:
    sys.exit(0)
for m in data:
    if m.get("id") == slug:
        p = {k: float(v) for k, v in (m.get("pricing") or {}).items() if k in ("prompt", "completion", "input_cache_read")}
        json.dump({model: p}, open(out, "w"), indent=2)
        print("pricing    :", slug, {k: f"${v * 1e6:.3f}/M" for k, v in p.items()})
' "$CORAL_MODEL" "$GROUP_DIR/pricing.json" || true
}

eval_cap_watch() {  # job_dir pid -- SIGTERM CORAL's process group once CORAL_MAX_EVALS attempts are graded
    local out="$1" pid="$2" n
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$CORAL_EVAL_POLL"
        n="$(py "$TOOLS" count --job-dir "$out" 2>/dev/null)" || continue
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        if (( n >= CORAL_MAX_EVALS )); then
            date +%s > "$out/.eval_cap_reached"
            echo "[$(ts)] cap    $n graded attempts (CORAL_MAX_EVALS=$CORAL_MAX_EVALS): stopping CORAL"
            kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
            return 0
        fi
    done
}

run_job() {     # lane
    local lane="$1" out rc pid port watcher="" why
    out="$(job_dir "$lane")"
    port="$(lane_port "$lane")"
    if job_done "$lane"; then
        echo "[$(ts)] skip   $(lane_task "$lane") (already finished)"
        return 0
    fi
    if [[ -d "$out" ]]; then
        local aside="$out.incomplete_$(date +%Y%m%d-%H%M%S)"
        mv "$out" "$aside"
        echo "[$(ts)] moved  half-finished $out -> $aside"
    fi
    port_free "$port" || { echo "[$(ts)] ERROR  port $port is busy (stale gateway?): lsof -i :$port"; return 0; }
    prepare_job "$lane"
    echo "[$(ts)] start  $(lane_task "$lane"): CORAL, $CORAL_AGENTS x OpenCode on $CORAL_MODEL, ${CORAL_SECONDS}s$( (( CORAL_MAX_EVALS > 0 )) && echo " or $CORAL_MAX_EVALS evals"), gateway :$port -> $out"
    date +%s > "$out/.started_at"
    local tag="coral_cp:$REPO_ROOT/$out:$(date +%s%N)"
    echo "$tag" > "$out/.job_tag"
    set +e
    # setsid: CORAL, its agents and grader in one process group we can clean up.
    # env -u PWD: CORAL hands its own environment to the agents, and OpenCode
    # creates its session in $PWD when set -- the job dir here, which has no
    # provider config -- instead of the worktree it was started in. Without
    # PWD it uses the real working directory.
    ( cd "$out" && exec env -u PWD CORAL_JOB_TAG="$tag" setsid timeout --signal=TERM --kill-after="$CORAL_KILL_AFTER" "$CORAL_SECONDS" \
          "$CORAL_BIN" start --config "$REPO_ROOT/$out/task.yaml" ) < /dev/null >> "$out/coral.log" 2>&1 &
    pid=$!
    echo "$pid" > "$out/.run.pid"
    if (( CORAL_MAX_EVALS > 0 )); then
        eval_cap_watch "$out" "$pid" &
        watcher=$!
    fi
    wait "$pid"
    rc=$?
    if [[ -n "$watcher" ]]; then kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null; fi
    set -e
    date +%s > "$out/.ended_at"
    echo "$rc" > "$out/.exit_code"
    # Anything CORAL left behind (an agent mid-tool-call, a grader worker).
    kill -TERM -- "-$pid" 2>/dev/null && sleep 10 && kill -KILL -- "-$pid" 2>/dev/null || true
    kill_leftovers "$out"
    if [[ -e "$out/.eval_cap_reached" ]]; then why="eval cap reached: $CORAL_MAX_EVALS evals"
    elif [[ $rc == 124 ]]; then why="time budget reached: ${CORAL_SECONDS}s"
    else why="see coral.log"; fi
    echo "[$(ts)] end    $(lane_task "$lane")  exit=$rc ($why)"
    return 0
}

lane_main() {
    local lane="$1"
    coral_env
    mkdir -p "$GROUP_DIR"
    echo "[$(ts)] lane $lane begins"
    run_job "$lane"
    echo "[$(ts)] lane $lane finished."
}

pick_lanes() {
    local l
    if [[ $# -eq 0 ]]; then echo "${LANES[*]}"; return; fi
    for l in "$@"; do lane_session "$l" >/dev/null; done
    echo "$*"
}

report_jobs() { local l; for l in "$@"; do echo "$l $(lane_task "$l") $REPO_ROOT/$(job_dir "$l")"; done; }

cmd_status() {
    local lanes l s
    # shellcheck disable=SC2046
    lanes=($(pick_lanes "$@"))
    report_jobs "${lanes[@]}" | py "$TOOLS" status --model "$CORAL_MODEL" --group-dir "$REPO_ROOT/$GROUP_DIR" \
        --max-evals "$CORAL_MAX_EVALS"
    echo
    for l in "${lanes[@]}"; do
        s="$(lane_session "$l")"
        tmux has-session -t "=$s" 2>/dev/null && echo "tmux $s: alive" || echo "tmux $s: not running"
    done
    echo "attempt log: (cd $(job_dir "${lanes[0]}") && PATH=$CORAL_DIR/.venv/bin:\$PATH coral log -n 10)"
}

cmd_compare() {
    local lanes
    # shellcheck disable=SC2046
    lanes=($(pick_lanes "$@"))
    report_jobs "${lanes[@]}" | py "$TOOLS" compare --model "$CORAL_MODEL" --group-dir "$REPO_ROOT/$GROUP_DIR" \
        --out "$REPO_ROOT/$GROUP_DIR"
}

cmd_stop() {
    local lanes l s d p
    # shellcheck disable=SC2046
    lanes=($(pick_lanes "$@"))
    for l in "${lanes[@]}"; do
        d="$(job_dir "$l")"
        if [[ -s "$d/.run.pid" ]] && kill -0 "$(cat "$d/.run.pid")" 2>/dev/null; then
            p="$(cat "$d/.run.pid")"
            date +%s > "$d/.stopped_by_user"      # so `start` re-runs it instead of skipping it
            kill -TERM -- "-$p" 2>/dev/null || kill -TERM "$p" 2>/dev/null || true
            echo "sent SIGTERM to CORAL $(lane_task "$l") (process group $p); waiting up to 60s for a clean shutdown"
            for _ in $(seq 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
        fi
        [[ -d "$d" ]] && kill_leftovers "$d"
        s="$(lane_session "$l")"
        tmux kill-session -t "=$s" 2>/dev/null && echo "killed tmux $s" || true
    done
}

cmd_check() {   # no LLM calls
    local ok=1 l d
    coral_env
    for t in uv opencode coral tmux g++; do
        if command -v "$t" >/dev/null; then echo "ok   $t: $(command -v "$t")"; else echo "FAIL $t not found (run scripts/server/setup_coral.sh)"; ok=0; fi
    done
    [[ "$ok" == 1 ]] || return 1
    echo "ok   opencode $(opencode --version 2>/dev/null)  coral $(coral --version 2>/dev/null || true)"
    for l in "${LANES[@]}"; do
        port_free "$(lane_port "$l")" && echo "ok   port $(lane_port "$l") free" || { echo "FAIL port $(lane_port "$l") busy"; ok=0; }
    done
    for l in "${LANES[@]}"; do
        d="outputs/server/${GROUP}_check/$(lane_task "$l")"
        rm -rf "$d"
        prepare_job "$l" "$d"
        echo "== coral validate $(lane_task "$l") (grades the seed with CORAL's grader)"
        local out
        out="$(cd "$d" && coral validate . 2>&1)" || ok=0
        echo "$out" | tail -15 | sed 's/^/   /'
        # A seed the grader cannot score means every attempt would fail the same way.
        if echo "$out" | grep -q "Score: None\|Evaluation failed\|Traceback"; then
            echo "FAIL $(lane_task "$l"): CORAL's grader could not score the seed"; ok=0
        fi
    done
    [[ "$ok" == 1 ]] && echo "check: OK (job dirs for inspection under outputs/server/${GROUP}_check/)" || { echo "check: FAILED"; return 1; }
}

cmd_smoke() {   # a few minutes of real CORAL, to see agents, evals and cost tracking work end to end
    local lanes l
    # shellcheck disable=SC2046
    lanes=($(pick_lanes "${@:-${LANES[0]}}"))
    for l in "${lanes[@]}"; do
        echo "smoke: CORAL on $(lane_task "$l") for 240s into outputs/server/coral_smoke (a few cents)"
        rm -rf "outputs/server/coral_smoke/$(lane_task "$l")"
        GROUP=coral_smoke CORAL_SECONDS=240 CORAL_KILL_AFTER=120 bash "$SELF" _lane "$l"
    done
    GROUP=coral_smoke bash "$SELF" status "${lanes[@]}"
    echo "If evals > 0 and calls > 0 with a cost, the pipeline works. Agent output: outputs/server/coral_smoke/<task>/results/*/*/.coral/public/logs/"
}

cmd_start() {
    local lanes l s env_args k
    # shellcheck disable=SC2046
    lanes=($(pick_lanes "$@"))
    command -v tmux >/dev/null || die "tmux is not installed"
    [[ -x "$CORAL_BIN" ]] || die "no coral at $CORAL_BIN (run scripts/server/setup_coral.sh)"
    [[ -x "$HOME/.opencode/bin/opencode" ]] || die "no opencode at ~/.opencode/bin (run scripts/server/setup_coral.sh)"
    for l in "${lanes[@]}"; do
        s="$(lane_session "$l")"
        tmux has-session -t "=$s" 2>/dev/null && die "tmux session '$s' already exists (status: $ENTRY status, stop: $ENTRY stop $l)"
    done
    save_pricing
    env_args="HOME='$HOME'"
    for k in "${KNOBS[@]}"; do env_args+=" $k='${!k}'"; done
    for l in "${lanes[@]}"; do
        mkdir -p "$GROUP_DIR"
        tmux new-session -d -s "$(lane_session "$l")" -c "$REPO_ROOT" \
            "env $env_args bash '$SELF' _lane $l 2>&1 | tee -a '$GROUP_DIR/lane_$l.log'; \
echo; bash '$SELF' status $l; exec bash"
        printf '%-3s tmux %-10s %-20s %s agent(s), %ss%s  ->  %s/\n' "$l" "$(lane_session "$l")" "$(lane_task "$l")" \
            "$CORAL_AGENTS" "$CORAL_SECONDS" "$( (( CORAL_MAX_EVALS > 0 )) && echo " or $CORAL_MAX_EVALS evals")" "$(job_dir "$l")"
    done
    cat <<EOF

watch      : tmux attach -t $(lane_session "${lanes[0]}")     (detach: Ctrl-b then d);  CORAL console: tail -f $(job_dir "${lanes[0]}")/coral.log
progress   : $ENTRY status            (live: watch -n 60 $ENTRY status)
table      : $ENTRY compare
stop       : $ENTRY stop [${LANES[*]}]
EOF
}

case "${1:-status}" in
    check)   shift; cmd_check ;;
    smoke)   shift; cmd_smoke "$@" ;;
    start)   shift; cmd_start "$@" ;;
    status)  shift; cmd_status "$@" ;;
    compare|table) shift; cmd_compare "$@" ;;
    stop)    shift; cmd_stop "$@" ;;
    _lane)   lane_main "${2:?lane}" ;;
    -h|--help) sed -n '2,/^# -\{20,\}$/p' "$SELF" ;;
    *) die "unknown command '$1' (check | smoke | start | status | compare | stop)" ;;
esac
