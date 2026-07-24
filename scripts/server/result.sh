#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# result.sh — pull the final result out of a run log without eyeballing it.
#
# Two jobs, auto-detected from the log:
#
#   * baseline / CO-Bench : the run announces progress with lines like
#         Dev Score 0.982 | Test Score 0.992 | Overall 0.985 [cost=$0.97 ...]
#         ... 🌟 New best solution found at iteration 31 ...
#     What you want is the LAST "New best" line together with the score line
#     just ABOVE it (the test score). This is the Ctrl-F-for-"New best"-then-
#     look-up workflow, automated. Context depth is adjustable.
#
#   * BLADE : the run ends with a report block
#         Best score        : 0.036234
#         Evaluations used  : 176
#         ...
#     `tail -N` cuts that off at a fixed line count. This prints from the
#     report header to the end of the file, however long it is.
#
# You rarely need to type a log path: refer to runs by recency instead.
#
# Usage:
#   ./scripts/server/result.sh [LOG | -N] [options]
#
#   LOG   Path to a run.log (or any log).
#   -N    The N-th most recent run under outputs/server/ (-1 = newest,
#         -2 = the one before, ...). Omit entirely to mean -1.
#
# Options:
#   -R, --recent N     Print the N most recent runs in one go, newest first,
#                      each under a header saying which log it came from.
#   -k, --keyword RE   Keyword regex to search for (overrides auto-detect).
#   -B, --before N     Lines to show BEFORE each match. (default: 3)
#   -A, --after N      Lines to show AFTER each match.  (default: 1)
#   -n, --num N        Show only the last N matches. 'all' = every match.
#                      (default: 1)
#   --from RE          Ignore keyword mode; print from the LAST line matching
#                      RE to the end of the file (flexible tail). This is what
#                      BLADE mode uses under the hood with RE='Best score'.
#   --report           Force BLADE report mode (== --from 'Best score').
#   --new-best         Force baseline mode (== --keyword 'New best').
#   -f, --follow       After printing, follow the log live (tail -f).
#   -l, --list         List recent run logs and exit.
#   -h, --help         This help.
#
# Short flags accept both the spaced and the glued grep-style form
# (-B 3 or -B3, -n all or -nall).
#
# Examples:
#   ./scripts/server/result.sh                       # newest run, auto
#   ./scripts/server/result.sh -2                    # the run before the newest
#   ./scripts/server/result.sh --recent 3            # the 3 newest runs at once
#   ./scripts/server/result.sh -B 5                  # more context above the hit
#   ./scripts/server/result.sh -n all                # every New best, not just last
#   ./scripts/server/result.sh path/to/run.log --report
#   ./scripts/server/result.sh -3 -k "Test Score" -B0 -A0
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LOG=""
INDEX=""         # N for the -N recency selector
RECENT=""        # N for --recent
KEYWORD=""
BEFORE=3
AFTER=1
NUM="1"
FROM=""
MODE="auto"      # auto | keyword | from
FOLLOW=0

die() { echo "ERROR: $*" >&2; exit 1; }

show_help() { sed -n '2,63p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# Every outputs/server/*/run.log, newest first.
_logs_by_recency() { ls -t "$REPO_ROOT"/outputs/server/*/run.log 2>/dev/null || true; }

# Resolve the N-th most recent run.log (1 = newest). Echoes path or nothing.
_nth_log() {
    local want="$1" i=0 l
    while IFS= read -r l; do
        i=$((i + 1))
        if [[ "$i" -eq "$want" ]]; then echo "$l"; return 0; fi
    done < <(_logs_by_recency)
    return 1
}

# A one-line status summary for a log (the footer's exit-status line, if any).
_log_status() {
    local log="$1" s
    s="$(grep -E '^ exit status' "$log" 2>/dev/null | tail -1 | sed 's/^ *//')"
    [[ -n "$s" ]] && echo "$s" || echo "(running or no footer yet)"
}

list_logs() {
    local dir="$REPO_ROOT/outputs/server" i=0 d
    [[ -d "$dir" ]] || { echo "No runs yet under outputs/server/"; return; }
    echo "Recent runs (newest first). Refer to them as -1, -2, ...:"
    while IFS= read -r d; do
        i=$((i + 1))
        [[ "$i" -gt 20 ]] && break
        local log="$d/run.log" status when
        when="$(date -r "$d" '+%Y-%m-%d %H:%M' 2>/dev/null || echo '?')"
        if [[ -f "$log" ]]; then status="$(_log_status "$log")"; else status="(no run.log)"; fi
        printf '  -%-2d  %s  %s\n           %s\n' "$i" "$when" "$(basename "$d")" "$status"
    done < <(ls -dt "$dir"/*/ 2>/dev/null)
}

# --- args -------------------------------------------------------------------
# Accept both the spaced form (-B 3) and the glued grep-style form (-B3).
while [[ $# -gt 0 ]]; do
    case "$1" in
        -R|--recent)  RECENT="$2"; shift 2 ;;
        -R[0-9]*)     RECENT="${1#-R}"; shift ;;
        -k|--keyword) KEYWORD="$2"; MODE="keyword"; shift 2 ;;
        -B|--before)  BEFORE="$2"; shift 2 ;;
        -A|--after)   AFTER="$2"; shift 2 ;;
        -n|--num)     NUM="$2"; shift 2 ;;
        -B[0-9]*)     BEFORE="${1#-B}"; shift ;;
        -A[0-9]*)     AFTER="${1#-A}"; shift ;;
        -n?*)         NUM="${1#-n}"; shift ;;
        -k?*)         KEYWORD="${1#-k}"; MODE="keyword"; shift ;;
        --from)       FROM="$2"; MODE="from"; shift 2 ;;
        --report)     FROM="Best score"; MODE="from"; shift ;;
        --new-best)   KEYWORD="New best"; MODE="keyword"; shift ;;
        -f|--follow)  FOLLOW=1; shift ;;
        -l|--list)    list_logs; exit 0 ;;
        -h|--help)    show_help; exit 0 ;;
        -[0-9]*)      INDEX="${1#-}"; shift ;;      # -1, -2, ... recency selector
        -*)           die "unknown option '$1' (try --help)" ;;
        *)            [[ -z "$LOG" ]] && LOG="$1" || die "unexpected argument '$1'"; shift ;;
    esac
done

[[ "$NUM" =~ ^([0-9]+|all)$ ]] || die "--num must be a number or 'all' (got '$NUM')"
[[ "$BEFORE" =~ ^[0-9]+$ ]] || die "--before must be a number"
[[ "$AFTER"  =~ ^[0-9]+$ ]] || die "--after must be a number"
[[ -z "$RECENT" || "$RECENT" =~ ^[0-9]+$ ]] || die "--recent must be a number (got '$RECENT')"
[[ -z "$INDEX"  || "$INDEX"  =~ ^[0-9]+$ ]] || die "recency selector must look like -1, -2, ..."
[[ -n "$LOG" && -n "$INDEX" ]] && die "give either a log path or a -N selector, not both"
[[ -n "$RECENT" && ( -n "$LOG" || -n "$INDEX" || "$FOLLOW" == "1" ) ]] \
    && die "--recent cannot be combined with a specific log or --follow"

# ===========================================================================
# extract_one <log> — run the detected/selected extraction against one log.
# ===========================================================================
extract_one() {
    local LOG="$1"
    local MODE="$MODE" KEYWORD="$KEYWORD" FROM="$FROM"

    # auto-detect per log (a --recent batch may mix BLADE and baseline runs)
    if [[ "$MODE" == "auto" ]]; then
        if grep -qE "NEW BEST ★|\[BLADE\]|Best score +:" "$LOG"; then
            MODE="from"; FROM="Best score"
        else
            MODE="keyword"; KEYWORD="New best"
        fi
    fi

    if [[ "$MODE" == "from" ]]; then
        local start
        start="$(grep -nE "$FROM" "$LOG" | tail -1 | cut -d: -f1 || true)"
        if [[ -z "$start" ]]; then
            echo "(no line matching '$FROM' found — is the run finished? showing last 40 lines)" >&2
            tail -40 "$LOG"
        else
            echo "----- from '$FROM' (line $start) to end -----"
            tail -n +"$start" "$LOG"
        fi
        return
    fi

    # keyword mode
    local total want
    total="$(grep -cE "$KEYWORD" "$LOG" || true)"
    if [[ "${total:-0}" -eq 0 ]]; then
        echo "(no line matching '$KEYWORD' in this log)" >&2
        echo "Tip: the run may still be starting, or try a different -k keyword." >&2
        local foot
        foot="$(grep -nE '^={10,}' "$LOG" | tail -2 | head -1 | cut -d: -f1 || true)"
        if [[ -n "$foot" ]]; then
            echo "     Showing the run's result footer instead:" >&2
            tail -n +"$foot" "$LOG"
        else
            echo "     Last few lines for reference:" >&2
            tail -8 "$LOG" >&2
        fi
        return
    fi

    if [[ "$NUM" == "all" ]]; then want="$total"; else want="$NUM"; fi
    echo "----- '$KEYWORD': showing last $want of $total match(es), -B$BEFORE -A$AFTER -----"
    awk -v kw="$KEYWORD" -v B="$BEFORE" -v A="$AFTER" -v want="$want" '
    { lines[NR] = $0 }
    $0 ~ kw {
        match_no++
        starts[match_no] = (NR - B > 0 ? NR - B : 1)
        ends[match_no]   = NR + A
        hit[match_no]    = NR
    }
    END {
        first = match_no - want + 1
        if (first < 1) first = 1
        for (m = first; m <= match_no; m++) {
            if (m > first) print "  ---"
            s = starts[m]; e = ends[m]
            if (e > NR) e = NR
            for (i = s; i <= e; i++) {
                prefix = (i == hit[m] ? ">> " : "   ")
                print prefix lines[i]
            }
        }
    }' "$LOG"
}

# ===========================================================================
# --recent N: iterate over the N newest runs, each under its own header.
# ===========================================================================
if [[ -n "$RECENT" ]]; then
    n_avail="$(_logs_by_recency | grep -c . || true)"
    [[ "${n_avail:-0}" -gt 0 ]] || die "no runs found under outputs/server/ (see --list)"
    show="$RECENT"; [[ "$show" -gt "$n_avail" ]] && show="$n_avail"
    for i in $(seq 1 "$show"); do
        log="$(_nth_log "$i")"
        [[ $i -gt 1 ]] && echo ""
        echo "==================================================================="
        echo "  [-$i]  $(basename "$(dirname "$log")")"
        echo "        log:    $log"
        echo "        status: $(_log_status "$log")"
        echo "==================================================================="
        extract_one "$log"
    done
    exit 0
fi

# ===========================================================================
# Single run: resolve the log (path, -N selector, or newest by default).
# ===========================================================================
if [[ -n "$INDEX" ]]; then
    LOG="$(_nth_log "$INDEX" || true)"
    [[ -n "$LOG" ]] || die "there is no -$INDEX run (only $(_logs_by_recency | grep -c . || echo 0) found; see --list)"
elif [[ -z "$LOG" ]]; then
    LOG="$(_nth_log 1 || true)"
    [[ -n "$LOG" ]] || die "no log given and none found under outputs/server/*/run.log (see --list)"
fi
[[ -f "$LOG" ]] || die "log not found: $LOG"

# Always annotate which log this came from.
echo "# log:    $LOG"
echo "# status: $(_log_status "$LOG")"
extract_one "$LOG"

if [[ "$FOLLOW" == "1" ]]; then
    echo ""
    echo "----- following $LOG (Ctrl-C to stop) -----"
    exec tail -f "$LOG"
fi
