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
# Usage:
#   ./scripts/server/result.sh [LOG] [options]
#
#   LOG   Path to a run.log (or any log). Omit to use the newest
#         outputs/server/<run-id>/run.log.
#
# Options:
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
# Examples:
#   ./scripts/server/result.sh                       # newest run, auto
#   ./scripts/server/result.sh -B 5                  # more context above the hit
#   ./scripts/server/result.sh -n all                # every New best, not just last
#   ./scripts/server/result.sh path/to/run.log --report
#   ./scripts/server/result.sh -k "Test Score" -B 0 -A 0 -n 1
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

LOG=""
KEYWORD=""
BEFORE=3
AFTER=1
NUM="1"
FROM=""
MODE="auto"      # auto | keyword | from
FOLLOW=0

die() { echo "ERROR: $*" >&2; exit 1; }

show_help() { sed -n '2,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

list_logs() {
    local dir="$REPO_ROOT/outputs/server"
    [[ -d "$dir" ]] || { echo "No runs yet under outputs/server/"; return; }
    echo "Recent runs (newest first):"
    # newest first; show run id, mtime, and the last status line if present
    ls -dt "$dir"/*/ 2>/dev/null | head -20 | while read -r d; do
        local log="$d/run.log" status="" when
        when="$(date -r "$d" '+%Y-%m-%d %H:%M' 2>/dev/null || echo '?')"
        if [[ -f "$log" ]]; then
            status="$(grep -E '^ exit status' "$log" 2>/dev/null | tail -1 | sed 's/^ *//')"
            [[ -z "$status" ]] && status="(running or no footer)"
        else
            status="(no run.log)"
        fi
        printf '  %s  %s\n      %s\n' "$when" "$(basename "$d")" "$status"
    done
}

# --- args -------------------------------------------------------------------
# Accept both the spaced form (-B 3) and the glued grep-style form (-B3).
while [[ $# -gt 0 ]]; do
    case "$1" in
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
        -*)           die "unknown option '$1' (try --help)" ;;
        *)            [[ -z "$LOG" ]] && LOG="$1" || die "unexpected argument '$1'"; shift ;;
    esac
done

[[ "$NUM" =~ ^([0-9]+|all)$ ]] || die "--num must be a number or 'all' (got '$NUM')"
[[ "$BEFORE" =~ ^[0-9]+$ ]] || die "--before must be a number"
[[ "$AFTER"  =~ ^[0-9]+$ ]] || die "--after must be a number"

# --- resolve the log --------------------------------------------------------
if [[ -z "$LOG" ]]; then
    LOG="$(ls -t "$REPO_ROOT"/outputs/server/*/run.log 2>/dev/null | head -1 || true)"
    [[ -n "$LOG" ]] || die "no log given and none found under outputs/server/*/run.log (see --list)"
    echo "# log: $LOG" >&2
fi
[[ -f "$LOG" ]] || die "log not found: $LOG"

# --- auto-detect mode -------------------------------------------------------
if [[ "$MODE" == "auto" ]]; then
    if grep -qE "NEW BEST ★|\[BLADE\]|Best score +:" "$LOG"; then
        MODE="from"; FROM="Best score"
    else
        MODE="keyword"; KEYWORD="New best"
    fi
fi

# ===========================================================================
# from-mode: print from the LAST line matching $FROM to end of file.
# ===========================================================================
if [[ "$MODE" == "from" ]]; then
    start="$(grep -nE "$FROM" "$LOG" | tail -1 | cut -d: -f1 || true)"
    if [[ -z "$start" ]]; then
        echo "(no line matching '$FROM' found — is the run finished? showing last 40 lines)" >&2
        tail -40 "$LOG"
    else
        echo "----- from '$FROM' (line $start) to end -----"
        tail -n +"$start" "$LOG"
    fi

# ===========================================================================
# keyword-mode: show the last N matches, each with B lines before / A after.
# ===========================================================================
else
    total="$(grep -cE "$KEYWORD" "$LOG" || true)"
    if [[ "${total:-0}" -eq 0 ]]; then
        echo "(no line matching '$KEYWORD' in this log)" >&2
        echo "Tip: the run may still be starting, or try a different -k keyword." >&2
        # Fall back to the run_bench footer (from the last '====' banner to EOF)
        # which always carries the final cost totals / results dir; else tail.
        foot="$(grep -nE '^={10,}' "$LOG" | tail -2 | head -1 | cut -d: -f1 || true)"
        if [[ -n "$foot" ]]; then
            echo "     Showing the run's result footer instead:" >&2
            tail -n +"$foot" "$LOG"
        else
            echo "     Last few lines for reference:" >&2
            tail -8 "$LOG" >&2
        fi
    else
        if [[ "$NUM" == "all" ]]; then
            want="$total"
        else
            want="$NUM"
        fi
        echo "----- '$KEYWORD': showing last $want of $total match(es), -B$BEFORE -A$AFTER -----"
        # awk: buffer B previous lines; on a match, emit the buffer + match +
        # the next A lines; collect blocks and print only the final $want.
        awk -v kw="$KEYWORD" -v B="$BEFORE" -v A="$AFTER" -v want="$want" '
        {
            lines[NR] = $0
        }
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
    fi
fi

# --- optional live follow ---------------------------------------------------
if [[ "$FOLLOW" == "1" ]]; then
    echo ""
    echo "----- following $LOG (Ctrl-C to stop) -----"
    exec tail -f "$LOG"
fi
