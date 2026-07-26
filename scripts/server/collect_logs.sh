#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# collect_logs.sh — gom run.log của các tmux session vào một thư mục, đặt tên
# theo session, để kéo về máy cá nhân vẽ hình.
#
# Vấn đề nó giải: tên thư mục run (`outputs/server/<run-id>`) có kèm ngày giờ
# nên không ai nhớ nổi, trong khi cái bạn nhớ là tên tmux session
# (`ada_signal_10`, `specevo_sql_10`, ...). Script tra ngược session -> run dir
# bằng chính dòng lệnh mà tmux đã dùng để khởi động pane (nó chứa
# `--output-dir`), nên mapping là chính xác chứ không đoán theo giờ.
#
#   ./scripts/server/collect_logs.sh                        # mọi session đang chạy
#   ./scripts/server/collect_logs.sh ada_signal_10 evox_signal_10
#   ./scripts/server/collect_logs.sh --out /tmp/logs --no-tar
#
# Kết quả: <out>/<session>.log  +  <out>/manifest.tsv  +  <out>.tar.gz
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

OUT_DIR="$REPO_ROOT/outputs/collected_logs"
MAKE_TAR=1
SESSIONS=()

usage() {
    cat <<'EOF'
Usage:
  collect_logs.sh [options] [SESSION ...]

Không truyền SESSION nào = gom tất cả tmux session đang có.

Options
  --out DIR     Thư mục đích. (default: outputs/collected_logs)
  --no-tar      Không nén .tar.gz ở cuối.
  -h, --help    Trợ giúp này.

Ví dụ
  ./scripts/server/collect_logs.sh
  ./scripts/server/collect_logs.sh specevo_signal_10 ada_signal_10
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)     OUT_DIR="$2"; shift 2 ;;
        --no-tar)  MAKE_TAR=0; shift ;;
        -h|--help) usage; exit 0 ;;
        -*)        die "unknown option '$1' (try --help)" ;;
        *)         SESSIONS+=("$1"); shift ;;
    esac
done

command -v tmux >/dev/null 2>&1 || die "tmux không có trên máy này"
tmux has-session 2>/dev/null || tmux ls >/dev/null 2>&1 || die "không có tmux session nào đang chạy"

# Không truyền tên -> lấy hết.
if [[ ${#SESSIONS[@]} -eq 0 ]]; then
    while IFS= read -r s; do SESSIONS+=("$s"); done < <(tmux list-sessions -F '#{session_name}')
    [[ ${#SESSIONS[@]} -gt 0 ]] || die "tmux không có session nào"
fi

# --------------------------------------------------------------------------
# session -> output dir
# --------------------------------------------------------------------------
# Dòng lệnh khởi động pane do run_bench.sh dựng lên, luôn có `--output-dir X`.
extract_outdir() {
    local cmd="$1"
    [[ "$cmd" =~ --output-dir[[:space:]]+\'?([^[:space:]\'\"]+) ]] && echo "${BASH_REMATCH[1]}"
}

# Dự phòng cho session mà tmux không giữ được pane_start_command (hiếm, và với
# tmux quá cũ): đọc dòng lệnh của các tiến trình con đang chạy trong pane.
outdir_from_processes() {
    local pane_pid="$1" p out
    for p in $(pgrep -P "$pane_pid" 2>/dev/null) $pane_pid; do
        for q in $(pgrep -P "$p" 2>/dev/null) "$p"; do
            out="$(extract_outdir "$(ps -o command= -p "$q" 2>/dev/null || true)")"
            [[ -n "$out" ]] && { echo "$out"; return 0; }
        done
    done
    return 1
}

mkdir -p "$OUT_DIR"
MANIFEST="$OUT_DIR/manifest.tsv"
printf 'session\trun_id\tstatus\tsize\tlast_modified\n' > "$MANIFEST"

found=0
missing=()

for s in "${SESSIONS[@]}"; do
    outdir=""

    # 1) Session còn sống: hỏi thẳng tmux. Dòng lệnh khởi động pane do
    #    run_bench.sh dựng nên, luôn kèm --output-dir.
    line="$(tmux list-panes -t "$s" -F '#{pane_pid}|#{pane_start_command}' 2>/dev/null | head -1 || true)"
    if [[ -n "$line" ]]; then
        outdir="$(extract_outdir "${line#*|}" || true)"
        [[ -n "$outdir" ]] || outdir="$(outdir_from_processes "${line%%|*}" || true)"
    fi

    # 2) Session đã bị kill: tìm run.log tự khai tên session ở header. Chỉ có
    #    với các run bắt đầu sau khi run_bench.sh ghi dòng "tmux session:".
    if [[ -z "$outdir" ]]; then
        hit="$(grep -rl "^ tmux session: $s\$" "$REPO_ROOT"/outputs/server/*/run.log 2>/dev/null | tail -1 || true)"
        [[ -n "$hit" ]] && outdir="$(dirname "$hit")"
    fi

    if [[ -z "$outdir" ]]; then
        missing+=("$s (không có trong tmux, cũng không run.log nào ghi tên session này)")
        continue
    fi

    # run_bench.sh nhận đường dẫn tương đối so với gốc repo.
    [[ "$outdir" = /* ]] || outdir="$REPO_ROOT/$outdir"
    log="$outdir/run.log"
    if [[ ! -f "$log" ]]; then
        missing+=("$s (chưa có $log)")
        continue
    fi

    cp "$log" "$OUT_DIR/$s.log"

    run_id="$(basename "$outdir")"
    if grep -q '^ exit status :' "$log"; then
        # Footer có thể kèm chú thích ("124  (TIMED OUT after ...)"), chỉ lấy mã số.
        status="$(grep '^ exit status :' "$log" | tail -1 | sed 's/^ exit status : *//' | awk '{print $1}')"
        status="done(exit $status)"
    else
        status="running"
    fi
    size="$(du -h "$log" | cut -f1)"
    mtime="$(date -r "$log" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || stat -c %y "$log" 2>/dev/null | cut -d. -f1)"

    printf '%s\t%s\t%s\t%s\t%s\n' "$s" "$run_id" "$status" "$size" "$mtime" >> "$MANIFEST"
    printf '  ✓ %-24s <- %s  [%s, %s]\n' "$s" "$run_id" "$status" "$size"
    found=$((found + 1))
done

echo ""
echo "Đã gom $found file run.log vào: $OUT_DIR"

if [[ ${#missing[@]} -gt 0 ]]; then
    echo ""
    echo "Bỏ qua ${#missing[@]} session:"
    printf '  ✗ %s\n' "${missing[@]}"
    echo ""
    echo "  Session đã bị kill thì tmux không còn giữ đường dẫn. Tìm tay bằng thời điểm chạy:"
    echo "    ls -lt outputs/server/ | head -20"
fi

[[ "$found" -gt 0 ]] || exit 1

TARBALL=""
if [[ "$MAKE_TAR" == "1" ]]; then
    TARBALL="$OUT_DIR.tar.gz"
    tar czf "$TARBALL" -C "$(dirname "$OUT_DIR")" "$(basename "$OUT_DIR")"
    echo "Đã nén: $TARBALL ($(du -h "$TARBALL" | cut -f1))"
fi

# --------------------------------------------------------------------------
# In sẵn lệnh chạy trên MÁY CÁ NHÂN
# --------------------------------------------------------------------------
REMOTE_DIR="${OUT_DIR#"$HOME"/}"
echo ""
echo "------------------------------------------------------------"
echo "Bây giờ THOÁT server, mở terminal máy cá nhân và chạy:"
echo ""
echo "  # kéo cả thư mục (chạy lại nhiều lần chỉ tải phần mới)"
echo "  rsync -avz cpujob:~/$REMOTE_DIR/ ./logs/"
if [[ -n "$TARBALL" ]]; then
    echo ""
    echo "  # hoặc kéo file nén rồi giải"
    echo "  scp cpujob:~/${TARBALL#"$HOME"/} ~/Desktop/"
    echo "  tar xzf ~/Desktop/$(basename "$TARBALL") -C ~/Desktop/"
fi
echo ""
echo "Đổi 'cpujob' nếu host SSH của bạn tên khác."
echo "------------------------------------------------------------"
