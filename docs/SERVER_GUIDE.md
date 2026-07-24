# Hướng dẫn chạy thực nghiệm trên server (SSH + tmux + conda)

Tài liệu này dành cho người **chưa từng dùng SSH / tmux / server**. Cứ làm theo
từng bước, mỗi lệnh copy-paste được luôn.

Quy trình làm việc tổng thể:

```
Máy của bạn (code + sửa code)  →  git push  →  GitHub  →  git pull  →  Server (chỉ chạy)
```

Trên server bạn **không sửa code**, chỉ `git pull` rồi chạy script.

---

## Mục lục

1. [Khái niệm 30 giây: SSH, tmux, conda](#1-khái-niệm-30-giây)
2. [Đăng nhập vào server bằng SSH](#2-đăng-nhập-vào-server-bằng-ssh)
3. [Lấy code về server](#3-lấy-code-về-server)
4. [Tạo file `.env` (chứa API key)](#4-tạo-file-env-chứa-api-key)
5. [Cài môi trường conda tên `minhhieu`](#5-cài-môi-trường-conda-tên-minhhieu)
6. [Chạy benchmark](#6-chạy-benchmark)
7. [tmux — chạy rồi tắt máy vẫn không sao](#7-tmux--chạy-rồi-tắt-máy-vẫn-không-sao)
8. [Xem log / xem kết quả](#8-xem-log--xem-kết-quả)
9. [Lấy kết quả từ server về máy](#9-lấy-kết-quả-từ-server-về-máy)
10. [Bảng tra tham số đầy đủ](#10-bảng-tra-tham-số-đầy-đủ)
11. [Sự cố thường gặp](#11-sự-cố-thường-gặp)

---

## 1. Khái niệm 30 giây

| Thứ | Là gì | Vì sao cần |
|---|---|---|
| **SSH** | Lệnh để "vào" một máy tính khác qua mạng và gõ terminal trên máy đó | Server không có màn hình, chỉ vào được bằng SSH |
| **tmux** | Một "màn hình ảo" sống **bên trong server** | Khi bạn tắt laptop / rớt mạng, tiến trình chạy trong tmux **vẫn tiếp tục**. Nếu chạy trực tiếp, tắt SSH là job **chết** |
| **conda** | Trình quản lý môi trường Python | Cài thư viện vào một "hộp" riêng tên `minhhieu`, không đụng vào Python hệ thống |

Ý quan trọng nhất: **luôn chạy job dài trong tmux.**

---

## 2. Đăng nhập vào server bằng SSH

Trên terminal **máy của bạn** (Mac: mở app Terminal):

```bash
ssh <username>@<địa-chỉ-server>
```

Ví dụ: `ssh minhhieu@172.26.92.15` hoặc `ssh minhhieu@server.comp.nus.edu.sg`

- Lần đầu nó hỏi `Are you sure you want to continue connecting (yes/no)?` → gõ `yes` rồi Enter.
- Sau đó nó hỏi password → gõ password (**màn hình sẽ không hiện gì cả, kể cả dấu \***, đó là bình thường) rồi Enter.

Khi thấy dấu nhắc kiểu `minhhieu@server:~$` là bạn đang **ở trong server**. Từ giờ mọi lệnh gõ ra là chạy trên server.

Thoát ra: gõ `exit` hoặc bấm `Ctrl-D`.

### (Tuỳ chọn nhưng rất nên làm) Khỏi phải gõ password mỗi lần

Trên **máy của bạn**:

```bash
ssh-keygen -t ed25519            # Enter 3 lần, để mặc định hết
ssh-copy-id <username>@<địa-chỉ-server>   # gõ password lần cuối cùng
```

Từ lần sau `ssh <username>@<server>` sẽ vào thẳng, không hỏi password.

---

## 3. Lấy code về server

Sau khi đã SSH vào server:

```bash
# Kiểm tra có git chưa
git --version

# Clone repo (lần đầu tiên duy nhất)
cd ~
git clone --recurse-submodules <URL-repo-của-bạn> skydiscover
cd skydiscover
```

> `--recurse-submodules` là cần thiết vì repo có submodule (ALE-Bench).

**Những lần sau**, mỗi khi bạn đã push code mới từ máy mình:

```bash
cd ~/skydiscover
git pull
```

Nếu `git pull` báo lỗi vì server có thay đổi lặt vặt (file `.pyc`, output cũ...):

```bash
git checkout -- .      # bỏ mọi thay đổi cục bộ trên server
git pull
```

---

## 4. Tạo file `.env` (chứa API key)

File `.env` nằm trong `.gitignore` nên **không bao giờ được push lên GitHub**
(đó là chủ ý — API key không nên nằm trên GitHub). Vì vậy sau khi `git clone`,
trên server sẽ **không có** file này và bạn phải tự tạo nó bằng tay.

### Cách 1 (khuyến nghị): gõ trực tiếp trên server

```bash
cd ~/skydiscover
cat > .env <<'EOF'
OPENAI_API_KEY=sk-or-v1-....................
EOF
chmod 600 .env        # chỉ mình bạn đọc được file này
```

Thay `sk-or-v1-...` bằng key OpenRouter thật của bạn (lấy trên
https://openrouter.ai/keys). Lưu ý:

- Dùng `<<'EOF'` có dấu nháy đơn để shell không "ăn" ký tự đặc biệt trong key.
- Tên biến là `OPENAI_API_KEY`; script cũng chấp nhận `OPENROUTER_API_KEY`.
  Key bắt đầu bằng `sk-or-` được repo tự hiểu là key OpenRouter.

### Cách 2: copy file `.env` từ máy bạn sang server

Chạy lệnh này trên **máy của bạn** (KHÔNG phải trên server):

```bash
scp /Users/apple/Desktop/All/NUS_INTERNSHIP/skydiscover/.env \
    <username>@<địa-chỉ-server>:~/skydiscover/.env
```

### Kiểm tra key hoạt động

Trên server, sau khi đã cài môi trường ở bước 5:

```bash
cd ~/skydiscover
conda activate minhhieu
python scripts/test_openrouter_key.py
```

Thấy dòng `OpenRouter API key hoat dong binh thuong.` là ổn.

---

## 5. Cài môi trường conda tên `minhhieu`

### 5a. Nếu server chưa có conda

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
exec $SHELL -l          # nạp lại shell để lệnh `conda` có hiệu lực
conda --version         # phải in ra số phiên bản
```

> Nếu server không có `wget`, thay bằng
> `curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`.
> Nếu server là máy ARM (hiếm), đổi `x86_64` thành `aarch64`.

### 5b. Tạo môi trường (chạy 1 lệnh là xong)

```bash
cd ~/skydiscover
bash scripts/server/setup_env.sh
```

Script này sẽ: tạo env `minhhieu` với Python 3.12 → cài `skydiscover` (chế độ
editable) → cài các thư viện trong `scripts/server/requirements-server.txt`
(bao gồm toàn bộ dependency của `levi/`, tức phần BLADE) → cuối cùng **tự kiểm
tra import** và in `All core imports OK (skydiscover + levi).`

Mất khoảng 3–8 phút. Sau đó:

```bash
conda activate minhhieu
```

### 5c. Nếu bạn muốn tự gõ tay thay vì dùng script

Đây chính xác là những gì `setup_env.sh` làm:

```bash
conda create -y -n minhhieu python=3.12 pip
conda activate minhhieu
cd ~/skydiscover
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r scripts/server/requirements-server.txt
```

Python **3.12** là bắt buộc: `skydiscover` cần `>=3.10,<3.14`, còn `levi` cần
`>=3.11,<3.13` — chỉ 3.11/3.12 thoả cả hai, và 3.12 là bản đã kiểm chứng.

### 5d. Các benchmark "nặng" (chỉ cài khi cần)

Mặc định script **không** cài `torch`, `jax`, `triton` (rất nặng, vài GB). Chỉ
cài thêm nếu bạn chạy đúng nhóm benchmark đó:

```bash
bash scripts/server/setup_env.sh --extra math    # jax, optax, torch, numba, cvxpy, pymoo...
bash scripts/server/setup_env.sh --extra adrs    # torch, pandas, networkx<3.4
bash scripts/server/setup_env.sh --extra torch   # chỉ torch (ADRS/eplb, kernelbench)
```

Với các benchmark thông thường (`benchmarks/math/*`, `benchmarks/co_bench/*`),
script chạy benchmark sẽ **tự cài** `requirements.txt` riêng của benchmark đó
mỗi lần chạy — bạn không phải làm gì cả.

Các lệnh conda hữu ích khác:

```bash
conda env list                              # xem các env đang có
bash scripts/server/setup_env.sh --recreate # xoá và tạo lại env từ đầu
conda deactivate                            # thoát env
```

### 5f. Kiểm tra toàn bộ setup (miễn phí, không gọi API)

```bash
bash scripts/server/selftest.sh
```

Script này kiểm tra: conda/tmux/git đã có chưa, env `minhhieu` đúng Python
3.12 chưa, import được `skydiscover` + `levi` chưa, `.env` có key hợp lệ
chưa (và có bị để chế độ ai cũng đọc được không), repo có đủ file không,
`run_bench.sh` bắt lỗi tham số sai đúng không, và **chạy thử 2 tmux session
nháp** để chứng minh cơ chế chạy nền hoạt động. Không tốn một xu nào.

Cuối cùng in ra `PASS: N  FAIL: 0` là mọi thứ sẵn sàng. Có `FAIL` thì nó nói
luôn cần sửa gì. Chạy lại script này bất cứ lúc nào thấy có gì đó bất thường.

### 5e. Cài tmux (nếu server chưa có)

```bash
tmux -V                     # nếu in ra "tmux 3.x" là đã có, bỏ qua bước này
sudo apt-get install -y tmux                # nếu bạn có quyền sudo
conda install -y -c conda-forge tmux        # nếu KHÔNG có quyền sudo
```

---

## 6. Chạy benchmark

Tất cả chạy qua **một file duy nhất**: `scripts/server/run_bench.sh`.

Script này sao chép **nguyên xi** hai workflow GitHub Actions
(`.github/workflows/blade.yml` và `.github/workflows/baseline.yml`): cùng biến
môi trường, cùng bước cài dependency, cùng bước tải dataset, cùng dòng lệnh
cuối cùng, cùng giá trị mặc định. Mọi tham số bạn từng điền vào form "Run
workflow" trên GitHub đều có một cờ (flag) tương ứng ở đây, **cùng tên**.

### Xem toàn bộ tham số

```bash
./scripts/server/run_bench.sh --help
```

### Chạy BLADE (phương pháp của bạn)

```bash
cd ~/skydiscover
./scripts/server/run_bench.sh blade --tmux \
    --benchmark levi/examples/circle_packing \
    --evaluations 200 \
    --seed 1
```

### Chạy baseline (để so sánh)

```bash
./scripts/server/run_bench.sh baseline --tmux \
    --baseline openevolve_native \
    --benchmark-dir benchmarks/math/circle_packing \
    --iterations 100 \
    --model openrouter/openai/gpt-5 \
    --seed 1
```

### Thay đổi tham số — hoàn toàn tự do

Đổi benchmark, đổi model, đổi ablation, đổi ngân sách… chỉ cần thêm/bớt cờ:

```bash
# CO-Bench, giới hạn chi phí $5, dùng nhiều worker hơn
./scripts/server/run_bench.sh blade --tmux \
    --benchmark levi/examples/co_bench/tsp \
    --dollars 5 --workers 8 --cobench-timeout 10 --cobench-max-instances 0

# Chạy ablation A7 (bỏ crossover), seed 3
./scripts/server/run_bench.sh blade --tmux \
    --benchmark levi/examples/heilbronn_triangle \
    --evaluations 300 --ablation no_crossover --seed 3

# Chỉnh các knob cấp thấp bằng JSON (giống ô advanced_options trên GitHub)
./scripts/server/run_bench.sh blade --tmux \
    --benchmark levi/examples/circle_packing_rect --evaluations 300 \
    --advanced-options '{"n_cells":50,"p_targeted_mutate":0.5,"meta_advice_inject_p":0.35}'

# Baseline khác trên benchmark khác
./scripts/server/run_bench.sh baseline --tmux \
    --baseline evox --benchmark-dir benchmarks/co_bench/bin_packing_1d --iterations 150
```

### Giới hạn thời gian: `--timeout`

Trên GitHub Actions, baseline bị chặn cứng ở 3 tiếng (`timeout-minutes: 180`).
Server của bạn không có giới hạn đó, nên `run_bench.sh` **giữ nguyên mặc định
3 tiếng cho baseline** để kết quả so sánh được với các số bạn đã thu trước đây.

```bash
--timeout 3h        # mặc định của baseline (= 10800 giây)
--timeout 6h        # cho chạy lâu hơn
--timeout 0         # BỎ giới hạn — tận dụng server không bị cap
```

Chế độ `blade` mặc định **không** có timeout (đúng như `blade.yml`), vì BLADE đã
có sẵn ngân sách mềm `--seconds` / `--dollars` / `--evaluations`.

Khác biệt giữa hai loại giới hạn:

| Tiêu chí | `--seconds` (blade) | `--timeout` (cả hai) |
| --- | --- | --- |
| Cách dừng | BLADE tự dừng **êm**, ghi đầy đủ kết quả | Giết tiến trình bằng SIGTERM/SIGKILL |
| Exit status | 0 | **124** |
| Dùng khi | muốn kết thúc gọn gàng | muốn chặn cứng, phòng job treo |

Khi bị timeout, **kết quả dở dang vẫn được giữ nguyên** (log, cost_log,
checkpoint đã ghi tới thời điểm đó), và dòng cuối log ghi rõ:
`exit status : 124  (TIMED OUT after 10800s — partial results kept)`.

### Thử trước khi chạy thật: `--dry-run`

**Rất nên dùng.** Nó in ra đúng dòng lệnh sẽ chạy rồi dừng lại, **không gọi API,
không tốn tiền**:

```bash
./scripts/server/run_bench.sh blade --dry-run \
    --benchmark levi/examples/circle_packing --evaluations 200 --ablation emb_only
```

Đọc dòng `>>> command:` để chắc chắn mọi tham số đúng ý, rồi bỏ `--dry-run`,
thêm `--tmux` và chạy thật.

### Chạy nhiều seed liên tiếp

```bash
for s in 1 2 3; do
  ./scripts/server/run_bench.sh blade --tmux --session blade_cp_seed$s \
      --benchmark levi/examples/circle_packing --evaluations 200 --seed $s
done
```

Ba job chạy **song song** trong ba tmux session. Nếu server yếu, giảm
`--workers` xuống, hoặc chạy tuần tự bằng cách bỏ `--tmux` trong vòng lặp và
bọc cả vòng lặp trong **một** tmux session.

---

## 7. tmux — chạy rồi tắt máy vẫn không sao

Khi bạn thêm cờ `--tmux`, script tự tạo một tmux session trên server rồi chạy
job trong đó, và in ra màn hình:

```
Started in tmux.
  session     : blade_levi_examples_circle_packing_full_seed1_20260723-180513
  output dir  : outputs/server/blade_levi_examples_circle_packing_full_seed1_20260723-180513
  log file    : outputs/server/.../run.log
```

Sau đó bạn có thể **thoát SSH, đóng laptop, mất mạng** — job vẫn chạy tiếp trên
server. Lần sau SSH vào lại, nó vẫn ở đó.

### Các lệnh tmux cần nhớ (chỉ 5 lệnh)

```bash
tmux ls                          # liệt kê các session đang chạy
tmux attach -t <tên-session>     # "chui vào" xem job đang chạy
                                 #   thoát ra: bấm Ctrl-b rồi thả tay, bấm d
tmux kill-session -t <tên>       # dừng hẳn một job
tmux kill-server                 # dừng TẤT CẢ (cẩn thận!)
```

> **Ctrl-b rồi d** là cách thoát đúng ("detach"). Nếu bạn bấm `Ctrl-c` thì bạn
> **giết job** chứ không phải thoát ra. Đây là lỗi phổ biến nhất của người mới.

Đặt tên session cho dễ nhớ bằng `--session`:

```bash
./scripts/server/run_bench.sh blade --tmux --session cp_seed1 --benchmark levi/examples/circle_packing
tmux attach -t cp_seed1
```

Khi job chạy xong, tmux session **vẫn còn** và hiện dòng
`=== run finished (exit 0) — press Enter to close this pane ===`. Đó là cố ý —
để bạn attach vào đọc kết quả. Số trong ngoặc là mã thoát: **0 = chạy trọn vẹn**,
khác 0 = job lỗi. Dọn session bằng cách attach vào rồi bấm **Enter**, hoặc
`tmux kill-session -t <tên>`.

> Lưu ý: khi đang attach mà bấm Enter thì pane đóng và session biến mất. Job
> lúc đó đã xong rồi nên không mất gì, nhưng nếu muốn giữ session lại để xem
> tiếp thì thoát bằng **Ctrl-b rồi d** thay vì Enter.

### Chạy không dùng tmux

Bỏ cờ `--tmux` thì job chạy ngay trước mắt bạn (foreground). Chỉ nên làm vậy
với `--dry-run` hoặc job rất ngắn, vì **đóng SSH là job chết**.

---

## 8. Xem log / xem kết quả

Mọi thứ nằm trong `outputs/server/<run-id>/`, trong đó `<run-id>` có dạng
`<mode>_<benchmark>_<ablation-hoặc-baseline>_seed<N>_<ngày-giờ>`.

> Nếu bạn phóng hai job **y hệt nhau trong cùng một giây**, run-id thứ hai sẽ
> tự thêm hậu tố `_2`, `_3`… nên hai job không bao giờ ghi đè hay trộn log vào
> nhau.

### Xem log đang chạy theo thời gian thực

```bash
tail -f outputs/server/<run-id>/run.log
```

Thoát khỏi `tail -f` bằng `Ctrl-c` (an toàn — nó chỉ dừng việc *xem*, job vẫn chạy).

Không nhớ tên run? Lấy cái mới nhất:

```bash
ls -t outputs/server/ | head -5                       # 5 run gần nhất
tail -f "outputs/server/$(ls -t outputs/server | head -1)/run.log"
```

### Các cách xem khác

```bash
less outputs/server/<run-id>/run.log      # cuộn xem cả file (thoát: q)
grep "Best" outputs/server/<run-id>/run.log        # lọc dòng có điểm tốt nhất
grep -E "Status|Best|cost" outputs/server/<run-id>/run.log | tail -20
tmux attach -t <session>                  # xem trực tiếp trong tmux
```

### Chi phí đã tiêu

```bash
cat outputs/server/<run-id>/cost_log.totals.json      # tổng USD, số token
wc -l outputs/server/<run-id>/cost_log.jsonl          # số lần gọi LLM
```

Phần cuối `run.log` cũng in sẵn bảng tổng kết chi phí + `exit status`
(`exit status : 0` = chạy xong bình thường; khác 0 = có lỗi).

### Kết quả chương trình tốt nhất

- Baseline: `outputs/server/<run-id>/run/best/` và `run/checkpoints/`
- BLADE: các file kết quả nằm thẳng trong `outputs/server/<run-id>/`

### Lấy nhanh điểm cuối cùng: `scripts/server/result.sh`

Thay cho việc `tail -f` rồi tự Ctrl-F tìm "New best", dùng script này. Nó tự
nhận biết log là BLADE hay baseline và làm đúng việc:

```bash
./scripts/server/result.sh                      # run mới nhất, tự động
./scripts/server/result.sh <đường-dẫn-run.log>  # chỉ định log cụ thể
./scripts/server/result.sh --list               # liệt kê các run gần đây để chọn
```

**Với baseline / CO-Bench** — nó tìm dòng `New best` **cuối cùng** và in kèm
mấy dòng phía trên (chính là dòng `Dev Score | Test Score | Overall`). Dòng có
dấu `>>` là dòng khớp keyword, các dòng còn lại là ngữ cảnh:

```bash
./scripts/server/result.sh                  # mặc định: New best cuối + 3 dòng trên
./scripts/server/result.sh -B 1 -A 0        # chỉ 1 dòng trên (đúng dòng test score)
./scripts/server/result.sh -n 3             # 3 lần New best cuối, không chỉ 1
./scripts/server/result.sh -n all           # tất cả các lần New best
```

**Với BLADE** — nó in **cả khối báo cáo cuối** (`Best score`, `Total cost`,
`Runtime`…) từ đầu báo cáo tới hết file, nên không bao giờ bị cắt cụt như
`tail -N`:

```bash
./scripts/server/result.sh path/to/blade/run.log   # tự nhận ra là BLADE
```

**Tìm keyword bất kỳ (linh hoạt hoàn toàn):** hai cờ `-B` (before, số dòng
phía trên) và `-A` (after, phía dưới) giống hệt `grep`, gõ dính hay tách đều
được (`-B3` hoặc `-B 3`):

```bash
./scripts/server/result.sh -k "Test Score" -B0 -A0      # chỉ các dòng Test Score
./scripts/server/result.sh -k "🌟" -B2                   # emoji cũng được
./scripts/server/result.sh -k "combined_score" -n all   # mọi lần xuất hiện
./scripts/server/result.sh --from "Runtime"             # in từ dòng 'Runtime' cuối tới hết
```

Các cờ chính:

| Cờ | Ý nghĩa | Mặc định |
| --- | --- | --- |
| `-k, --keyword RE` | Từ khoá cần tìm (regex) | tự nhận: `New best` cho baseline |
| `-B, --before N` | Số dòng in phía **trên** mỗi lần khớp | `3` |
| `-A, --after N` | Số dòng in phía **dưới** mỗi lần khớp | `1` |
| `-n, --num N` | Chỉ lấy N lần khớp **cuối** (`all` = tất cả) | `1` |
| `--from RE` | In từ dòng cuối khớp RE tới hết file (tail linh hoạt) | — |
| `--report` | Ép chế độ báo cáo BLADE (= `--from 'Best score'`) | — |
| `-f, --follow` | In xong rồi `tail -f` theo dõi tiếp | tắt |
| `-l, --list` | Liệt kê các run gần đây rồi thoát | — |

---

## 9. Lấy kết quả từ server về máy

Chạy trên **máy của bạn** (không phải trên server):

```bash
# Kéo về một run cụ thể
scp -r <username>@<server>:~/skydiscover/outputs/server/<run-id> ./ket_qua/

# Kéo về toàn bộ (dùng rsync, chỉ tải phần thay đổi — chạy lại nhiều lần rất nhanh)
rsync -avz <username>@<server>:~/skydiscover/outputs/server/ ./outputs/server/
```

Sau đó vẽ hình bằng các script sẵn có trên máy bạn (xem `scripts/plots/`).

---

## 10. Bảng tra tham số đầy đủ

### Tham số dùng chung

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--seed N` | `1` | Nhãn seed; chỉ đặt tên run/thư mục, **không** đổi cách chạy (giống hệt trên GitHub Actions) |
| `--cobench-timeout N` | `10` | CO-Bench: giới hạn giây cho mỗi instance |
| `--cobench-max-cases N` | tất cả | CO-Bench: số file test-case tối đa |
| `--cobench-max-instances N` | `3` | CO-Bench: số instance mỗi file (`0` = toàn bộ test set) |
| `--timeout SPEC` | `3h` cho baseline, **không giới hạn** cho blade | Trần thời gian thực. Hết giờ thì job bị SIGTERM (30 giây sau là SIGKILL) và exit status thành **124**. Nhận `10800`, `600s`, `180m`, `3h`; `0` hoặc `none` để tắt |
| `--tmux` | tắt | Chạy nền trong tmux |
| `--session NAME` | tự sinh | Tên tmux session |
| `--output-dir DIR` | `outputs/server/<run-id>` | Nơi lưu kết quả |
| `--run-id NAME` | tự sinh | Đặt tên run theo ý bạn |
| `--conda-env NAME` | `minhhieu` | Env conda cần kích hoạt |
| `--no-conda` | – | Không đụng vào conda, dùng python đang active |
| `--no-install-deps` | – | Bỏ qua bước tự cài dependency của benchmark (nhanh hơn khi chạy lại) |
| `--dry-run` | – | Chỉ in lệnh, không chạy, không tốn tiền |

### Tham số riêng của `blade` (khớp `blade.yml`)

| Cờ | Mặc định |
|---|---|
| `--benchmark PATH` | `levi/examples/circle_packing` |
| `--evaluations N` | rỗng (không giới hạn) |
| `--dollars N` | rỗng (tắt) |
| `--seconds N` | rỗng (tắt) |
| `--mutation-model ID` | `openrouter/qwen/qwen3-30b-a3b-instruct-2507` |
| `--paradigm-model ID` | `openrouter/openai/gpt-5` |
| `--workers N` | `4` |
| `--pe-interval N` | `10` |
| `--eval-timeout N` | `600` |
| `--n-diverse-seeds N` | `5` |
| `--n-variants-per-seed N` | `20` |
| `--ablation NAME` | `full` |
| `--advanced-options JSON` | rỗng |

**Các giá trị `--ablation` hợp lệ:** `full`, `ast_only`, `emb_only`,
`static_cells`, `no_meta_advice`, `meta_errors_only`, `no_targeted_mutate`,
`no_crossover`, `no_paradigm`.

**Các khoá dùng được trong `--advanced-options`:** `problem_module`,
`target_score`, `embedding_model`, `eval_processes`, `n_paradigm_variants`,
`paradigm_n_anchors`, `paradigm_n_inspirations`, `n_cells`, `recluster_every`,
`embedding_dim`, `meta_advice_disabled`, `meta_advice_interval`,
`meta_advice_mode` (`rich`|`errors_only`), `meta_advice_inject_p`,
`targeted_mutate_disabled`, `analyzer_interval`, `analyzer_top_k`,
`p_targeted_mutate`, `p_crossover`, `paradigm_synthesis_max_stagnation`,
`paradigm_surgical_max_stagnation`, `paradigm_synthesis_n_anchors`,
`paradigm_shift_n_anchors`, `paradigm_surgical_n_inspirations`.

Xem danh sách benchmark có sẵn: `ls levi/examples/` và `ls levi/examples/co_bench/`.

### Tham số riêng của `baseline` (khớp `baseline.yml`)

| Cờ | Mặc định |
|---|---|
| `--baseline NAME` | `evox` — chọn trong `openevolve_native`, `gepa_native`, `adaevolve`, `evox` |
| `--benchmark-dir DIR` | `benchmarks/math/circle_packing` |
| `--iterations N` | `100` |
| `--model ID` | `openrouter/openai/gpt-5` |

Xem danh sách benchmark: `ls benchmarks/math/` và `ls benchmarks/co_bench/`.

---

## 11. Sự cố thường gặp

**`conda: command not found`**
→ Chưa cài conda hoặc chưa nạp lại shell. Làm lại bước 5a, nhớ `exec $SHELL -l`.

**`could not activate conda env 'minhhieu'`**
→ Chưa tạo env. Chạy `bash scripts/server/setup_env.sh`.

**`ERROR: no API key found`**
→ Chưa có file `.env`, hoặc file rỗng, hoặc bạn đang đứng sai thư mục. Kiểm tra:
```bash
cd ~/skydiscover && ls -la .env && head -c 20 .env
```

**Không chắc lỗi ở đâu**
→ Chạy `bash scripts/server/selftest.sh` trước tiên. Nó kiểm tra toàn bộ chuỗi
setup (conda → thư viện → `.env` → tmux) và chỉ đúng chỗ hỏng, hoàn toàn miễn phí.

**`tmux: command not found`**
→ Cài tmux theo bước 5e.

**`a tmux session named '...' already exists`**
→ Session cũ còn sót. Xem bằng `tmux ls`, rồi hoặc `tmux kill-session -t <tên>`,
hoặc chạy lại với `--session <tên-khác>`.

**Đóng SSH xong quay lại thì job biến mất**
→ Bạn quên cờ `--tmux`. Không có nó, job chết theo phiên SSH.

**Nhỡ bấm `Ctrl-c` trong lúc attach vào tmux**
→ Job đã bị giết. Chạy lại. Lần sau thoát bằng **Ctrl-b rồi d**.

**`ModuleNotFoundError` khi chạy một benchmark cụ thể**
→ Benchmark đó cần thư viện nặng. Chạy
`bash scripts/server/setup_env.sh --extra math` (hoặc `--extra adrs` / `--extra torch`),
hoặc kiểm tra chắc chắn bạn **không** truyền `--no-install-deps`.

**`git pull` báo conflict**
→
```bash
cd ~/skydiscover && git checkout -- . && git pull
```
(An toàn vì server không phải nơi bạn sửa code.)

**Server chậm / hết RAM khi chạy nhiều job**
→ Giảm `--workers` và `advanced_options.eval_processes`. Xem tài nguyên bằng
`htop` (hoặc `top`), xem dung lượng đĩa bằng `df -h`.

**Muốn biết job có còn chạy không**
```bash
tmux ls                                   # session còn sống?
tail -3 outputs/server/<run-id>/run.log   # log có tiến triển không?
```
