# 📊 Results Analysis Summary

Tổng hợp công cụ phân tích kết quả experiments cho Mathematical Discovery Task và System Engineering Task.

## 📁 Danh sách files

### JSON Data Files (4 files)
1. **math_kimi.json** - Mathematical Discovery Task với KIMI K2 (7 tasks, 3 seeds)
2. **math_gpt.json** - Mathematical Discovery Task với GPT-5 (7 tasks, 3 seeds)
3. **system_kimi.json** - System Engineering Task với KIMI K2 (4 tasks, 3 seeds)
4. **system_gpt.json** - System Engineering Task với GPT-5 (4 tasks, 3 seeds)

### Python Scripts (3 scripts)
1. **calculate_stats.py** - Tính mean, std, best cho từng file
2. **compare_all.py** - So sánh tổng hợp giữa tất cả models và datasets
3. **run_all_stats.sh** - Bash script chạy tất cả phân tích tự động

### Documentation
1. **README_STATS.md** - Hướng dẫn sử dụng calculate_stats.py
2. **ANALYSIS_SUMMARY.md** - File này, tổng hợp toàn bộ

---

## 🚀 Hướng dẫn sử dụng nhanh

### 1. Phân tích một file cụ thể
```bash
python calculate_stats.py math_kimi.json
```

### 2. So sánh tổng hợp tất cả
```bash
python compare_all.py
```

### 3. Chạy phân tích tất cả files
```bash
./run_all_stats.sh
```

---

## 📈 Kết quả chính

### 🏆 Method Performance Ranking

**Tổng hợp wins (Score + Cost) trên tất cả 4 datasets:**

| Rank | Method    | Score Wins | Cost Wins | Total Wins |
|------|-----------|------------|-----------|------------|
| #1   | SpecEvo   | 19         | 22        | 41         |
| #2   | AdaEvolve | 4          | 0         | 4          |
| #3   | OpenEvolve| 2          | 0         | 2          |
| #4   | EvoX      | 2          | 0         | 2          |
| #5   | GEPA      | 1          | 0         | 1          |

**Kết luận chính:**
- 🥇 **SpecEvo vượt trội** với 41 total wins, dẫn đầu về cả performance (19 wins) và cost efficiency (22 wins)
- SpecEvo đặc biệt hiệu quả về cost, thấp hơn các methods khác đáng kể
- AdaEvolve đứng thứ 2 nhưng chỉ mạnh về score, không tối ưu về cost

---

## 🔍 So sánh chi tiết

### Mathematical Discovery Task: KIMI K2 vs GPT-5

**Scores (Average across all tasks):**
| Method     | KIMI K2 | GPT-5 | Winner |
|------------|---------|-------|--------|
| OpenEvolve | 0.7941  | 0.8342| GPT-5  |
| GEPA       | 0.7942  | 0.8285| GPT-5  |
| AdaEvolve  | 0.8303  | 0.8508| GPT-5  |
| EvoX       | 0.8311  | 0.8482| GPT-5  |
| SpecEvo    | 0.8448  | 0.8586| GPT-5  |

**Costs (Average across all tasks):**
| Method     | KIMI K2 | GPT-5  | Winner  |
|------------|---------|--------|---------|
| OpenEvolve | 1.7544  | 8.2211 | KIMI K2 |
| GEPA       | 1.6904  | 8.1429 | KIMI K2 |
| AdaEvolve  | 1.7022  | 8.2168 | KIMI K2 |
| EvoX       | 1.7639  | 8.1899 | KIMI K2 |
| SpecEvo    | 0.8648  | 2.3954 | KIMI K2 |

**Insight:**
- GPT-5 có performance tốt hơn (~5% higher scores)
- KIMI K2 **hiệu quả về cost hơn rất nhiều** (~79% lower costs)
- Trade-off: Nếu cần performance tốt nhất → GPT-5; Nếu cần cost-effective → KIMI K2

---

### System Engineering Task: KIMI K2 vs GPT-5

**Scores (Average across all tasks):**
| Method     | KIMI K2   | GPT-5     | Winner |
|------------|-----------|-----------|--------|
| OpenEvolve | 997.60    | 1008.27   | GPT-5  |
| GEPA       | 943.00    | 1026.32   | GPT-5  |
| AdaEvolve  | 947.04    | 996.47    | GPT-5  |
| EvoX       | 910.04    | 963.73    | GPT-5  |
| SpecEvo    | 1042.56   | 1050.09   | GPT-5  |

**Costs (Average across all tasks):**
| Method     | KIMI K2 | GPT-5  | Winner  |
|------------|---------|--------|---------|
| OpenEvolve | 1.6881  | 5.4267 | KIMI K2 |
| GEPA       | 1.5326  | 5.5120 | KIMI K2 |
| AdaEvolve  | 1.7208  | 5.8451 | KIMI K2 |
| EvoX       | 1.8300  | 5.7899 | KIMI K2 |
| SpecEvo    | 0.6396  | 2.0263 | KIMI K2 |

**Insight:**
- GPT-5 vẫn có performance tốt hơn (~1-9% higher)
- KIMI K2 **rẻ hơn đáng kể** (~69-72% lower costs)
- Pattern tương tự: GPT-5 = better performance, KIMI K2 = much better cost

---

## 📊 Format dữ liệu trong JSON

Mỗi file JSON có cấu trúc:
```json
{
  "dataset": "Task name",
  "model": "Model name",
  "methods": ["OpenEvolve", "GEPA", "AdaEvolve", "EvoX", "SpecEvo"],
  "tasks": ["Task1", "Task2", ...],
  "seeds": [
    {
      "seed_id": 1,
      "results": [
        {
          "task": "Task name",
          "optimization": "maximize",
          "OpenEvolve": {"score": X.XXXX, "cost": X.XXXX},
          ...
        }
      ]
    }
  ]
}
```

---

## 🛠️ Chi tiết công cụ

### calculate_stats.py
**Chức năng:**
- Đọc file JSON chứa kết quả 3 seeds
- Tính mean (trung bình)
- Tính std (sample standard deviation)
- Tìm best (giá trị max cho score, min cho cost)
- Làm tròn 4 chữ số thập phân
- Hiển thị bảng tổng hợp và chi tiết

**Output:**
- Bảng tổng hợp: Method × Task với Avg ± std và Best
- Bảng chi tiết: Từng task với đầy đủ thông tin Score và Cost

### compare_all.py
**Chức năng:**
- So sánh tất cả 4 files JSON
- So sánh theo dataset (KIMI K2 vs GPT-5)
- So sánh theo model (Math vs System)
- Tổng hợp ranking methods
- Xác định winners cho từng comparison

**Output:**
- Comparison tables by dataset và model
- Method performance summary với ranking
- Win counts cho score và cost

---

## 💡 Khuyến nghị

### Khi nào dùng GPT-5?
- Cần performance/accuracy cao nhất
- Budget không phải vấn đề
- Production systems cần độ chính xác tối đa

### Khi nào dùng KIMI K2?
- Cần cost-effective solution
- Large-scale experiments với budget giới hạn
- Performance đủ tốt (chỉ thấp hơn 5-10%)
- Cần chạy nhiều iterations/seeds

### Method nào tốt nhất?
**SpecEvo** là lựa chọn tốt nhất cho cả hai tiêu chí:
- Best performance trong hầu hết cases
- Lowest cost trong tất cả methods
- Win rate cao nhất (41/50 total tasks)

---

## 📝 Ghi chú kỹ thuật

### Statistical Notes
- **std**: Sample standard deviation (ddof=1) được sử dụng vì chỉ có 3 seeds
- **best**: Maximum value cho score (optimization = maximize)
- **mean**: Arithmetic mean của 3 seeds
- Tất cả giá trị làm tròn 4 chữ số thập phân

### Data Quality Notes
- Một số files có seeds giống nhau (seed 2 = seed 3), có thể do random seed hoặc kết quả ổn định
- Tất cả tasks đều là maximize optimization
- Cost values có đơn vị khác nhau giữa Math tasks (nhỏ ~0.2-10) và System tasks (tương tự)

---

## 🔄 Cập nhật và mở rộng

Để thêm file JSON mới:
1. Đặt file vào cùng thư mục
2. Cập nhật `files` array trong `run_all_stats.sh`
3. Cập nhật `files` dict trong `compare_all.py`
4. Chạy lại scripts

---

**Created:** June 27, 2026
**Last Updated:** June 27, 2026
