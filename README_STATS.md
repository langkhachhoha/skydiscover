# Calculate Statistics Script

Script Python để tính toán mean, std, và best từ các file JSON chứa kết quả experiments với nhiều seeds.

## Cách sử dụng

### Cách 1: Chạy với command line argument
```bash
python calculate_stats.py <đường_dẫn_tới_file_json>
```

Ví dụ:
```bash
python calculate_stats.py math_kimi.json
python calculate_stats.py math_gpt.json
python calculate_stats.py system_kimi.json
```

### Cách 2: Chạy interactive
```bash
python calculate_stats.py
```
Sau đó nhập đường dẫn file khi được hỏi:
```
Enter path to JSON file: math_kimi.json
```

## Output

Script sẽ hiển thị:

1. **Thông tin cơ bản**: Dataset, Model, số seeds, số tasks, các methods
2. **Bảng tổng hợp**: Hiển thị Score (mean ± std) và Best cho mỗi task và method
3. **Thống kê chi tiết**: Bảng chi tiết cho từng task với cả Score và Cost

## Format dữ liệu

- **mean**: Trung bình của 3 seeds, làm tròn 4 chữ số thập phân
- **std**: Độ lệch chuẩn mẫu (sample standard deviation), làm tròn 4 chữ số thập phân  
- **best**: Giá trị tốt nhất (max) từ 3 seeds, làm tròn 4 chữ số thập phân

## Yêu cầu

- Python 3.x
- numpy

Cài đặt numpy nếu chưa có:
```bash
pip install numpy
```

## Các file JSON có sẵn

- `math_kimi.json` - Mathematical Discovery Task với KIMI K2 (7 tasks, 3 seeds)
- `math_gpt.json` - Mathematical Discovery Task với GPT-5 (7 tasks, 3 seeds)
- `system_kimi.json` - System Engineering Task với KIMI K2 (4 tasks, 3 seeds)
- `system_gpt.json` - System Engineering Task với GPT-5 (4 tasks, 3 seeds)

## Ví dụ output

```
====================================================================================================
Model: KIMI K2
====================================================================================================
Strategy                   EPLB (↑)         LLM-SQL (↑)     Transaction (↑)           PRISM (↑)
                      Avg       Best       Avg       Best       Avg       Best       Avg       Best
----------------------------------------------------------------------------------------------------
OpenEvolve        0.1339 ± 0.010     0.1453   0.6980 ± 0.005     0.7012  3966.1333 ± 368.924  4237.2000  23.4430 ± 0.922    24.4980
GEPA              0.1287 ± 0.001     0.1297   0.7043 ± 0.003     0.7071  3748.1000 ± 126.677  3891.0000  23.0493 ± 0.932    24.1200
...
```
