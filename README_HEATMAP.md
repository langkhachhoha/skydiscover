# 📊 Cost Heatmap Visualization

Script Python để tạo heatmap thống kê chi phí đẹp mắt cho experiments.

## 🎯 Mô tả

Script này tạo ra một **heatmap visualization** chuyên nghiệp để so sánh chi phí giữa:
- **5 Methods**: OpenEvolve, GEPA, AdaEvolve, EvoX, SpecEvo
- **4 Datasets**: Math (GPT-5), Math (KIMI K2), System (GPT-5), System (KIMI K2)

### ✨ Đặc điểm nổi bật

1. **Màu sắc trực quan**
   - Gradient từ Đỏ (chi phí cao) → Xanh lá (chi phí thấp)
   - Dễ nhận biết ngay performance

2. **Thông tin chi tiết cho SpecEvo**
   - **Tổng chi phí** (số chính, to nhất)
   - **Reduction factor** (giảm bao nhiêu lần so với avg các methods khác)
   - **Per-task reduction** (giảm bao nhiêu lần trên mỗi task)

3. **Thiết kế chuyên nghiệp**
   - Số to, rõ ràng, dễ đọc
   - Grid lines đậm nét
   - SpecEvo row được highlight với viền vàng
   - Colorbar với chú thích
   - Thông điệp tổng kết ở footer

## 🚀 Cách sử dụng

### Yêu cầu
- Python 3.x
- matplotlib
- numpy

### Chạy script
```bash
python plot_cost_heatmap.py
```

### Output files
Script sẽ tạo ra 2 files:
1. `cost_heatmap.png` - PNG format (300 DPI)
2. `cost_heatmap.pdf` - PDF format (publication quality)

Và hiển thị summary statistics trong terminal.

## 📊 Output mẫu

### Terminal Output
```
✓ Heatmap saved to: cost_heatmap.png
✓ PDF version saved to: cost_heatmap.pdf

================================================================================
COST SUMMARY STATISTICS
================================================================================

Math (GPT-5):
------------------------------------------------------------
  OpenEvolve      $   57.55
  GEPA            $   57.00
  AdaEvolve       $   57.52
  EvoX            $   57.33
  SpecEvo         $   16.77  (↓3.42× reduction)

Math (KIMI K2):
------------------------------------------------------------
  OpenEvolve      $   12.28
  GEPA            $   11.83
  AdaEvolve       $   11.92
  EvoX            $   12.35
  SpecEvo         $    6.05  (↓2.00× reduction)

System (GPT-5):
------------------------------------------------------------
  OpenEvolve      $   21.71
  GEPA            $   22.05
  AdaEvolve       $   23.38
  EvoX            $   23.16
  SpecEvo         $    8.11  (↓2.79× reduction)

System (KIMI K2):
------------------------------------------------------------
  OpenEvolve      $    6.75
  GEPA            $    6.13
  AdaEvolve       $    6.88
  EvoX            $    7.32
  SpecEvo         $    2.56  (↓2.65× reduction)

================================================================================
Overall SpecEvo average cost reduction: 2.71×
================================================================================
```

### Visual Output
Heatmap hiển thị:
- **Title**: "Cost Analysis Heatmap"
- **Subtitle**: "Comparison of Methods Across Datasets"
- **Grid**: 5 methods × 4 datasets với màu sắc gradient
- **SpecEvo row**: Highlighted với viền vàng, chứa 3 metrics
- **Colorbar**: Bên phải với label "Total Cost (Lower is Better)"
- **Footer**: Thông điệp tổng kết về cost reduction của SpecEvo

## 🎨 Chi tiết thiết kế

### Color Scheme
- **High cost** (bad): Đỏ (#ff0000)
- **Medium cost**: Vàng (#ffff00)
- **Low cost** (good): Xanh lá (#00ff00)

### Font Sizes
- **Title**: 24pt, bold
- **Subtitle**: 16pt, italic
- **Axis labels**: 18pt, bold
- **Main numbers**: 30pt, bold (28pt cho SpecEvo)
- **SpecEvo metrics**: 15pt (reduction), 13pt (per-task)
- **Footer**: 16pt, bold

### Layout
- **Figure size**: 16×11 inches
- **DPI**: 300 (high resolution)
- **Grid lines**: 2.5pt, black
- **SpecEvo border**: 4pt, gold

## 📐 Cách tính toán

### Total Cost
```python
Total Cost = Sum of (Mean cost across 3 seeds) for all tasks
```

### Reduction Factor (cho SpecEvo)
```python
Reduction = Average cost of other 4 methods / SpecEvo cost
```

### Per-Task Reduction
```python
Per-task = (Avg cost per task of others) / (SpecEvo cost per task)
```

## 🔧 Customization

Để thay đổi thiết kế, chỉnh sửa các tham số trong `plot_cost_heatmap.py`:

```python
# Figure size
fig, ax = plt.subplots(figsize=(16, 11))

# Font sizes
fontsize=30  # Main numbers
fontsize=18  # Axis labels

# Colors
colors = ['#00ff00', '#90ff00', '#ffff00', '#ffaa00', '#ff5500', '#ff0000']

# Grid line width
linewidth=2.5
```

## 📝 Dependencies

```bash
pip install matplotlib numpy
```

## 💡 Use Cases

1. **Research Papers**: Sử dụng PDF version cho publication
2. **Presentations**: Sử dụng PNG version với high DPI
3. **Reports**: Embed vào technical reports
4. **Website**: Display kết quả experiments online

## 🎯 Key Insights từ Heatmap

Từ visualization có thể thấy:

1. **SpecEvo dominates**: Luôn có màu xanh nhất (lowest cost)
2. **GPT-5 vs KIMI K2**: KIMI K2 rẻ hơn ~70% cho cùng tasks
3. **Math vs System**: System tasks có chi phí thấp hơn
4. **Consistency**: SpecEvo giảm cost ổn định ~2-3.5× trên tất cả datasets

## 📖 Related Files

- `calculate_stats.py` - Tính mean, std, best cho từng file
- `compare_all.py` - So sánh tổng hợp tất cả methods
- `ANALYSIS_SUMMARY.md` - Tổng hợp chi tiết kết quả

---

**Created**: June 27, 2026  
**Script**: `plot_cost_heatmap.py`  
**Output**: `cost_heatmap.png`, `cost_heatmap.pdf`
