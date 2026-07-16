#!/usr/bin/env python3
"""Tính mean và std từ 3 giá trị kết quả (có làm tròn)."""

import statistics

# ===== ĐIỀN 3 GIÁ TRỊ KẾT QUẢ TẠI ĐÂY =====
VALUES = [
   4237.2 ,
   4237.2,
   3623.188406,
]

DECIMALS = 4  # số chữ số thập phân khi làm tròn


def main() -> None:
    mean = statistics.mean(VALUES)
    std = statistics.stdev(VALUES)  # sample std (ddof=1)

    mean_rounded = round(mean, DECIMALS)
    std_rounded = round(std, DECIMALS)

    print(f"Values: {VALUES}")
    print(f"Mean:   {mean_rounded}")
    print(f"Std:    {std_rounded}")


if __name__ == "__main__":
    main()

