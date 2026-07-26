#!/usr/bin/env python3
"""
生成 fft_conv1d 的测试输入与 golden 输出（float32 裸二进制）。

golden 来自阶段 1 的直接卷积参考实现（float64 累加后转 float32），
即本项目的数值金标准，不是 FFT 结果 —— 这样才能真正检验 NPU 上的 FFT 路径。

用法：
    python3 scripts/gen_data.py --b 2 --h 16 --l 257 --k 17 --flip 0 --out data/
生成：
    data/x.bin       [B,H,L]  float32
    data/kernel.bin  [H,K]    float32
    data/golden.bin  [B,H,L]  float32
    data/meta.txt    形状与属性
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from fft_conv1d_reference import MODE_CORR, MODE_MATH, direct_conv1d_naive  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--l", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--flip", type=int, default=0, help="0=math 语义, 1=corr 语义")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data")
    args = ap.parse_args()

    if args.k > args.l:
        raise SystemExit(f"约束要求 K <= L，实际 K={args.k}, L={args.l}")
    if args.l + args.k - 1 > 16384:
        raise SystemExit(f"v1 约束要求 L+K-1 <= 16384，实际 {args.l + args.k - 1}")

    os.makedirs(args.out, exist_ok=True)
    g = torch.Generator().manual_seed(args.seed)
    x = torch.randn(args.b, args.h, args.l, generator=g, dtype=torch.float32)
    k = torch.randn(args.h, args.k, generator=g, dtype=torch.float32)

    mode = MODE_CORR if args.flip else MODE_MATH
    golden = direct_conv1d_naive(x, k, mode=mode, dtype=torch.float64).to(torch.float32)

    x.numpy().astype(np.float32).tofile(os.path.join(args.out, "x.bin"))
    k.numpy().astype(np.float32).tofile(os.path.join(args.out, "kernel.bin"))
    golden.numpy().astype(np.float32).tofile(os.path.join(args.out, "golden.bin"))
    with open(os.path.join(args.out, "meta.txt"), "w") as f:
        f.write(f"B={args.b}\nH={args.h}\nL={args.l}\nK={args.k}\n"
                f"flip_kernel={args.flip}\nmode={mode}\nseed={args.seed}\n")

    nfft = 4
    while nfft < max(2, args.l + args.k - 1):
        nfft *= 4
    algo = "DIRECT" if args.k < 64 else "FFT"
    print(f"[gen_data] B={args.b} H={args.h} L={args.l} K={args.k} mode={mode}")
    print(f"[gen_data] 预期算法路径={algo}  N_fft={nfft}  N1=N2={int(nfft ** 0.5)}")
    print(f"[gen_data] 输出目录: {args.out}")


if __name__ == "__main__":
    main()
