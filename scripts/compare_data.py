#!/usr/bin/env python3
"""
比较 NPU 输出与 golden。

判定标准（与题目阶段 4 的要求一致）：
  1. torch.testing.assert_close(rtol=5e-3, atol=5e-3)  —— 必过
  2. 相对最大误差 max|diff| / max(1, max|golden|) < 1e-4 —— float32 的目标线
     未达标时**不放宽阈值**，而是打印误差分布定位来源。

用法：
    python3 scripts/compare_data.py --dir data --out data/npu_out.bin
"""
import argparse
import os

import numpy as np
import torch


def parse_meta(path):
    meta = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                meta[key] = val
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default="data")
    ap.add_argument("--out", type=str, default=None, help="NPU 输出 .bin，默认 <dir>/npu_out.bin")
    args = ap.parse_args()

    meta = parse_meta(os.path.join(args.dir, "meta.txt"))
    B, H, L = int(meta["B"]), int(meta["H"]), int(meta["L"])
    shape = (B, H, L)

    golden = torch.from_numpy(
        np.fromfile(os.path.join(args.dir, "golden.bin"), dtype=np.float32).reshape(shape))
    out_path = args.out or os.path.join(args.dir, "npu_out.bin")
    npu = torch.from_numpy(np.fromfile(out_path, dtype=np.float32).reshape(shape))

    diff = (npu - golden).abs()
    max_abs = diff.max().item()
    scale = max(1.0, golden.abs().max().item())
    rel = max_abs / scale

    print(f"[compare] shape={shape} K={meta['K']}")
    print(f"[compare] golden 幅值范围 [{golden.min().item():.4f}, {golden.max().item():.4f}]")
    print(f"[compare] 最大绝对误差 = {max_abs:.3e}")
    print(f"[compare] 最大相对误差 = {rel:.3e}   (目标 < 1e-4)")

    ok = True
    try:
        torch.testing.assert_close(npu, golden, rtol=5e-3, atol=5e-3)
        print("[compare] assert_close(rtol=5e-3, atol=5e-3): PASS")
    except AssertionError as e:
        ok = False
        print("[compare] assert_close(rtol=5e-3, atol=5e-3): FAIL")
        print(str(e)[:800])

    if rel >= 1e-4:
        ok = False
        print("[compare] 相对误差超过 1e-4 目标线。定位信息：")
        # 逐行统计，判断是个别行错还是整体精度问题
        per_row = diff.reshape(B * H, L).max(dim=1).values
        bad = (per_row / scale >= 1e-4).nonzero().flatten()
        print(f"          超标行数: {bad.numel()} / {B * H}")
        if bad.numel() > 0:
            print(f"          前几个超标行号: {bad[:8].tolist()}")
            r = int(bad[0])
            pos = int(diff.reshape(B * H, L)[r].argmax())
            print(f"          行 {r} 最差位置 t={pos}: npu={npu.reshape(B*H,L)[r,pos]:.6f} "
                  f"golden={golden.reshape(B*H,L)[r,pos]:.6f}")
            print(f"          该行前 8 个输出 npu   : "
                  f"{npu.reshape(B*H,L)[r,:8].tolist()}")
            print(f"          该行前 8 个输出 golden: "
                  f"{golden.reshape(B*H,L)[r,:8].tolist()}")
        if bad.numel() == B * H:
            print("          提示: 全部行超标 => 更像是旋转因子/归一化/裁剪位置问题，"
                  "而不是多核切分问题")
        else:
            print("          提示: 部分行超标 => 优先查多核行切分、尾块与 kernel 频谱复用")

    print("[compare] 结论:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
