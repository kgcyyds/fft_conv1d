#!/usr/bin/env python3
"""
配合 op_kernel 里的 DEBUG_DUMP_STAGE，打印各级中间结果的**期望值**，
与 npu_out.bin 实际读回的内容比对，定位 FFT 路径从哪一级开始出错。

用法（先在 kernel 里把 DEBUG_DUMP_STAGE 改成对应的值并重编）：
    python3 scripts/dump_expect.py --stage 3 --dir data
"""
import argparse
import math
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from fft_conv1d_four_step import FourStepTables, plan_v1  # noqa: E402

STAGE_NAME = {
    0: "Dr  常量表实部（GM_DUMP_BUF=0）",
    1: "Dr  常量表实部",
    2: "Tr  旋转因子实部",
    3: "scr0 补零后的输入行",
    4: "scr1 步骤A的 Br = Dr @ xmat",
    5: "scr5 正变换输出实部",
    6: "Kfr kernel 频谱实部",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[0, 1, 2, 3, 4, 5, 6])
    ap.add_argument("--dir", type=str, default="data")
    ap.add_argument("--n", type=int, default=16, help="打印前几个元素")
    args = ap.parse_args()

    meta = {}
    with open(os.path.join(args.dir, "meta.txt")) as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                meta[k] = v
    B, H, L, K = int(meta["B"]), int(meta["H"]), int(meta["L"]), int(meta["K"])

    x = torch.from_numpy(
        np.fromfile(os.path.join(args.dir, "x.bin"), dtype=np.float32).reshape(B, H, L))
    k = torch.from_numpy(
        np.fromfile(os.path.join(args.dir, "kernel.bin"), dtype=np.float32).reshape(H, K))

    plan = plan_v1(L, K)
    N, N1 = plan.N, plan.N1
    tb = FourStepTables.build(plan)

    if args.stage == 0:   # GM_DUMP_BUF=0 -> Dr
        exp = tb.D1r.reshape(-1)[:L]
    elif args.stage == 1:
        exp = tb.D1r.reshape(-1)[:L]
    elif args.stage == 2:
        exp = tb.Tr.reshape(-1)[:L]
    elif args.stage == 3:
        exp = torch.zeros(N)
        exp[:L] = x[0, 0]
        exp = exp[:L]
    elif args.stage == 4:
        xp = torch.zeros(N)
        xp[:L] = x[0, 0]
        exp = (tb.D1r @ xp.reshape(N1, N1)).reshape(-1)[:L]
    elif args.stage == 5:
        xp = torch.zeros(N)
        xp[:L] = x[0, 0]
        Br = tb.D1r @ xp.reshape(N1, N1)
        Bi = tb.D1i @ xp.reshape(N1, N1)
        Br, Bi = Br * tb.Tr - Bi * tb.Ti, Br * tb.Ti + Bi * tb.Tr
        exp = (Br @ tb.D1r - Bi @ tb.D1i).reshape(-1)[:L]
    else:  # 6
        kp = torch.zeros(N)
        kp[:K] = k[0]
        Br = tb.D1r @ kp.reshape(N1, N1)
        Bi = tb.D1i @ kp.reshape(N1, N1)
        Br, Bi = Br * tb.Tr - Bi * tb.Ti, Br * tb.Ti + Bi * tb.Tr
        exp = (Br @ tb.D1r - Bi @ tb.D1i).reshape(-1)[:L]

    print(f"shape B={B} H={H} L={L} K={K}   N={N}  N1=N2={N1}")
    print(f"DEBUG_DUMP_STAGE={args.stage}  ({STAGE_NAME[args.stage]})")
    print(f"期望（行 b=0,h=0 的前 {args.n} 个）:")
    print("  " + "  ".join(f"{v:9.4f}" for v in exp[:args.n].tolist()))

    out_path = os.path.join(args.dir, "npu_out.bin")
    if os.path.exists(out_path):
        npu = torch.from_numpy(
            np.fromfile(out_path, dtype=np.float32).reshape(B, H, L))[0, 0]
        print(f"NPU 实际:")
        print("  " + "  ".join(f"{v:9.4f}" for v in npu[:args.n].tolist()))
        err = (npu[:len(exp)] - exp).abs().max().item()
        print(f"\n最大误差 = {err:.3e}   ->  {'一致 ✔' if err < 1e-3 else '不一致 ✘'}")
        if npu.abs().max().item() == 0.0:
            print("  NPU 全为 0：这一级（或更早）就已经是 0")
    else:
        print(f"（没找到 {out_path}，先跑一次 NPU）")


if __name__ == "__main__":
    main()
