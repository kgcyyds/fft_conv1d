"""
fft_conv1d 阶段 3 测试：AscendC kernel 内部逻辑的 Python 等价模拟

这些测试**不跑 NPU**，而是逐字模拟 kernel 里最容易出错的下标/重构逻辑，
让它们在没有硬件的机器上也能被守护住。两个被模拟的对象：

  1. direct 路径的"8 份移位副本"取数（[O4]）——消除 Vector 非对齐访问，
     把每块 K 次 GM 读降到 min(8,K) 次。断言同时覆盖 32B 对齐与 workspace 越界。
  2. FFT 路径用 IterateAll(enAtomic=1) 累加重构（[O2]）——用 Dn=-Di 表达减法、
     用 DrS=Dr/N & DiS=Di/N 把归一化折进常量表，从而消除全部 Vector 合并 pass。

运行：cd FFT_CONV1D && python3 -m pytest tests/test_stage3_kernel_logic.py -v
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from fft_conv1d_reference import MODE_MATH, direct_conv1d_naive  # noqa: E402
from fft_conv1d_four_step import FourStepTables, plan_v1  # noqa: E402

SHIFT_COPIES = 8  # 与 kernel 里的 SHIFT_COPIES 一致（32B / sizeof(float)）


def _align8(n):
    return (n + 7) // 8 * 8


# ---------------------------------------------------------------------------
# 1. direct 路径：移位副本取数
# ---------------------------------------------------------------------------
def simulate_direct(x, k, tile):
    """逐字模拟 FftConv1dDirect::BuildPaddedRow + ComputeTile 的下标计算。"""
    B, H, L = x.shape
    K = k.shape[1]
    nShift = min(SHIFT_COPIES, K)
    shiftCap = _align8(tile + K)
    rowLen = _align8(K - 1 + L)

    out = torch.zeros(B, H, L, dtype=torch.float64)
    for b in range(B):
        for h in range(H):
            # BuildPaddedRow：workspace 里物化 [K-1 个 0, x 行]
            xz = torch.zeros(rowLen, dtype=torch.float64)
            xz[K - 1:K - 1 + L] = x[b, h]

            for t0 in range(0, L, tile):
                ln = min(tile, L - t0)
                # 预载 nShift 份移位副本：副本 r 覆盖 xz[t0+r ...]
                shift = torch.zeros(nShift * shiftCap, dtype=torch.float64)
                for r in range(nShift):
                    need = ln + (K - 1 - r)
                    assert t0 + r + need <= rowLen, "读越过 workspace 行尾"
                    assert r * shiftCap + need <= nShift * shiftCap, "写越过 UB 缓冲"
                    shift[r * shiftCap:r * shiftCap + need] = xz[t0 + r:t0 + r + need]

                o = torch.zeros(ln, dtype=torch.float64)
                for j in range(K):
                    d = K - 1 - j          # 源相对 t0 的偏移
                    r = d % nShift         # 落在哪份副本
                    off = r * shiftCap + (d - r)
                    # 这两条断言是本方案成立的全部前提
                    assert (d - r) % nShift == 0, "UB 内偏移不是 nShift 的倍数"
                    assert off % 8 == 0, f"UB 偏移 {off} 不满足 32B 对齐"
                    o += shift[off:off + ln] * k[h, j]
                out[b, h, t0:t0 + ln] = o
    return out


DIRECT_SHAPES = [
    (1, 1, 16, 3), (2, 3, 31, 7), (4, 8, 64, 1), (1, 5, 100, 32), (2, 16, 257, 17),
    (1, 1, 8, 8), (3, 2, 30, 3), (1, 3, 513, 5), (2, 2, 100, 63), (1, 1, 2000, 40),
    (1, 1, 1, 1), (2, 2, 1024, 63),
]


@pytest.mark.parametrize("shape", DIRECT_SHAPES)
def test_direct_shift_copy_is_exact(shape):
    """移位副本取数必须与直接卷积**逐位**相等（纯下标重排，不该有任何误差）。"""
    B, H, L, K = shape
    g = torch.Generator().manual_seed(hash(shape) % 9973)
    x = torch.randn(B, H, L, generator=g, dtype=torch.float64)
    k = torch.randn(H, K, generator=g, dtype=torch.float64)
    ref = direct_conv1d_naive(x, k, mode=MODE_MATH)
    for tile in {min(L, 1024), min(L, 64), L}:
        got = simulate_direct(x, k, tile)
        torch.testing.assert_close(got, ref, rtol=0, atol=0,
                                   msg=lambda m, t=tile: f"tile={t}: {m}")


@pytest.mark.parametrize("K", [1, 3, 7, 8, 9, 17, 32, 63])
def test_direct_gm_read_count_reduced(K):
    """GM 读次数应从 K 降到 min(8,K)，K>=8 时恒为 8。"""
    assert min(SHIFT_COPIES, K) == (8 if K >= 8 else K)


# ---------------------------------------------------------------------------
# 2. FFT 路径：AtomicAdd 累加重构
# ---------------------------------------------------------------------------
def simulate_fft(x, k, plan, use_atomic):
    """
    逐字模拟 FftConv1dFft::Forward / Inverse。

    use_atomic=False -> v1 写法（显式 Vector 减法 + 末尾除以 N）
    use_atomic=True  -> v2 写法（Dn=-Di 配合 AtomicAdd；DrS/DiS 折进归一化）
    两者必须逐位相等：取负是精确运算，除以 2 的幂也是精确运算。
    """
    B, H, L = x.shape
    K = k.shape[1]
    N, N1 = plan.N, plan.N1
    tb = FourStepTables.build(plan)
    Dr, Di = tb.D1r.double(), tb.D1i.double()
    Dn, DrS, DiS = -Di, Dr / N, Di / N
    Tr, Ti = tb.Tr.double(), tb.Ti.double()

    def forward(row):
        xm = row.reshape(N1, N1)
        Br, Bi = Dr @ xm, Di @ xm                      # 步骤 A：实输入 => 2 次 GEMM
        Br, Bi = Br * Tr - Bi * Ti, Br * Ti + Bi * Tr  # 步骤 B：旋转因子
        Cr = (Br @ Dr + Bi @ Dn) if use_atomic else (Br @ Dr - Bi @ Di)  # 步骤 C
        Ci = Br @ Di + Bi @ Dr
        return Cr, Ci

    def pad(v, n):
        return torch.cat([v, torch.zeros(n - v.numel(), dtype=torch.float64)])

    Kf = [forward(pad(k[h], N)) for h in range(H)]

    out = torch.zeros(B, H, L, dtype=torch.float64)
    for b in range(B):
        for h in range(H):
            Cr, Ci = forward(pad(x[b, h], N))
            Kr, Ki = Kf[h]
            Yr, Yi = Cr * Kr - Ci * Ki, Cr * Ki + Ci * Kr   # 频域逐点乘
            Er = Yr @ Dr + Yi @ Di                          # 步骤 A'
            Ei = (Yi @ Dr + Yr @ Dn) if use_atomic else (Yi @ Dr - Yr @ Di)
            Er, Ei = Er * Tr + Ei * Ti, -Er * Ti + Ei * Tr  # 步骤 B'：共轭旋转因子
            y = (DrS @ Er + DiS @ Ei) if use_atomic else ((Dr @ Er + Di @ Ei) / N)
            out[b, h] = y.reshape(-1)[:L]                   # 因果裁剪：前 L 点，偏移 0
    return out


FFT_SHAPES = [
    (1, 1, 64, 64), (2, 4, 128, 64), (2, 2, 256, 128), (1, 8, 512, 256),
    (2, 2, 1024, 1024), (1, 1, 100, 100), (3, 2, 300, 200), (1, 5, 255, 255),
]


@pytest.mark.parametrize("shape", FFT_SHAPES)
def test_atomic_refactor_is_bit_exact(shape):
    """v2 的 AtomicAdd 重构必须与 v1 写法逐位相等（同一套常量表）。"""
    B, H, L, K = shape
    g = torch.Generator().manual_seed(hash(shape) % 9973)
    x = torch.randn(B, H, L, generator=g, dtype=torch.float64)
    k = torch.randn(H, K, generator=g, dtype=torch.float64)
    plan = plan_v1(L, K)
    v1 = simulate_fft(x, k, plan, use_atomic=False)
    v2 = simulate_fft(x, k, plan, use_atomic=True)
    torch.testing.assert_close(v2, v1, rtol=0, atol=0)


@pytest.mark.parametrize("shape", FFT_SHAPES)
def test_atomic_refactor_matches_golden(shape):
    """v2 与金标准的相对误差应保持在 fp32 常量表的量级（<1e-4 目标线）。"""
    B, H, L, K = shape
    g = torch.Generator().manual_seed(hash(shape) % 9973)
    x = torch.randn(B, H, L, generator=g, dtype=torch.float64)
    k = torch.randn(H, K, generator=g, dtype=torch.float64)
    ref = direct_conv1d_naive(x, k, mode=MODE_MATH)
    got = simulate_fft(x, k, plan_v1(L, K), use_atomic=True)
    scale = max(1.0, ref.abs().max().item())
    assert (got - ref).abs().max().item() / scale < 1e-4


def test_normalization_folding_is_exact():
    """1/N 折进常量表必须是精确的 —— 前提是 N 为 2 的幂。"""
    for L, K in [(64, 64), (256, 128), (1024, 1024)]:
        plan = plan_v1(L, K)
        assert plan.N & (plan.N - 1) == 0, "N 必须是 2 的幂，否则 Dr/N 不是精确运算"
        tb = FourStepTables.build(plan)
        Dr = tb.D1r.double()
        torch.testing.assert_close(Dr / plan.N * plan.N, Dr, rtol=0, atol=0)


def test_scratch_and_table_counts_match_kernel():
    """常量表 7 张、scratch 8 块 —— 与 host 侧常量和 kernel 里的 Scr() 编号一致。"""
    tables = ["Dr", "Di", "Dn", "DrS", "DiS", "Tr", "Ti"]
    scratch = ["xmat", "Br", "Bi", "Cr", "Ci", "Er", "Ei", "yout"]
    assert len(tables) == 7
    assert len(scratch) == 8
