"""
fft_conv1d 阶段 1 测试：语义一致性 + FFT 正确性

运行：
    cd FFT_CONV1D && python3 -m pytest tests/test_stage1_semantics.py -v
或：
    cd FFT_CONV1D && python3 tests/test_stage1_semantics.py     # 打印误差表
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from fft_conv1d_reference import (  # noqa: E402
    MODE_CORR,
    MODE_MATH,
    choose_n_fft,
    complex_mul,
    direct_conv1d_naive,
    direct_conv1d_torch,
    fft_conv1d_manual_dft,
    fft_conv1d_reference_original,
    fft_conv1d_torch,
    next_pow2,
)

# 题目要求必须覆盖的 shape
REQUIRED_SHAPES = [
    (1, 1, 16, 3),
    (2, 3, 31, 7),
    (4, 8, 64, 1),
    (1, 5, 100, 32),
    (2, 16, 257, 17),
]

# 额外边界 shape：K==L、L 为 2 的幂、L+K-1 恰好是 2 的幂、大 K
EXTRA_SHAPES = [
    (1, 1, 1, 1),
    (1, 1, 8, 8),
    (3, 2, 30, 3),      # L+K-1 = 32，正好 2 的幂
    (2, 4, 128, 64),
    (1, 3, 513, 5),
    (5, 7, 33, 33),
]

ALL_SHAPES = REQUIRED_SHAPES + EXTRA_SHAPES


def _rand(B, H, L, K, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, H, L, generator=g, dtype=dtype)
    k = torch.randn(H, K, generator=g, dtype=dtype)
    return x, k


# ---------------------------------------------------------------------------
# 1. 语义澄清：题目给的参考片段 == correlation 语义，而不是题目的数学公式
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_original_snippet_is_correlation(shape):
    """用 float64 精确验证：题目片段 == sum_j input[t-j]*kernel[K-1-j]。"""
    B, H, L, K = shape
    x, k = _rand(B, H, L, K, dtype=torch.float64, seed=hash(shape) % 10000)

    got = fft_conv1d_reference_original(x, k)
    expect_corr = direct_conv1d_naive(x, k, mode=MODE_CORR)
    torch.testing.assert_close(got, expect_corr, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_original_snippet_differs_from_math_formula_when_K_gt_1(shape):
    """K>1 且 kernel 非对称时，题目片段 != 题目数学公式（说明确实存在语义歧义）。"""
    B, H, L, K = shape
    x, k = _rand(B, H, L, K, dtype=torch.float64, seed=1234)

    got = fft_conv1d_reference_original(x, k)
    expect_math = direct_conv1d_naive(x, k, mode=MODE_MATH)

    if K == 1:
        torch.testing.assert_close(got, expect_math, rtol=1e-12, atol=1e-12)
    else:
        assert not torch.allclose(got, expect_math, rtol=1e-6, atol=1e-6), (
            f"shape={shape} 下两种语义竟然一致，随机数据应当能区分它们"
        )


# ---------------------------------------------------------------------------
# 2. direct 路径自洽：naive 循环 == F.conv1d(翻转 kernel)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ALL_SHAPES)
@pytest.mark.parametrize("mode", [MODE_MATH, MODE_CORR])
def test_direct_torch_matches_naive(shape, mode):
    B, H, L, K = shape
    x, k = _rand(B, H, L, K, dtype=torch.float64, seed=7)

    got = direct_conv1d_torch(x, k, mode=mode)
    expect = direct_conv1d_naive(x, k, mode=mode)
    torch.testing.assert_close(got, expect, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. 因果性 / 冲激响应：最能暴露 kernel 方向和裁剪位置错误的用例
# ---------------------------------------------------------------------------
def test_impulse_response_layout():
    """
    input = 单位冲激 delta[t]，则数学卷积输出应为 kernel 本身放在 t=0..K-1。
    若 kernel 方向错了，会得到翻转的 kernel；若裁剪位置错了，会整体平移。
    """
    B, H, L, K = 1, 2, 16, 5
    x = torch.zeros(B, H, L)
    x[:, :, 0] = 1.0
    k = torch.arange(1, H * K + 1, dtype=torch.float32).reshape(H, K)

    expect = torch.zeros(B, H, L)
    expect[0, :, :K] = k  # 未翻转的 kernel

    for fn in (direct_conv1d_naive, direct_conv1d_torch, fft_conv1d_torch, fft_conv1d_manual_dft):
        got = fn(x, k, mode=MODE_MATH)
        torch.testing.assert_close(got, expect, rtol=1e-5, atol=1e-5,
                                   msg=lambda m, f=fn: f"{f.__name__}: {m}")


def test_causality_no_future_leak():
    """把 input 的后半段改掉，输出前半段必须完全不变（因果性）。"""
    B, H, L, K = 2, 3, 64, 9
    x, k = _rand(B, H, L, K, seed=42)
    y1 = fft_conv1d_torch(x, k, mode=MODE_MATH)

    x2 = x.clone()
    x2[:, :, L // 2:] = torch.randn(B, H, L - L // 2)
    y2 = fft_conv1d_torch(x2, k, mode=MODE_MATH)

    torch.testing.assert_close(y1[:, :, : L // 2], y2[:, :, : L // 2], rtol=1e-5, atol=1e-5)


def test_shift_property():
    """input 整体右移 s，输出也应整体右移 s（前 s 个点为 0）。"""
    B, H, L, K, s = 1, 2, 48, 6, 5
    x, k = _rand(B, H, L, K, seed=11)
    x_shift = torch.zeros_like(x)
    x_shift[:, :, s:] = x[:, :, : L - s]

    y = fft_conv1d_torch(x, k, mode=MODE_MATH)
    y_shift = fft_conv1d_torch(x_shift, k, mode=MODE_MATH)

    torch.testing.assert_close(y_shift[:, :, s:], y[:, :, : L - s], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y_shift[:, :, :s], torch.zeros(B, H, s), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. FFT 卷积 == 直接卷积（float32，主目标）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", ALL_SHAPES)
@pytest.mark.parametrize("mode", [MODE_MATH, MODE_CORR])
def test_fft_matches_direct_fp32(shape, mode):
    B, H, L, K = shape
    x, k = _rand(B, H, L, K, dtype=torch.float32, seed=2024)

    ref = direct_conv1d_naive(x, k, mode=mode, dtype=torch.float64).to(torch.float32)
    got = fft_conv1d_torch(x, k, mode=mode)

    torch.testing.assert_close(got, ref, rtol=5e-3, atol=5e-3)
    max_abs = (got - ref).abs().max().item()
    scale = max(1.0, ref.abs().max().item())
    # 阶段 4 的严格目标：fp32 下相对最大误差 < 1e-4
    assert max_abs / scale < 1e-4, f"shape={shape} mode={mode} 相对误差 {max_abs/scale:.3e} 超标"


@pytest.mark.parametrize("shape", ALL_SHAPES)
def test_manual_dft_matches_direct_fp32(shape):
    """手写 DFT 矩阵 + 实虚分离复乘 + 显式 IRFFT，必须与直接卷积一致。"""
    B, H, L, K = shape
    x, k = _rand(B, H, L, K, dtype=torch.float32, seed=99)

    ref = direct_conv1d_naive(x, k, mode=MODE_MATH, dtype=torch.float64).to(torch.float32)
    got = fft_conv1d_manual_dft(x, k, mode=MODE_MATH)

    torch.testing.assert_close(got, ref, rtol=5e-3, atol=5e-3)
    scale = max(1.0, ref.abs().max().item())
    assert (got - ref).abs().max().item() / scale < 1e-4


@pytest.mark.parametrize("cmul", ["4mul", "3mul"])
def test_complex_mul_variants_agree(cmul):
    """4 次实乘与 3 次实乘（Karatsuba）在 fp32 下结果应一致到 1e-5。"""
    g = torch.Generator().manual_seed(5)
    ar, ai, br, bi = (torch.randn(1024, generator=g) for _ in range(4))
    r4, i4 = complex_mul(ar, ai, br, bi, "4mul")
    r3, i3 = complex_mul(ar, ai, br, bi, "3mul")
    torch.testing.assert_close(r4, r3, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(i4, i3, rtol=1e-5, atol=1e-5)
    # 用 cmul 参数走一遍完整算子路径
    x, k = _rand(2, 3, 64, 7, seed=6)
    ref = direct_conv1d_naive(x, k, mode=MODE_MATH, dtype=torch.float64).to(torch.float32)
    got = fft_conv1d_manual_dft(x, k, mode=MODE_MATH, cmul_method=cmul)
    torch.testing.assert_close(got, ref, rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# 5. FFT 长度选择 / 混叠验证
# ---------------------------------------------------------------------------
def test_next_pow2_and_choose_n_fft():
    assert [next_pow2(n) for n in [1, 2, 3, 5, 16, 17, 1023, 1024]] == [1, 2, 4, 8, 16, 32, 1024, 1024]
    assert choose_n_fft(100, 32) == 256      # L+K-1 = 131 -> 256
    assert choose_n_fft(30, 3) == 32         # L+K-1 = 32  -> 32（边界不进位）
    assert choose_n_fft(16, 3, n_fft=64) == 64
    with pytest.raises(ValueError):
        choose_n_fft(16, 3, n_fft=16)        # 16 < L+K-1 = 18


def test_n_fft_too_small_causes_aliasing():
    """故意用 N < L+K-1 会产生循环卷积混叠，证明 N >= L+K-1 是必要条件。"""
    B, H, L, K = 1, 1, 16, 5
    x, k = _rand(B, H, L, K, seed=3)
    ref = direct_conv1d_naive(x, k, mode=MODE_MATH)

    ok = fft_conv1d_torch(x, k, mode=MODE_MATH, n_fft=32)
    torch.testing.assert_close(ok, ref, rtol=1e-4, atol=1e-4)

    # 绕过校验，直接用 N=16 < 20 做循环卷积
    import torch.nn.functional as F
    N = 16
    Xf = torch.fft.rfft(x, n=N, dim=-1)
    Kf = torch.fft.rfft(F.pad(k, (0, N - K)), n=N, dim=-1)
    bad = torch.fft.irfft(Xf * Kf.unsqueeze(0), n=N, dim=-1)[..., :L]
    assert not torch.allclose(bad, ref, rtol=1e-3, atol=1e-3), "N 过小竟然没有混叠，测试设计有误"


def test_larger_n_fft_gives_same_result():
    """任何 N >= L+K-1 都应给出相同结果（裁剪位置与 N 无关）。"""
    B, H, L, K = 2, 3, 50, 9
    x, k = _rand(B, H, L, K, seed=8)
    ref = direct_conv1d_naive(x, k, mode=MODE_MATH, dtype=torch.float64).to(torch.float32)
    for N in [64, 128, 256, 1024]:
        got = fft_conv1d_torch(x, k, mode=MODE_MATH, n_fft=N)
        torch.testing.assert_close(got, ref, rtol=5e-3, atol=5e-3, msg=f"N_fft={N} 不一致")


# ---------------------------------------------------------------------------
# 6. 随机 shape 大规模对拍
# ---------------------------------------------------------------------------
def test_random_shapes_sweep():
    g = torch.Generator().manual_seed(20260726)
    worst = 0.0
    for trial in range(60):
        B = int(torch.randint(1, 5, (1,), generator=g))
        H = int(torch.randint(1, 17, (1,), generator=g))
        L = int(torch.randint(1, 300, (1,), generator=g))
        K = int(torch.randint(1, L + 1, (1,), generator=g))
        x, k = _rand(B, H, L, K, seed=trial)

        ref = direct_conv1d_naive(x, k, mode=MODE_MATH, dtype=torch.float64).to(torch.float32)
        got = fft_conv1d_torch(x, k, mode=MODE_MATH)
        torch.testing.assert_close(got, ref, rtol=5e-3, atol=5e-3,
                                   msg=f"B={B},H={H},L={L},K={K}")
        scale = max(1.0, ref.abs().max().item())
        worst = max(worst, (got - ref).abs().max().item() / scale)
    assert worst < 1e-4, f"随机 sweep 最大相对误差 {worst:.3e}"


def test_shape_validation():
    x = torch.randn(2, 3, 16)
    with pytest.raises(ValueError):
        direct_conv1d_naive(x, torch.randn(4, 3))          # H 不匹配
    with pytest.raises(ValueError):
        direct_conv1d_naive(x, torch.randn(3, 32))         # K > L
    with pytest.raises(ValueError):
        direct_conv1d_naive(torch.randn(2, 16), torch.randn(2, 3))  # 维度不对


# ---------------------------------------------------------------------------
# 手动运行：打印误差表
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"{'B,H,L,K':>14} | {'N_fft':>6} | {'fft vs direct':>13} | "
          f"{'manualDFT vs direct':>19} | {'rel':>9}")
    print("-" * 78)
    for (B, H, L, K) in ALL_SHAPES:
        x, k = _rand(B, H, L, K, seed=2024)
        ref = direct_conv1d_naive(x, k, mode=MODE_MATH, dtype=torch.float64).to(torch.float32)
        f = fft_conv1d_torch(x, k, mode=MODE_MATH)
        m = fft_conv1d_manual_dft(x, k, mode=MODE_MATH)
        e1 = (f - ref).abs().max().item()
        e2 = (m - ref).abs().max().item()
        scale = max(1.0, ref.abs().max().item())
        print(f"{B},{H},{L},{K}".rjust(14) + f" | {choose_n_fft(L, K):>6} | "
              f"{e1:>13.3e} | {e2:>19.3e} | {e1/scale:>9.2e}")
