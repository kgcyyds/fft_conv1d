"""
fft_conv1d 阶段 2 测试：四步 Cooley-Tukey 分解原型

验证目标（都是阶段 3 AscendC 实现要依赖的性质）：
  1. 四步分解 == 阶段 1 参考实现
  2. D1/D2 是对称阵（=> kernel 里不需要任何转置）
  3. 频谱置换序 [k1][k2] 与 torch.fft 的自然序只差一个已知置换
  4. 正/逆变换的置换互相抵消（kernel 频谱用同一置换即可，无需 digit-reverse）
  5. 零填充跳算 / 输出裁剪跳算不改变结果
  6. batch 配对打包 == 不打包

运行：cd FFT_CONV1D && python3 -m pytest tests/test_stage2_four_step.py -v
"""

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))

from fft_conv1d_reference import (  # noqa: E402
    MODE_CORR,
    MODE_MATH,
    direct_conv1d_naive,
    fft_conv1d_reference_original,
)
from fft_conv1d_four_step import (  # noqa: E402
    L1_TABLE_BUDGET_BYTES,
    tables_bytes,
    FourStepPlan,
    FourStepTables,
    count_macs,
    estimate_crossover_k,
    fft_conv1d_four_step,
    fft_conv1d_four_step_packed,
    four_step_forward_real,
    plan_four_step,
    plan_v1,
)

SHAPES = [
    (1, 1, 16, 3),
    (2, 3, 31, 7),
    (4, 8, 64, 1),
    (1, 5, 100, 32),
    (2, 16, 257, 17),
    (1, 1, 8, 8),
    (3, 2, 30, 3),
    (2, 4, 128, 64),
    (1, 3, 513, 5),
    (2, 2, 1024, 256),
]


def _rand(B, H, L, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(B, H, L, generator=g), torch.randn(H, K, generator=g))


def _ref(x, k, mode=MODE_MATH):
    return direct_conv1d_naive(x, k, mode=mode, dtype=torch.float64).to(torch.float32)


# ---------------------------------------------------------------------------
# 1. 四步分解 == 阶段 1 参考
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("mode", [MODE_MATH, MODE_CORR])
def test_four_step_matches_reference(shape, mode):
    B, H, L, K = shape
    x, k = _rand(B, H, L, K, seed=11)
    got = fft_conv1d_four_step(x, k, mode=mode)
    ref = _ref(x, k, mode=mode)
    torch.testing.assert_close(got, ref, rtol=5e-3, atol=5e-3)
    scale = max(1.0, ref.abs().max().item())
    assert (got - ref).abs().max().item() / scale < 1e-4


@pytest.mark.parametrize("shape", SHAPES)
def test_four_step_corr_mode_matches_original_snippet(shape):
    """corr 模式必须复现题目原始 PyTorch 片段。"""
    B, H, L, K = shape
    x, k = _rand(B, H, L, K, seed=12)
    got = fft_conv1d_four_step(x, k, mode=MODE_CORR)
    ref = fft_conv1d_reference_original(x, k)
    torch.testing.assert_close(got, ref, rtol=5e-3, atol=5e-3)


# ---------------------------------------------------------------------------
# 2. DFT 矩阵对称性 => 硬件上不需要转置
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("M", [8, 16, 32, 64])
def test_dft_matrix_is_symmetric(M):
    plan = FourStepPlan(N=M * M, N1=M, N2=M, L=M * M, K=1)
    tb = FourStepTables.build(plan)
    torch.testing.assert_close(tb.D1r, tb.D1r.t(), rtol=0, atol=0)
    torch.testing.assert_close(tb.D1i, tb.D1i.t(), rtol=0, atol=0)
    torch.testing.assert_close(tb.D2r, tb.D2r.t(), rtol=0, atol=0)
    torch.testing.assert_close(tb.D2i, tb.D2i.t(), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# 3/4. 频谱置换关系，以及置换在正/逆之间抵消
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L,K", [(64, 8), (100, 32), (257, 17), (1024, 128)])
def test_spectrum_is_torch_fft_under_known_permutation(L, K):
    """
    四步正变换输出 C[k1][k2] 应等于 torch.fft.fft(x)[k2*N1 + k1]。
    这是"置换序"说法的严格验证。
    """
    plan = plan_four_step(L, K)
    N, N1, N2 = plan.N, plan.N1, plan.N2
    tb = FourStepTables.build(plan)

    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, L, generator=g)
    x_pad = torch.zeros(2, N)
    x_pad[:, :L] = x

    Cr, Ci = four_step_forward_real(x_pad, plan, tb, plan.R1_in)     # [2,N1,N2]
    ref = torch.fft.fft(x_pad.to(torch.float64), n=N, dim=-1)        # [2,N] 自然序

    # C[k1][k2] 对应自然序下标 k = k2*N1 + k1
    k1 = torch.arange(N1).view(N1, 1)
    k2 = torch.arange(N2).view(1, N2)
    perm = (k2 * N1 + k1).reshape(-1)                                # [N1*N2]
    ref_perm = ref[:, perm].reshape(2, N1, N2)

    scale = ref.abs().max().item()
    assert (Cr - ref_perm.real.float()).abs().max().item() / scale < 1e-5
    assert (Ci - ref_perm.imag.float()).abs().max().item() / scale < 1e-5


def test_permutation_cancels_end_to_end():
    """
    正变换、逆变换都停留在置换序，且 kernel 频谱也用同一置换 ——
    整条链路不做任何 digit-reverse 也能得到正确的时域结果。
    这一点由 test_four_step_matches_reference 隐含验证，这里再显式确认一次：
    人为把 kernel 频谱按自然序存放（即错误地做了一次重排）会得到错误结果。
    """
    B, H, L, K = 1, 1, 64, 8
    x, k = _rand(B, H, L, K, seed=5)
    plan = plan_four_step(L, K)
    N, N1, N2 = plan.N, plan.N1, plan.N2
    tb = FourStepTables.build(plan)

    good = fft_conv1d_four_step(x, k, plan=plan)
    torch.testing.assert_close(good, _ref(x, k), rtol=5e-3, atol=5e-3)

    # 故意把 kernel 频谱转成自然序再参与逐点乘 —— 必须算错
    k_pad = torch.zeros(H, N)
    k_pad[:, :K] = k
    Kr, Ki = four_step_forward_real(k_pad, plan, tb, plan.R1_ker)
    Xr, Xi = four_step_forward_real(
        torch.nn.functional.pad(x.reshape(B * H, L), (0, N - L)), plan, tb, plan.R1_in)
    k1 = torch.arange(N1).view(N1, 1)
    k2 = torch.arange(N2).view(1, N2)
    perm = (k2 * N1 + k1).reshape(-1)
    inv_perm = torch.argsort(perm)
    Kr_bad = Kr.reshape(H, N)[:, inv_perm].reshape(H, N1, N2)
    Ki_bad = Ki.reshape(H, N)[:, inv_perm].reshape(H, N1, N2)
    Yr = Xr * Kr_bad - Xi * Ki_bad
    assert not torch.allclose(Yr, Xr * Kr - Xi * Ki, rtol=1e-3, atol=1e-3), \
        "置换序被错误重排后结果竟然不变，测试设计有误"


# ---------------------------------------------------------------------------
# 5. 稀疏跳算不改变结果
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L,K", [(100, 32), (257, 17), (513, 5), (1024, 128)])
def test_zero_row_skipping_is_exact(L, K):
    """
    首次 GEMM 只对前 ceil(L/N2) 行做收缩（其余是补的零）。
    与"老老实实对全部 N1 行收缩"必须逐位相等（跳过的行确实全零）。
    """
    plan = plan_four_step(L, K)
    N = plan.N
    tb = FourStepTables.build(plan)
    g = torch.Generator().manual_seed(7)
    x_pad = torch.zeros(3, N)
    x_pad[:, :L] = torch.randn(3, L, generator=g)

    skipped = four_step_forward_real(x_pad, plan, tb, plan.R1_in)
    full = four_step_forward_real(x_pad, plan, tb, plan.N1)
    torch.testing.assert_close(skipped[0], full[0], rtol=0, atol=0)
    torch.testing.assert_close(skipped[1], full[1], rtol=0, atol=0)


@pytest.mark.parametrize("L,K", [(100, 32), (257, 17), (1024, 128)])
def test_output_row_cropping_covers_L(L, K):
    """末次 GEMM 只产出 ceil(L/N2) 行，必须刚好覆盖 [0:L] 且不多算整行。"""
    plan = plan_four_step(L, K)
    assert (plan.R1_out - 1) * plan.N2 < L <= plan.R1_out * plan.N2
    assert plan.R1_out <= plan.N1


# ---------------------------------------------------------------------------
# 6. batch 配对打包
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(2, 3, 64, 16), (4, 2, 128, 32), (5, 3, 100, 20),
                                  (2, 16, 257, 17), (1, 4, 64, 8), (3, 1, 512, 64)])
@pytest.mark.parametrize("mode", [MODE_MATH, MODE_CORR])
def test_packed_matches_unpacked(shape, mode):
    """打包路径（含奇数 B 的尾行处理）必须与逐行路径一致。"""
    B, H, L, K = shape
    x, k = _rand(B, H, L, K, seed=21)
    packed = fft_conv1d_four_step_packed(x, k, mode=mode)
    ref = _ref(x, k, mode=mode)
    torch.testing.assert_close(packed, ref, rtol=5e-3, atol=5e-3)
    scale = max(1.0, ref.abs().max().item())
    assert (packed - ref).abs().max().item() / scale < 1e-4


# ---------------------------------------------------------------------------
# 7. plan 与代价模型的自洽性
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("L,K", [(64, 8), (256, 32), (1024, 128), (4096, 512), (8192, 1024)])
def test_plan_invariants(L, K):
    plan = plan_four_step(L, K)
    assert plan.N1 * plan.N2 == plan.N
    assert plan.N >= L + K - 1
    assert plan.N & (plan.N - 1) == 0, "首版本要求 N 为 2 的幂"
    if plan.N >= 256:
        assert plan.N1 >= 16 and plan.N2 >= 16, "fp32 Cube 分形约束"


@pytest.mark.parametrize("L,K", [(256, 64), (1024, 256), (4096, 1024)])
def test_plan_is_cost_optimal(L, K):
    """
    plan 选出的 (N1,N2) 应当是所有**同时满足分形约束和 L1 常量表约束**的
    2 的幂分解中 MAC 数最小的。
    """
    plan = plan_four_step(L, K)
    best = count_macs(1, 1, L, K, plan).cube_mac
    n2 = 1
    while n2 <= plan.N:
        n1 = plan.N // n2
        if (n1 * n2 == plan.N and n1 >= 16 and n2 >= 16
                and tables_bytes(n1, n2, plan.N) <= L1_TABLE_BUDGET_BYTES):
            alt = FourStepPlan(N=plan.N, N1=n1, N2=n2, L=L, K=K)
            assert count_macs(1, 1, L, K, alt).cube_mac >= best, \
                f"存在更优分解 N1={n1},N2={n2}"
        n2 *= 2


@pytest.mark.parametrize("L,K", [(64, 8), (256, 32), (1024, 128), (4096, 1024), (8192, 1024)])
def test_plan_tables_fit_l1(L, K):
    """常量表必须能常驻 L1，否则每个 tile 都要重载 DFT 矩阵。"""
    plan = plan_four_step(L, K)
    assert tables_bytes(plan.N1, plan.N2, plan.N) <= L1_TABLE_BUDGET_BYTES


# ---------------------------------------------------------------------------
# 8. 阶段 3 首版本（v1）方案：N 取 4 的幂、N1==N2、关闭稀疏跳算
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES + [(1, 1, 1, 1), (2, 2, 512, 256), (1, 1, 1024, 1024)])
@pytest.mark.parametrize("mode", [MODE_MATH, MODE_CORR])
def test_v1_plan_matches_reference(shape, mode):
    """v1 的简化分解必须与阶段 1 参考一致 —— 这是 AscendC kernel 的数值契约。"""
    B, H, L, K = shape
    x, k = _rand(B, H, L, K, seed=31)
    plan = plan_v1(L, K)
    got = fft_conv1d_four_step(x, k, mode=mode, plan=plan)
    ref = _ref(x, k, mode=mode)
    torch.testing.assert_close(got, ref, rtol=5e-3, atol=5e-3)
    scale = max(1.0, ref.abs().max().item())
    assert (got - ref).abs().max().item() / scale < 1e-4


@pytest.mark.parametrize("L,K", [(1, 1), (16, 3), (100, 32), (257, 17), (1024, 1024)])
def test_v1_plan_invariants(L, K):
    """v1 的三条结构性约束：N 是 4 的幂、N1==N2、不跳算（GEMM 形状统一）。"""
    plan = plan_v1(L, K)
    assert plan.N1 == plan.N2, "v1 要求 N1==N2，D1 与 D2 才是同一个矩阵"
    assert plan.N1 * plan.N1 == plan.N
    assert plan.N >= max(2, L + K - 1)
    assert plan.N // 4 < max(2, L + K - 1), "N 应是满足条件的最小 4 的幂"
    # 关闭跳算 => 12 次 GEMM 形状全部是 (N1, N1, N1)
    assert plan.R1_in == plan.R1_ker == plan.R1_out == plan.N1
    # 常量表（D + T，D1/D2 共用）必须能常驻 L1
    assert 8 * (plan.N1 * plan.N1 + plan.N) <= L1_TABLE_BUDGET_BYTES


def test_crossover_monotonic_in_rho():
    """Cube/Vector 算力比 rho 越大，FFT 越早占优（K* 单调不增）。"""
    for L in (256, 1024, 4096):
        ks = [estimate_crossover_k(L, r) for r in (8, 16, 32)]
        assert ks[0] >= ks[1] >= ks[2], f"L={L} 的 K* 随 rho 非单调: {ks}"


def test_long_conv_speedup():
    """K=L 的长卷积场景，FFT 路径必须有显著加速（这是本算子的目标区间）。"""
    for L in (1024, 4096):
        plan = plan_four_step(L, L)
        mc = count_macs(1, 1, L, L, plan)
        fft_eq = mc.cube_mac / 16.0 + mc.vector_flop
        assert L * L / fft_eq > 5.0, f"L=K={L} 加速比不足: {L*L/fft_eq:.1f}x"
