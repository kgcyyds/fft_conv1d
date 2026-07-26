"""
fft_conv1d 阶段 2：四步 Cooley-Tukey 分解的数值原型

本文件是"未来 AscendC kernel 的逐步等价物"：
  - 所有矩阵乘都是**实数 GEMM**（可直接映射到 Cube / Matmul API）
  - 所有逐点运算都是**实虚分离的向量运算**（可直接映射到 Vector 指令）
  - 不使用 torch.fft，不使用 complex dtype
  - 旋转因子/DFT 矩阵用 float64 角度生成后转 float32（与 ops-fft 的 host 侧做法一致）

分解（N = N1 * N2，W_M = exp(-2πi/M)）
-------------------------------------------------
时域下标 n = n1*N2 + n2   -> x_mat[n1][n2] 是 x 的**自然 row-major reshape**
频域下标 k = k2*N1 + k1   -> 频谱以 [k1][k2] 矩阵形式存放（置换序，不还原）

正变换：
  A. B [k1][n2] = Σ_n1 D1[k1][n1] · x_mat[n1][n2]      = D1 @ x_mat      (GEMM)
  B. B'[k1][n2] = B[k1][n2] · W_N^{k1·n2}              = B ⊙ T           (Vector)
  C. C [k1][k2] = Σ_n2 B'[k1][n2] · D2[k2][n2]         = B' @ D2         (GEMM, D2 对称)

逆变换（完全镜像，共轭 + 1/N）：
  A'. E [k1][n2] = Σ_k2 Y[k1][k2] · conj(D2)[k2][n2]   = Y @ conj(D2)    (GEMM)
  B'. E'[k1][n2] = E[k1][n2] · W_N^{-k1·n2}            = E ⊙ conj(T)     (Vector)
  C'. y_mat[n1][n2] = (1/N) Σ_k1 conj(D1)[n1][k1]·E'[k1][n2]             (GEMM)
      -> y_mat 就是输出的**自然 row-major reshape**

三个关键性质（都在测试中验证）
-------------------------------------------------
1. D1、D2 都是**对称矩阵**（D[k][n]=W^{nk}=D[n][k]），所以全程**不需要任何转置**。
2. 频谱停留在置换序 [k1][k2]，正变换的置换与逆变换的置换互相抵消；
   只要 kernel 频谱用**同一套置换**存放，就**永远不需要做 digit-reverse 重排**。
3. 实输入 / 实输出让首尾两次 GEMM 各自从 4 次实数 GEMM 降为 2 次。
   零填充与输出裁剪进一步把首尾 GEMM 的一个维度从 N1 缩到 ceil(L/N2)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch

from fft_conv1d_reference import MODE_MATH, check_shapes, choose_n_fft, next_pow2

__all__ = [
    "L1_SIZE_BYTES",
    "L0A_SIZE_BYTES",
    "L0B_SIZE_BYTES",
    "L0C_SIZE_BYTES",
    "UB_SIZE_BYTES",
    "L1_TABLE_BUDGET_BYTES",
    "MAX_N_FFT",
    "tables_bytes",
    "FourStepPlan",
    "plan_four_step",
    "plan_v1",
    "fft_conv1d_four_step",
    "MacCount",
    "count_macs",
    "estimate_crossover_k",
]


# ============================================================================
# 0. 硬件常量（Ascend 910B / V220，数值取自 ops-fft 的 hardware.h，已核对）
# ============================================================================
L1_SIZE_BYTES = 512 * 1024
L0A_SIZE_BYTES = 64 * 1024
L0B_SIZE_BYTES = 64 * 1024
L0C_SIZE_BYTES = 128 * 1024
UB_SIZE_BYTES = 192 * 1024

# 常量表在 L1 上的预算：留 64KB 给 A/B 矩阵的 ping-pong 载入
L1_TABLE_BUDGET_BYTES = 448 * 1024

# 首版本 N_fft 上限（见设计文档 §5 的容量推导）
MAX_N_FFT = 8192


def tables_bytes(N1: int, N2: int, N: int) -> int:
    """常量表字节数：D1(复) + D2(复) + T(复)，fp32。"""
    return 8 * (N1 * N1 + N2 * N2 + N)


# ============================================================================
# 1. 分解方案选择
# ============================================================================
@dataclass
class FourStepPlan:
    """N = N1 * N2 的分解方案 + 由 L/K 决定的稀疏跳算参数。"""
    N: int
    N1: int
    N2: int
    L: int
    K: int
    skip_sparse: bool = True   # 是否启用零填充/裁剪跳算（v1 关闭以统一 GEMM 形状）
    R1_in: int = 0    # 输入非零行数 ceil(L/N2)：首次 GEMM 的收缩维
    R1_ker: int = 0   # kernel 非零行数 ceil(K/N2)
    R1_out: int = 0   # 输出需要的行数 ceil(L/N2)：末次 GEMM 的输出维

    def __post_init__(self):
        assert self.N1 * self.N2 == self.N, "N1*N2 必须等于 N"
        if self.skip_sparse:
            self.R1_in = (self.L + self.N2 - 1) // self.N2
            self.R1_ker = (self.K + self.N2 - 1) // self.N2
            self.R1_out = (self.L + self.N2 - 1) // self.N2
        else:
            self.R1_in = self.R1_ker = self.R1_out = self.N1


def plan_v1(L: int, K: int) -> FourStepPlan:
    """
    阶段 3 首版本（v1）的分解方案。

    与 plan_four_step 的区别，以及为什么这样取舍：
      1. N 取 **4 的幂**（N = 4^ceil(log4(L+K-1))），于是 N1 = N2 = sqrt(N)。
         => D1 和 D2 是**同一个矩阵**，常量表减半。
      2. 关闭稀疏跳算（R1_in = R1_ker = R1_out = N1）。
         => 四个步骤的 GEMM 形状全部是 (M=N1, K=N1, N=N1)，**12 次 GEMM 共用一个
            Matmul 对象和一套 TCubeTiling**，host/kernel 代码量和出错面大幅下降。
      3. 代价：N 最多浪费 2 倍；MAC 数约为最优解的 2.7 倍。
         L=K=1024 时相对 direct 仍有约 3.7x（ρ=16），可接受。
      4. 结构上仍然是四步 Cooley-Tukey，阶段 5 打开跳算 + 非方形分解是**优化而非重写**。
    """
    need = max(2, L + K - 1)
    n = 4
    while n < need:
        n *= 4
    root = int(round(math.sqrt(n)))
    assert root * root == n
    return FourStepPlan(N=n, N1=root, N2=root, L=L, K=K, skip_sparse=False)


def plan_four_step(L: int, K: int, n_fft: Optional[int] = None) -> FourStepPlan:
    """
    选择 (N1, N2)。

    代价模型（每行的实数 MAC 数，见 count_macs 的精确版本）：
        total ≈ 8·N·N2 + 4·L·N1 = 8·N·N2 + 4·L·N/N2
    对 N2 求导置零 =>  N2* = sqrt(L/2)
    再取最接近的 2 的幂（且保证 N1 = N/N2 >= 1）。

    直观解释：
      - 中间两次 GEMM（步骤 C 和 A'）代价 ∝ N·N2，希望 N2 小；
      - 首尾两次 GEMM 代价 ∝ L·N1 = L·N/N2，希望 N2 大；
      - 两者平衡点在 sqrt(L/2)。因为首尾 GEMM 已被零填充/裁剪削掉了一大半，
        所以最优 N2 明显小于教科书上的 sqrt(N)。

    硬件约束（fp32 Cube 分形为 16(M) x 8(K)，L0C 输出块按 16 对齐）：
      N >= 256 时强制 N1 >= 16 且 N2 >= 16，避免 GEMM 退化成瘦长条浪费分形；
      N < 256 时放宽到 >= 8（这些尺寸下 FFT 路径本来也竞争不过 direct）。
    """
    N = choose_n_fft(L, K, n_fft)
    if N < 4:
        return FourStepPlan(N=N, N1=N, N2=1, L=L, K=K)

    floor_dim = 16 if N >= 256 else min(8, N)
    best = None
    n2 = 1
    while n2 <= N:
        n1 = N // n2
        if n1 * n2 == N and n1 >= floor_dim and n2 >= floor_dim:
            # L1 硬约束：常量表 D1(2·N1²) + D2(2·N2²) + T(2·N) 必须常驻 L1，
            # 否则每个 tile 都要重新从 GM 载入 DFT 矩阵，搬运量会彻底压垮收益。
            if tables_bytes(n1, n2, N) > L1_TABLE_BUDGET_BYTES:
                n2 *= 2
                continue
            cost = 8.0 * N * n2 + 4.0 * L * n1
            if best is None or cost < best[0]:
                best = (cost, n1, n2)
        n2 *= 2
    if best is None:      # N 太小，无法满足分形约束，退化成单级 DFT
        return FourStepPlan(N=N, N1=N, N2=1, L=L, K=K)
    _, n1, n2 = best
    return FourStepPlan(N=N, N1=n1, N2=n2, L=L, K=K)


# ============================================================================
# 2. DFT 矩阵与旋转因子（float64 生成 -> float32 存储）
# ============================================================================
def _angle(prod: torch.Tensor, M: int) -> torch.Tensor:
    """
    由整数乘积 prod 生成角度 -2π·(prod mod M)/M。

    两个关键点（硬件生成旋转因子时同样要照做）：
      1. 先做**整数**乘法再取模，而不是 (-2π·a)·b/M 这样的浮点连乘 ——
         浮点乘法不满足结合律，(-2π·a)·b ≠ (-2π·b)·a，会让本应对称的
         DFT 矩阵出现 ~1e-14 的不对称，破坏"D 对称所以不用转置"这一前提。
      2. mod M 把角度压回 [0, 2π)，避免大幅角的三角函数参数规约误差
         （N 很大时 k·n 可达 N²，直接算 cos 会损失有效位）。
    """
    ang = prod.remainder(M).to(torch.float64)
    return -2.0 * math.pi * ang / M


def _dft_matrix(M: int, dtype: torch.dtype, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """D[k][n] = W_M^{n·k} = exp(-2πi·nk/M)，返回 (实部, 虚部)。D 是对称阵。"""
    idx = torch.arange(M, dtype=torch.int64, device=device)
    ang = _angle(idx.view(M, 1) * idx.view(1, M), M)
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def _twiddle(N1: int, N2: int, N: int, dtype: torch.dtype, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """T[k1][n2] = W_N^{k1·n2}，形状 [N1, N2]。"""
    k1 = torch.arange(N1, dtype=torch.int64, device=device).view(N1, 1)
    n2 = torch.arange(N2, dtype=torch.int64, device=device).view(1, N2)
    ang = _angle(k1 * n2, N)
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


@dataclass
class FourStepTables:
    """所有常量表。在 NPU 上这些常驻 L1（对所有 batch/channel 共享）。"""
    D1r: torch.Tensor
    D1i: torch.Tensor
    D2r: torch.Tensor
    D2i: torch.Tensor
    Tr: torch.Tensor
    Ti: torch.Tensor

    @staticmethod
    def build(plan: FourStepPlan, dtype=torch.float32, device="cpu") -> "FourStepTables":
        D1r, D1i = _dft_matrix(plan.N1, dtype, device)
        D2r, D2i = _dft_matrix(plan.N2, dtype, device)
        Tr, Ti = _twiddle(plan.N1, plan.N2, plan.N, dtype, device)
        return FourStepTables(D1r, D1i, D2r, D2i, Tr, Ti)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for t in (self.D1r, self.D1i, self.D2r, self.D2i, self.Tr, self.Ti))


# ============================================================================
# 3. 复数逐点乘（Vector，4 次实乘）
# ============================================================================
def _cmul(ar, ai, br, bi):
    """逐点复乘，4 次实乘 2 次加减。Vector 上对应 Mul/Mul/Sub + Mul/Mul/Add。"""
    return ar * br - ai * bi, ar * bi + ai * br


# ============================================================================
# 4. 正变换 / 逆变换（全部是实数 GEMM）
# ============================================================================
def four_step_forward_real(
    x: torch.Tensor, plan: FourStepPlan, tb: FourStepTables, valid_rows: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    实输入正变换。
      x          : [R, N]，已右侧补零（补零部分不参与计算）
      valid_rows : x_mat 中非零的 n1 行数（= ceil(有效长度/N2)）
    返回频谱 (Cr, Ci)，形状 [R, N1, N2]，下标含义 [k1][k2]（置换序）。
    """
    R = x.shape[0]
    N1, N2 = plan.N1, plan.N2
    x_mat = x.view(R, N1, N2)

    # --- 步骤 A：列方向 DFT_{N1}。实输入 => 只需 2 次实数 GEMM ---
    # 只对前 valid_rows 行做收缩（其余行是补的零，直接跳过）
    xs = x_mat[:, :valid_rows, :]                 # [R, R1, N2]
    Br = tb.D1r[:, :valid_rows] @ xs              # [N1,R1] @ [R,R1,N2] -> [R,N1,N2]
    Bi = tb.D1i[:, :valid_rows] @ xs

    # --- 步骤 B：乘旋转因子 W_N^{k1·n2}（Vector 逐点复乘）---
    Br, Bi = _cmul(Br, Bi, tb.Tr, tb.Ti)

    # --- 步骤 C：行方向 DFT_{N2}。D2 对称，无需转置。4 次实数 GEMM ---
    Cr = Br @ tb.D2r - Bi @ tb.D2i
    Ci = Br @ tb.D2i + Bi @ tb.D2r
    return Cr, Ci


def four_step_inverse_real(
    Yr: torch.Tensor, Yi: torch.Tensor, plan: FourStepPlan, tb: FourStepTables, out_rows: int
) -> torch.Tensor:
    """
    逆变换并只产出前 out_rows 行（即前 out_rows*N2 个时间点），实输出。
    返回 [R, out_rows, N2]，reshape 后就是时域自然顺序。
    """
    # --- 步骤 A'：行方向 IDFT_{N2}，conj(D2) = (D2r, -D2i) ---
    Er = Yr @ tb.D2r + Yi @ tb.D2i
    Ei = -Yr @ tb.D2i + Yi @ tb.D2r

    # --- 步骤 B'：乘共轭旋转因子 ---
    Er, Ei = _cmul(Er, Ei, tb.Tr, -tb.Ti)

    # --- 步骤 C'：列方向 IDFT_{N1}，实输出 => 只取实部，2 次实数 GEMM ---
    # Re{ conj(D1) @ E' } = D1r @ E'r + D1i @ E'i，且只算前 out_rows 行
    y = tb.D1r[:out_rows, :] @ Er + tb.D1i[:out_rows, :] @ Ei
    return y / plan.N          # backward 归一化


# ============================================================================
# 5. 完整算子
# ============================================================================
def fft_conv1d_four_step(
    input: torch.Tensor,
    kernel: torch.Tensor,
    mode: str = MODE_MATH,
    n_fft: Optional[int] = None,
    plan: Optional[FourStepPlan] = None,
) -> torch.Tensor:
    """
    四步分解版 fft_conv1d，纯实数 GEMM + 实虚分离向量运算。
    数值路径与规划中的 AscendC kernel 一一对应。
    """
    B, H, L, K = check_shapes(input, kernel)
    if plan is None:
        plan = plan_four_step(L, K, n_fft)
    N, N1, N2 = plan.N, plan.N1, plan.N2
    dev = input.device
    tb = FourStepTables.build(plan, torch.float32, dev)

    x = input.to(torch.float32).reshape(B * H, L)
    k = kernel.to(torch.float32)
    if mode != MODE_MATH:
        k = k.flip(-1)          # corr 语义：时域翻转 kernel

    # 右侧补零到 N
    x_pad = torch.zeros(B * H, N, dtype=torch.float32, device=dev)
    x_pad[:, :L] = x
    k_pad = torch.zeros(H, N, dtype=torch.float32, device=dev)
    k_pad[:, :K] = k

    # kernel 频谱（每个通道算一次，NPU 上放 workspace，[k1][k2] 置换序）
    Kr, Ki = four_step_forward_real(k_pad, plan, tb, plan.R1_ker)      # [H,N1,N2]

    # 输入频谱
    Xr, Xi = four_step_forward_real(x_pad, plan, tb, plan.R1_in)       # [B*H,N1,N2]

    # 频域逐点复乘：kernel 频谱按通道广播到 batch
    Kr_b = Kr.view(1, H, N1, N2).expand(B, H, N1, N2).reshape(B * H, N1, N2)
    Ki_b = Ki.view(1, H, N1, N2).expand(B, H, N1, N2).reshape(B * H, N1, N2)
    Yr, Yi = _cmul(Xr, Xi, Kr_b, Ki_b)

    # 逆变换，只产出前 R1_out 行
    y = four_step_inverse_real(Yr, Yi, plan, tb, plan.R1_out)          # [B*H,R1_out,N2]
    y = y.reshape(B * H, plan.R1_out * N2)[:, :L]                      # 裁剪到 [0:L]
    return y.reshape(B, H, L).to(input.dtype)


# ============================================================================
# 5b. Batch 配对打包（阶段 5 优化项："RFFT 共轭对称优化"的实际落地形式）
# ============================================================================
def four_step_forward_complex(
    xr: torch.Tensor, xi: torch.Tensor, plan: FourStepPlan, tb: FourStepTables, valid_rows: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """复输入正变换：步骤 A 变成 4 次实数 GEMM（其余与实输入版相同）。"""
    R, N1, N2 = xr.shape[0], plan.N1, plan.N2
    ar = xr.view(R, N1, N2)[:, :valid_rows, :]
    ai = xi.view(R, N1, N2)[:, :valid_rows, :]
    D1r, D1i = tb.D1r[:, :valid_rows], tb.D1i[:, :valid_rows]
    Br = D1r @ ar - D1i @ ai
    Bi = D1r @ ai + D1i @ ar
    Br, Bi = _cmul(Br, Bi, tb.Tr, tb.Ti)
    Cr = Br @ tb.D2r - Bi @ tb.D2i
    Ci = Br @ tb.D2i + Bi @ tb.D2r
    return Cr, Ci


def four_step_inverse_complex(
    Yr: torch.Tensor, Yi: torch.Tensor, plan: FourStepPlan, tb: FourStepTables, out_rows: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """复输出逆变换：步骤 C' 变成 4 次实数 GEMM，实部/虚部分别是两路结果。"""
    Er = Yr @ tb.D2r + Yi @ tb.D2i
    Ei = -Yr @ tb.D2i + Yi @ tb.D2r
    Er, Ei = _cmul(Er, Ei, tb.Tr, -tb.Ti)
    D1r, D1i = tb.D1r[:out_rows, :], tb.D1i[:out_rows, :]
    yr = D1r @ Er + D1i @ Ei          # Re{conj(D1) @ E'}
    yi = D1r @ Ei - D1i @ Er          # Im{conj(D1) @ E'}
    return yr / plan.N, yi / plan.N


def fft_conv1d_four_step_packed(
    input: torch.Tensor,
    kernel: torch.Tensor,
    mode: str = MODE_MATH,
    n_fft: Optional[int] = None,
    plan: Optional[FourStepPlan] = None,
) -> torch.Tensor:
    """
    把**同一通道的两个 batch** 打包成一路复数序列做变换：z = x_{b1} + i·x_{b2}。

    为什么不需要任何拆分（split）步骤：
        Z[k] = X1[k] + i·X2[k]                       （DFT 线性）
        两路共用同一个 kernel 频谱 K̂_h，于是
        Z[k]·K̂_h[k] = X1[k]K̂[k] + i·X2[k]K̂[k] = Y1[k] + i·Y2[k]
        逆变换后 Re = y1，Im = y2。
    正因为"配对的两行共享同一个 kernel"，经典的 Hermitian 拆分/合并被完全省掉，
    也就不需要 conj(Z[N-k]) 这种反序访问（在 Vector 上代价很高）。

    代价对比（每 2 行）：
        不打包：2 x (2+4+4+2) = 24 次实数 GEMM 单位
        打包后：      4+4+4+4  = 16 次实数 GEMM 单位   -> 1.5x
    约束：需要同一通道内至少 2 个 batch。B 为奇数时最后一行补零单独走实输入路径。
    """
    B, H, L, K = check_shapes(input, kernel)
    if plan is None:
        plan = plan_four_step(L, K, n_fft)
    N, N1, N2 = plan.N, plan.N1, plan.N2
    dev = input.device
    tb = FourStepTables.build(plan, torch.float32, dev)

    if B < 2:
        return fft_conv1d_four_step(input, kernel, mode=mode, plan=plan)

    x = input.to(torch.float32)
    k = kernel.to(torch.float32)
    if mode != MODE_MATH:
        k = k.flip(-1)

    k_pad = torch.zeros(H, N, dtype=torch.float32, device=dev)
    k_pad[:, :K] = k
    Kr, Ki = four_step_forward_real(k_pad, plan, tb, plan.R1_ker)      # [H,N1,N2]

    out = torch.zeros(B, H, L, dtype=torch.float32, device=dev)
    npair = B // 2

    if npair > 0:
        # 配对：(b=2p, b=2p+1) 同通道 -> 实部/虚部
        xr = torch.zeros(npair * H, N, dtype=torch.float32, device=dev)
        xi = torch.zeros(npair * H, N, dtype=torch.float32, device=dev)
        xr[:, :L] = x[0:2 * npair:2].reshape(npair * H, L)
        xi[:, :L] = x[1:2 * npair:2].reshape(npair * H, L)

        Zr, Zi = four_step_forward_complex(xr, xi, plan, tb, plan.R1_in)
        Kr_b = Kr.view(1, H, N1, N2).expand(npair, H, N1, N2).reshape(npair * H, N1, N2)
        Ki_b = Ki.view(1, H, N1, N2).expand(npair, H, N1, N2).reshape(npair * H, N1, N2)
        Yr, Yi = _cmul(Zr, Zi, Kr_b, Ki_b)

        y1, y2 = four_step_inverse_complex(Yr, Yi, plan, tb, plan.R1_out)
        y1 = y1.reshape(npair * H, plan.R1_out * N2)[:, :L].reshape(npair, H, L)
        y2 = y2.reshape(npair * H, plan.R1_out * N2)[:, :L].reshape(npair, H, L)
        out[0:2 * npair:2] = y1
        out[1:2 * npair:2] = y2

    if B % 2 == 1:      # 尾巴：单独一行走实输入路径
        out[B - 1:B] = fft_conv1d_four_step(x[B - 1:B], kernel, mode=mode, plan=plan)

    return out.to(input.dtype)


# ============================================================================
# 6. 复杂度 / 搬运量分析
# ============================================================================
@dataclass
class MacCount:
    """一次算子调用的实数 MAC 数与 GM 搬运量。"""
    cube_mac: int = 0          # 走 Cube 的实数乘加次数
    vector_flop: int = 0       # 走 Vector 的实数浮点运算次数
    gm_bytes: int = 0          # GM <-> 片上 的字节数
    detail: dict = field(default_factory=dict)


def count_macs(B: int, H: int, L: int, K: int, plan: FourStepPlan) -> MacCount:
    """精确统计四步方案的 MAC 数（每次 GEMM 的 M*N*K）。"""
    N, N1, N2 = plan.N, plan.N1, plan.N2
    R = B * H
    R1i, R1k, R1o = plan.R1_in, plan.R1_ker, plan.R1_out

    # 每行正变换：步骤 A 2 次 [N1,R1]@[R1,N2]；步骤 C 4 次 [N1,N2]@[N2,N2]
    fwd_x = 2 * (N1 * R1i * N2) + 4 * (N1 * N2 * N2)
    fwd_k = 2 * (N1 * R1k * N2) + 4 * (N1 * N2 * N2)
    # 每行逆变换：步骤 A' 4 次 [N1,N2]@[N2,N2]；步骤 C' 2 次 [R1o,N1]@[N1,N2]
    inv = 4 * (N1 * N2 * N2) + 2 * (R1o * N1 * N2)

    cube = R * (fwd_x + inv) + H * fwd_k

    # Vector：正变换旋转因子 6N/行，逆变换旋转因子 6N/行，频域复乘 6N/行
    vec = R * (6 * N + 6 * N + 6 * N) + H * (6 * N)

    # GM 搬运：读 input + 读 kernel + 写 output + kernel 频谱一读一写
    #（中间结果全部驻留片上，不落 GM —— 见设计文档 §4）
    gm = (R * L + H * K + R * L + 2 * H * N * 2) * 4

    return MacCount(cube_mac=int(cube), vector_flop=int(vec), gm_bytes=int(gm),
                    detail={"fwd_x": fwd_x, "fwd_k": fwd_k, "inv": inv,
                            "N": N, "N1": N1, "N2": N2,
                            "R1_in": R1i, "R1_ker": R1k, "R1_out": R1o})


def estimate_crossover_k(L: int, rho: float) -> int:
    """
    估算 direct conv 与 FFT conv 的性能分界 K*。

    rho = Cube 的 fp32 MAC/cycle ÷ Vector 的 fp32 MAC/cycle（910B 上约 8~32，
          具体值需在阶段 5 用实测标定）。

    direct（Vector）：每行 L*K 次 MAC
    FFT：每行 cube_mac/rho + vector_flop 折算到 Vector 的等效 cycle
    返回使 FFT 开始占优的最小 K。
    """
    for K in range(1, L + 1):
        plan = plan_four_step(L, K)
        mc = count_macs(1, 1, L, K, plan)
        fft_equiv = mc.cube_mac / rho + mc.vector_flop
        direct_equiv = L * K
        if fft_equiv < direct_equiv:
            return K
    return L + 1        # 该 L 下 FFT 从不占优


# ============================================================================
# 自检
# ============================================================================
if __name__ == "__main__":
    from fft_conv1d_reference import direct_conv1d_naive

    torch.manual_seed(0)
    print(f"{'B,H,L,K':>14} {'N':>6} {'N1':>4} {'N2':>4} {'R1in':>5} {'maxerr':>10} "
          f"{'cubeMAC':>12} {'vecFLOP':>10} {'GM KB':>8}")
    print("-" * 92)
    for (B, H, L, K) in [(1, 1, 16, 3), (2, 3, 31, 7), (4, 8, 64, 1),
                         (1, 5, 100, 32), (2, 16, 257, 17), (2, 4, 1024, 128)]:
        x = torch.randn(B, H, L)
        w = torch.randn(H, K)
        plan = plan_four_step(L, K)
        got = fft_conv1d_four_step(x, w, plan=plan)
        ref = direct_conv1d_naive(x, w, dtype=torch.float64).to(torch.float32)
        mc = count_macs(B, H, L, K, plan)
        print(f"{B},{H},{L},{K}".rjust(14)
              + f" {plan.N:>6} {plan.N1:>4} {plan.N2:>4} {plan.R1_in:>5}"
              + f" {(got-ref).abs().max().item():>10.2e}"
              + f" {mc.cube_mac:>12,} {mc.vector_flop:>10,} {mc.gm_bytes/1024:>8.1f}")

    print("\n[1] 分界点 K*（FFT 开始优于 direct 的最小 K；> L 表示该 L 下 FFT 永不占优）")
    print(f"{'L':>8} " + " ".join(f"rho={r:<6}" for r in (8, 16, 32)))
    for L in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
        row = " ".join(f"{estimate_crossover_k(L, r):<10}" for r in (8, 16, 32))
        print(f"{L:>8} {row}")

    print("\n[2] 长卷积场景 K = L（Hyena/S4/FlashFFTConv 的典型区间）加速比")
    print(f"{'L=K':>8} {'N':>7} {'N1':>5} {'N2':>5} {'direct MAC':>13} {'FFT等效(ρ=16)':>15} {'加速比':>9}")
    for L in (256, 512, 1024, 2048, 4096, 8192):
        K = L
        plan = plan_four_step(L, K)
        mc = count_macs(1, 1, L, K, plan)
        fft_eq = mc.cube_mac / 16.0 + mc.vector_flop
        print(f"{L:>8} {plan.N:>7} {plan.N1:>5} {plan.N2:>5} {L*K:>13,} "
              f"{fft_eq:>15,.0f} {L*K/fft_eq:>8.1f}x")

    print("\n[3] batch 配对打包（packed）正确性")
    for (B, H, L, K) in [(2, 3, 64, 16), (4, 2, 128, 32), (5, 3, 100, 20), (2, 16, 257, 17)]:
        x = torch.randn(B, H, L)
        w = torch.randn(H, K)
        ref = direct_conv1d_naive(x, w, dtype=torch.float64).to(torch.float32)
        p = fft_conv1d_four_step_packed(x, w)
        print(f"  B={B},H={H},L={L},K={K}: packed vs direct maxerr = "
              f"{(p-ref).abs().max().item():.2e}")
