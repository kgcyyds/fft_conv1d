"""
fft_conv1d 阶段 1：语义定义与参考实现（纯 PyTorch，CPU）

算子语义（以数学公式为准）
--------------------------------
输入  input : [B, H, L]
      kernel: [H, K]
输出  output: [B, H, L]

因果 depthwise Conv1D（每个通道 h 独立，无通道间求和）：

    output[b, h, t] = sum_{j=0}^{K-1} input[b, h, t - j] * kernel[h, j],  t = 0..L-1
    其中 t - j < 0 时 input 视为 0

注意：这是**数学意义上的卷积**（convolution），不是 cross-correlation。
      即 kernel[h, 0] 与"当前时刻" input[b,h,t] 相乘，kernel[h,K-1] 与最旧的
      input[b,h,t-K+1] 相乘。

两种模式
--------------------------------
本文件把两种可能的语义都实现出来，由 mode 参数选择：

  mode="math"  : output[t] = sum_j input[t-j] * kernel[j]          <-- 题目数学公式（默认，权威）
  mode="corr"  : output[t] = sum_j input[t-j] * kernel[K-1-j]      <-- 题目给出的 PyTorch 片段实际算的

两者互为 kernel 时间翻转，K == 1 或 kernel 对称时才相等。详见 docs/01_semantics_and_fft_derivation.md。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "next_pow2",
    "choose_n_fft",
    "check_shapes",
    "fft_conv1d_reference_original",
    "direct_conv1d_naive",
    "direct_conv1d_torch",
    "fft_conv1d_torch",
    "fft_conv1d_manual_dft",
    "complex_mul",
]

MODE_MATH = "math"
MODE_CORR = "corr"


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def next_pow2(n: int) -> int:
    """返回 >= n 的最小 2 的幂。"""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def choose_n_fft(L: int, K: int, n_fft: Optional[int] = None) -> int:
    """
    选择 FFT 长度。

    线性卷积无混叠的充要条件： N_fft >= L + K - 1
    首版本约束： N_fft = 2^ceil(log2(L+K-1))，便于 radix-2/4/16 分解与对齐。
    另外强制 N_fft >= 2：N=1 时 RFFT 退化（不存在 Nyquist 频点），
    硬件实现也不会走这条路径，统一抬到 2 不影响正确性（仍满足 N >= L+K-1）。
    """
    need = L + K - 1
    if n_fft is None:
        return max(2, next_pow2(need))
    if n_fft < need:
        raise ValueError(f"n_fft={n_fft} < L+K-1={need}，会产生循环卷积混叠")
    return n_fft


def check_shapes(input: torch.Tensor, kernel: torch.Tensor) -> Tuple[int, int, int, int]:
    """校验 shape 约束，返回 (B, H, L, K)。约束与算子文档保持一致。"""
    if input.dim() != 3:
        raise ValueError(f"input 必须是 3 维 [B,H,L]，实际 {tuple(input.shape)}")
    if kernel.dim() != 2:
        raise ValueError(f"kernel 必须是 2 维 [H,K]，实际 {tuple(kernel.shape)}")
    B, H, L = input.shape
    Hk, K = kernel.shape
    if H != Hk:
        raise ValueError(f"通道数不一致：input H={H}, kernel H={Hk}")
    if B < 1 or H < 1 or L < 1 or K < 1:
        raise ValueError("B/H/L/K 必须 >= 1")
    if K > L:
        raise ValueError(f"当前约束要求 K <= L，实际 K={K}, L={L}")
    return B, H, L, K


def _mode_kernel(kernel: torch.Tensor, mode: str) -> torch.Tensor:
    """
    把 kernel 统一归一到 "math" 语义下使用的抽头顺序。

    mode="math": 直接用 kernel
    mode="corr": 等价于用时间翻转后的 kernel 做数学卷积（推导见文档 §2.3）
    """
    if mode == MODE_MATH:
        return kernel
    if mode == MODE_CORR:
        return kernel.flip(-1)
    raise ValueError(f"未知 mode: {mode}")


# ----------------------------------------------------------------------------
# 0) 题目原始给出的参考实现（原样保留，用于对照它到底算的是什么）
# ----------------------------------------------------------------------------
def fft_conv1d_reference_original(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    题目中给出的 PyTorch 片段，原样照抄。
    实测语义为： output[t] = sum_j input[t-j] * kernel[K-1-j]  （即 mode="corr"）
    原因：F.conv1d 做的是 cross-correlation，而这里没有对 kernel 做翻转。
    """
    B, H, L = input.shape
    Hk, K = kernel.shape
    assert H == Hk

    x = F.pad(input, (K - 1, 0))
    weight = kernel.unsqueeze(1)

    return F.conv1d(x, weight, bias=None, stride=1, padding=0, groups=H)


# ----------------------------------------------------------------------------
# 1) 直接卷积参考实现（ground truth，逐字翻译数学公式）
# ----------------------------------------------------------------------------
def direct_conv1d_naive(
    input: torch.Tensor,
    kernel: torch.Tensor,
    mode: str = MODE_MATH,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    最朴素的实现：完全按公式 output[b,h,t] = sum_{j} input[b,h,t-j]*k[h,j] 累加。
    只在 t >= j 时累加，等价于 t-j<0 时 input 取 0。
    这是本项目的"金标准"，其它所有实现都必须和它对齐。
    """
    B, H, L, K = check_shapes(input, kernel)
    k = _mode_kernel(kernel, mode)

    compute_dtype = dtype if dtype is not None else input.dtype
    x = input.to(compute_dtype)
    k = k.to(compute_dtype)

    out = torch.zeros((B, H, L), dtype=compute_dtype, device=input.device)
    for j in range(K):
        # t 从 j 开始才有效：out[:, :, t] += x[:, :, t-j] * k[:, j]
        out[:, :, j:] += x[:, :, : L - j] * k[:, j].view(1, H, 1)
    return out.to(input.dtype)


# ----------------------------------------------------------------------------
# 2) 用 F.conv1d 的直接卷积实现（高效 direct 路径，也是 NPU 小尺寸分支的语义基准）
# ----------------------------------------------------------------------------
def direct_conv1d_torch(
    input: torch.Tensor,
    kernel: torch.Tensor,
    mode: str = MODE_MATH,
) -> torch.Tensor:
    """
    用 F.conv1d（cross-correlation）实现数学卷积：
      左侧 pad K-1 个 0，weight 取 kernel 的时间翻转，groups=H。
    推导见文档 §2.2。
    """
    B, H, L, K = check_shapes(input, kernel)

    # F.conv1d 是 correlation：out[t] = sum_j x[t+j]*w[j]，x[p]=input[p-(K-1)]
    #   => out[t] = sum_j input[t+j-K+1]*w[j]，令 j'=K-1-j
    #   => out[t] = sum_{j'} input[t-j']*w[K-1-j']
    # 要得到 mode="math"（乘 kernel[j']），需令 w[K-1-j'] = kernel[j']，即 w = kernel.flip(-1)。
    w = kernel if mode == MODE_CORR else kernel.flip(-1)

    x = F.pad(input, (K - 1, 0))
    weight = w.unsqueeze(1).to(input.dtype)
    return F.conv1d(x, weight, bias=None, stride=1, padding=0, groups=H)


# ----------------------------------------------------------------------------
# 3) FFT 卷积参考实现（torch.fft.rfft / irfft）
# ----------------------------------------------------------------------------
def fft_conv1d_torch(
    input: torch.Tensor,
    kernel: torch.Tensor,
    mode: str = MODE_MATH,
    n_fft: Optional[int] = None,
    compute_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    FFT 线性卷积：
      1. input 右侧补零到 N_fft
      2. kernel 按语义决定是否翻转，然后右侧补零到 N_fft
      3. 分别 rfft（得到 N_fft/2+1 个频点）
      4. 逐频点复数乘
      5. irfft(n=N_fft)
      6. 裁剪 [0:L]   <-- 因果卷积，取线性卷积结果的前 L 个点，无任何偏移

    关键点：频域乘法实现的就是**数学卷积**，所以 mode="math" 时 kernel 不翻转；
            要复现题目原片段（correlation）时才需要 kernel.flip(-1)。
    """
    B, H, L, K = check_shapes(input, kernel)
    N = choose_n_fft(L, K, n_fft)

    orig_dtype = input.dtype
    cdt = compute_dtype if compute_dtype is not None else torch.float32

    x = input.to(cdt)
    k = _mode_kernel(kernel, mode).to(cdt)

    # 补零到 N（右侧补零 = 时域后面接 0，不改变前 L/K 个样本的时间原点）
    x_pad = F.pad(x, (0, N - L))                 # [B,H,N]
    k_pad = F.pad(k, (0, N - K))                 # [H,N]

    Xf = torch.fft.rfft(x_pad, n=N, dim=-1)      # [B,H,N/2+1] complex
    Kf = torch.fft.rfft(k_pad, n=N, dim=-1)      # [H,N/2+1]   complex

    Yf = Xf * Kf.unsqueeze(0)                    # 逐频点复数乘（广播 batch）

    y = torch.fft.irfft(Yf, n=N, dim=-1)         # [B,H,N] 实数，torch 默认 backward 归一化(1/N)

    return y[..., :L].to(orig_dtype)             # 裁剪：线性卷积前 L 点


# ----------------------------------------------------------------------------
# 4) 复数乘法：4 次实乘 vs 3 次实乘（Karatsuba）
# ----------------------------------------------------------------------------
def complex_mul(ar, ai, br, bi, method: str = "4mul"):
    """
    (ar + i*ai) * (br + i*bi) = (ar*br - ai*bi) + i*(ar*bi + ai*br)

    method="4mul": 4 次乘 + 2 次加减，数值最稳，向量指令直接对应 mul/mla。
    method="3mul": Karatsuba，3 次乘 + 5 次加减：
        t1 = ar*br;  t2 = ai*bi;  t3 = (ar+ai)*(br+bi)
        re = t1 - t2;  im = t3 - t1 - t2
      乘法少 25%，但加法多，且 t3 - t1 - t2 存在抵消误差放大。
    """
    if method == "4mul":
        return ar * br - ai * bi, ar * bi + ai * br
    if method == "3mul":
        t1 = ar * br
        t2 = ai * bi
        t3 = (ar + ai) * (br + bi)
        return t1 - t2, t3 - t1 - t2
    raise ValueError(f"未知 method: {method}")


# ----------------------------------------------------------------------------
# 5) 手写 DFT 矩阵版（实部/虚部分离），对齐未来 AscendC 的数据表示与计算路径
# ----------------------------------------------------------------------------
def _dft_matrices(N: int, dtype: torch.dtype, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成 RFFT 用的实数 DFT 矩阵，只保留 M = N/2+1 个频点。
      C[k,n] =  cos(2*pi*k*n/N)
      S[k,n] =  sin(2*pi*k*n/N)
    则  X[k] = sum_n x[n] e^{-i2πkn/N} => Xr = x @ C^T,  Xi = -(x @ S^T)
    注意：角度用 float64 生成再转目标 dtype，避免旋转因子本身引入误差。
    """
    M = N // 2 + 1
    k = torch.arange(M, dtype=torch.float64, device=device).view(M, 1)
    n = torch.arange(N, dtype=torch.float64, device=device).view(1, N)
    theta = 2.0 * math.pi * k * n / N
    return torch.cos(theta).to(dtype), torch.sin(theta).to(dtype)


def fft_conv1d_manual_dft(
    input: torch.Tensor,
    kernel: torch.Tensor,
    mode: str = MODE_MATH,
    n_fft: Optional[int] = None,
    compute_dtype: torch.dtype = torch.float32,
    cmul_method: str = "4mul",
) -> torch.Tensor:
    """
    用显式 DFT 矩阵乘 + 实虚分离复数乘 + 显式 IRFFT 公式实现，
    完全不依赖 torch.fft，用来验证未来 AscendC kernel 的每一步数学。

    IRFFT（N 为偶数，Hermitian 对称）：
      y[n] = (1/N) * [ Yr[0] + (-1)^n * Yr[N/2]
                       + 2 * sum_{k=1}^{N/2-1} ( Yr[k]cos(2πkn/N) - Yi[k]sin(2πkn/N) ) ]
    """
    B, H, L, K = check_shapes(input, kernel)
    N = choose_n_fft(L, K, n_fft)
    if N % 2 != 0:
        raise ValueError("手写实现要求 N_fft 为偶数（首版本固定为 2 的幂）")

    dev = input.device
    orig_dtype = input.dtype
    M = N // 2 + 1

    C, S = _dft_matrices(N, compute_dtype, dev)  # [M,N]

    x = F.pad(input.to(compute_dtype), (0, N - L))                     # [B,H,N]
    k = F.pad(_mode_kernel(kernel, mode).to(compute_dtype), (0, N - K))  # [H,N]

    # ---- RFFT（forward，无归一化）----
    Xr = x @ C.t()          # [B,H,M]
    Xi = -(x @ S.t())
    Kr = k @ C.t()          # [H,M]
    Ki = -(k @ S.t())

    # ---- 逐频点复数乘 ----
    Yr, Yi = complex_mul(Xr, Xi, Kr.unsqueeze(0), Ki.unsqueeze(0), method=cmul_method)

    # DC(k=0) 与 Nyquist(k=N/2) 对实信号而言虚部恒为 0，乘积虚部也应为 0；
    # 这里显式清零，避免浮点残差污染 IRFFT（硬件实现同样可以省掉这两点的虚部通路）。
    Yi[..., 0] = 0
    Yi[..., M - 1] = 0

    # ---- IRFFT ----
    # 输出时间点只需要前 L 个（因果裁剪），但为了验证完整性这里先算全 N 再裁剪。
    n_idx = torch.arange(N, dtype=torch.float64, device=dev).view(1, N)
    k_idx = torch.arange(M, dtype=torch.float64, device=dev).view(M, 1)
    theta = 2.0 * math.pi * k_idx * n_idx / N
    Ci = torch.cos(theta).to(compute_dtype)   # [M,N]
    Si = torch.sin(theta).to(compute_dtype)

    # 权重：k=0 和 k=N/2 权重 1，其余权重 2（共轭对称合并）
    w = torch.full((M,), 2.0, dtype=compute_dtype, device=dev)
    w[0] = 1.0
    w[M - 1] = 1.0

    y = (Yr * w) @ Ci - (Yi * w) @ Si          # [B,H,N]
    y = y / N                                  # backward 归一化，与 torch.fft.irfft 一致

    return y[..., :L].to(orig_dtype)


# ----------------------------------------------------------------------------
# 自检入口
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    shapes = [(1, 1, 16, 3), (2, 3, 31, 7), (4, 8, 64, 1), (1, 5, 100, 32), (2, 16, 257, 17)]
    print(f"{'B,H,L,K':>16} | {'N_fft':>6} | {'orig==corr':>10} | "
          f"{'fft-direct(math)':>16} | {'dft-direct(math)':>16}")
    print("-" * 90)
    for B, H, L, K in shapes:
        x = torch.randn(B, H, L)
        k = torch.randn(H, K)
        N = choose_n_fft(L, K)

        ref_orig = fft_conv1d_reference_original(x, k)
        d_corr = direct_conv1d_naive(x, k, mode=MODE_CORR)
        d_math = direct_conv1d_naive(x, k, mode=MODE_MATH)
        f_math = fft_conv1d_torch(x, k, mode=MODE_MATH)
        m_math = fft_conv1d_manual_dft(x, k, mode=MODE_MATH)

        e_orig = (ref_orig - d_corr).abs().max().item()
        e_fft = (f_math - d_math).abs().max().item()
        e_dft = (m_math - d_math).abs().max().item()
        print(f"{B},{H},{L},{K:>3}".rjust(16) + f" | {N:>6} | {e_orig:>10.2e} | "
              f"{e_fft:>16.2e} | {e_dft:>16.2e}")
