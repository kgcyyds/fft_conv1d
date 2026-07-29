/**
 * fft_conv1d TilingData 定义
 *
 * 语义与约束见 docs/01_semantics_and_fft_derivation.md、docs/03_ascendc_v1.md
 */
#ifndef FFT_CONV1D_TILING_H
#define FFT_CONV1D_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling
{

// 算法分支：host 侧根据 K 与 L 静态选择，kernel 侧不做动态判断
constexpr uint32_t FFT_CONV1D_ALGO_DIRECT = 0; // 直接因果卷积（Vector）
constexpr uint32_t FFT_CONV1D_ALGO_FFT = 1;     // UB 常驻（N <= FFT_CONV1D_MAX_NFFT_UB）
constexpr uint32_t FFT_CONV1D_ALGO_FFT_GM = 2;  // GM 版（更大的 N）

// ---------------------------------------------------------------------------
// 算法分派的边界常量（都由硬件容量推导，不是经验值）
// ---------------------------------------------------------------------------
// N_fft 取 >= L+K-1 的最小 4 的幂，故 N1 = N2 = sqrt(N_fft)。
//
// FFT-UB：12 个长度 N 的缓冲常驻 UB（Dr Di Tr Ti Kfr Kfi Xr Xi Yr Yi Zr Zi）
//         N=1024 -> 12*1024*4 = 48KB，很宽裕；N=4096 -> 192KB，放不下。
constexpr uint32_t FFT_CONV1D_MAX_NFFT_UB = 1024;   // => N1 = 32
//
// FFT-GM：数据放 GM；逐点运算按 VEC_CHUNK(2048) 分块经 UB，
//         矩阵乘 A/B/C 全在 GM（Cube 直接读写，不占 UB）。
//         => UB 占用恒为 6*2048*4 + 8KB tmp + 4*N1*4 ≈ 60KB，**与 N 无关**。
//         上限改由这三条决定，取最紧的一条：
//           a) N1 = sqrt(N) 要能被 Matmul tiling 接受（N=16384 -> N1=128，可行）
//           b) Row() 逐行生成时 N1 <= VEC_CHUNK（128 << 2048，宽裕）
//           c) workspace = 12 * N * 核数 * 4B（N=16384、20 核约 15.7MB）
// constexpr uint32_t FFT_CONV1D_MAX_NFFT_GM = 16384;  // => N1 = 128
//
// 超过 FFT_CONV1D_MAX_NFFT_GM 的 shape 回退 DIRECT：数值仍然正确，只是更慢，
// 因此算子支持的 shape 范围不受这些容量上限影响。
// K 小于该值时直接卷积更快（DIRECT 为 O(L*K) Vector，FFT 为 O(12*N*N1) Cube）。
// 该分界点来自 docs/02 §9 的代价模型估算，尚未用实测标定。
constexpr uint32_t FFT_CONV1D_FFT_MIN_K = 64;
constexpr uint32_t FFT_CONV1D_DIRECT_TILE = 4096; // direct 路径的输出分块长度

// FFT-UB 全部数据常驻 UB，不需要用户 workspace。
// FFT-GM 每个 AIV worker 在 workspace 上占用下列 12 个长度 N_fft 的缓冲：
//   Dr Di Tr Ti  Kfr Kfi  Xr Xi  Yr Yi  Zr Zi
// 必须与 op_kernel 里的 FFT_GM_BUFFER_COUNT 保持一致（两侧共同决定
// workspace 的分片布局）。定义放在本头文件中，Host 编译单元包含后可直接使用。
constexpr uint32_t FFT_CONV1D_GM_BUFFER_COUNT = 12U;

BEGIN_TILING_DATA_DEF(FftConv1dTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);      // B
TILING_DATA_FIELD_DEF(uint32_t, channel);    // H
TILING_DATA_FIELD_DEF(uint32_t, seqLen);     // L
TILING_DATA_FIELD_DEF(uint32_t, kernelLen);  // K
TILING_DATA_FIELD_DEF(uint32_t, totalRows);  // R = B*H
TILING_DATA_FIELD_DEF(uint32_t, algo);       // FFT_CONV1D_ALGO_*
// TILING_DATA_FIELD_DEF(uint32_t, flipKernel); // 0=math 语义, 1=corr 语义（时域翻转 kernel）
TILING_DATA_FIELD_DEF(uint32_t, nFft);       // N（4 的幂）
TILING_DATA_FIELD_DEF(uint32_t, nRadix);     // N1 = N2 = sqrt(N)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, tileLen);    // direct 路径输出分块长度
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling); // 供 kernel 的 Cube 开关使用
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FftConv1d, FftConv1dTilingData)

} // namespace optiling

#endif // FFT_CONV1D_TILING_H
