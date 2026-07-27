/**
 * fft_conv1d TilingData 定义
 *
 * 语义（唯一语义，不再有 flip 分支）：
 *   output[b,h,t] = sum_{j=0}^{K-1} input[b,h,t-j] * kernel[h,j]，t-j<0 时 input 视为 0
 *
 * 约束见 docs/01_semantics_and_fft_derivation.md
 */
#ifndef FFT_CONV1D_TILING_H
#define FFT_CONV1D_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling
{

// 算法分支：host 侧根据 K 静态选择，kernel 侧不做动态判断
constexpr uint32_t FFT_CONV1D_ALGO_DIRECT = 0; // 直接因果卷积（Vector）
constexpr uint32_t FFT_CONV1D_ALGO_FFT = 1;    // 四步 Cooley-Tukey（Cube + Vector）

// v1 约束（同时写入 host 校验与文档，不做隐式假设）
constexpr uint32_t FFT_CONV1D_MAX_NFFT = 16384; // N_fft 上限 => N1 = 128
constexpr uint32_t FFT_CONV1D_FFT_MIN_K = 64;   // K 小于该值走 direct

// direct 路径输出分块长度。
// 取 1024 而非更大值：direct 用 8 份移位副本消除 Vector 非对齐访问，
// UB 占用为 8*(tile+K)*4B，tile=1024/K<64 时约 35KB，留足余量。
constexpr uint32_t FFT_CONV1D_DIRECT_TILE = 1024;

// FFT 路径常量表个数（每个长度 N_fft，float32）：
//   Dr, Di, Dn(=-Di), DrS(=Dr/N), DiS(=Di/N), Tr, Ti
// Dn/DrS/DiS 是为了让 AtomicAdd 累加能吃掉减法和归一化，从而消除所有 Vector 合并 pass
constexpr uint32_t FFT_CONV1D_TABLES = 7;

// FFT 路径每个核的 scratch 缓冲个数（每个长度 N_fft，float32）
//   0:xmat 1:Br 2:Bi 3:Cr 4:Ci 5:Er 6:Ei 7:yout
constexpr uint32_t FFT_CONV1D_SCRATCH_BUFS = 8;

BEGIN_TILING_DATA_DEF(FftConv1dTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);      // B
TILING_DATA_FIELD_DEF(uint32_t, channel);    // H
TILING_DATA_FIELD_DEF(uint32_t, seqLen);     // L
TILING_DATA_FIELD_DEF(uint32_t, kernelLen);  // K
TILING_DATA_FIELD_DEF(uint32_t, totalRows);  // R = B*H
TILING_DATA_FIELD_DEF(uint32_t, algo);       // FFT_CONV1D_ALGO_*
TILING_DATA_FIELD_DEF(uint32_t, nFft);       // N（4 的幂）
TILING_DATA_FIELD_DEF(uint32_t, nRadix);     // N1 = N2 = sqrt(N)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, tileLen);    // direct 路径输出分块长度
TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FftConv1d, FftConv1dTilingData)

} // namespace optiling

#endif // FFT_CONV1D_TILING_H
