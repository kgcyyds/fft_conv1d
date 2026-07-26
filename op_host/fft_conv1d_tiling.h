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
constexpr uint32_t FFT_CONV1D_ALGO_FFT = 1;    // 四步 Cooley-Tukey（Cube + Vector）

// v1 约束（同时写入 host 校验与文档，不做隐式假设）
constexpr uint32_t FFT_CONV1D_MAX_NFFT = 16384;  // N_fft 上限 => N1 = 128
constexpr uint32_t FFT_CONV1D_FFT_MIN_K = 64;    // K 小于该值走 direct（见设计文档 §9）
constexpr uint32_t FFT_CONV1D_DIRECT_TILE = 4096; // direct 路径的输出分块长度

// FFT 路径每个核需要的 scratch 缓冲个数（每个长度 N_fft，float32）
// 0:xmat 1:br 2:bi 3:t0 4:t1 5:cr 6:ci 7:er 8:ei 9:yout
constexpr uint32_t FFT_CONV1D_SCRATCH_BUFS = 10;

BEGIN_TILING_DATA_DEF(FftConv1dTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);      // B
TILING_DATA_FIELD_DEF(uint32_t, channel);    // H
TILING_DATA_FIELD_DEF(uint32_t, seqLen);     // L
TILING_DATA_FIELD_DEF(uint32_t, kernelLen);  // K
TILING_DATA_FIELD_DEF(uint32_t, totalRows);  // R = B*H
TILING_DATA_FIELD_DEF(uint32_t, algo);       // FFT_CONV1D_ALGO_*
TILING_DATA_FIELD_DEF(uint32_t, flipKernel); // 0=math 语义, 1=corr 语义（时域翻转 kernel）
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
