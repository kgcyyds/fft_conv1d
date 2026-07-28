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

// 调试开关：强制单核（blockDim = 1）。
// 多核路径（UB 版 + Cube）已实测验证通过，因此默认关闭。
// 保留这个开关是因为它在定位问题时价值很高：一旦怀疑某个 bug 与多核分片有关，
// 打开它就能一次性排除 scratch 分片、跨核共享、栅栏计数三类因素。
// 注意：单核 != 单执行单元。MIX 模式下 blockDim=1 仍有 1 个 AIC + 2 个 AIV，
//       AIV 的 GetBlockIdx() 仍会取到 0 和 1，所以 CoreIdx() 的除法始终必要。
constexpr uint32_t FFT_CONV1D_FORCE_SINGLE_CORE = 0;

// v1 约束（同时写入 host 校验与文档，不做隐式假设）
// FFT 路径 N_fft 上限。重写版把中间结果全部放 UB：12 个长度 N 的缓冲，
// N=1024 时 48KB（很宽裕），N=4096 时 192KB（放不下）。故上限取 1024。
// 超出时 host 自动回退 DIRECT —— DIRECT 对任意 K 数值都正确，功能覆盖不减。
constexpr uint32_t FFT_CONV1D_MAX_NFFT_UB = 1024;  // UB 常驻上限 => N1 = 32（48KB）
// GM 版上限：重设计后 Cube 永不碰 GM，运算整块经 UB 中转，
// 复数逐点乘需同时驻留 6 个长度 N 的 UB 缓冲 => N=4096 时 6*16KB=96KB 为上限。
// 超出由 host 回退 DIRECT（数值仍正确）。
// FFT-GM 开关。0 = 关闭（N>1024 的 shape 回退 DIRECT，数值正确但较慢）
//              1 = 启用
// 当前置 0：FFT-GM 尚未验证通过，关闭后算子对所有 shape 都是正确的。
// 代码保留在 op_kernel 里，拿到观测手段后再打开排查。
constexpr uint32_t FFT_CONV1D_ENABLE_GM = 0;

constexpr uint32_t FFT_CONV1D_MAX_NFFT =
    FFT_CONV1D_ENABLE_GM ? 4096u : FFT_CONV1D_MAX_NFFT_UB; // GM 版上限 => N1 = 64
constexpr uint32_t FFT_CONV1D_GM_BUFS = 12;        // GM 版每核缓冲个数，与 kernel 一致
constexpr uint32_t FFT_CONV1D_FFT_MIN_K = 64;    // K 小于该值走 direct（见设计文档 §9）
constexpr uint32_t FFT_CONV1D_DIRECT_TILE = 4096; // direct 路径的输出分块长度

// 重写后 FFT 路径全部数据常驻 UB，不再需要 GM scratch

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
