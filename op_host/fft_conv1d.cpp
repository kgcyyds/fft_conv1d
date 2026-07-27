/**
 * fft_conv1d host 侧：shape 校验、tiling、算子注册
 *
 * 算子语义（唯一语义，详见 docs/01_semantics_and_fft_derivation.md）：
 *   output[b,h,t] = sum_{j=0}^{K-1} input[b,h,t-j] * kernel[h,j],  t-j<0 时 input 视为 0
 *   即因果 depthwise 卷积，每个通道独立，不做通道间求和。
 *
 * 注意：这是数学意义的卷积，不是 cross-correlation。若调用方的参考实现写成
 *   F.conv1d(F.pad(x,(K-1,0)), kernel.unsqueeze(1), groups=H)
 * 那算的是 kernel 时域翻转后的结果，需要自行对 kernel 做 flip(-1) 后再调用本算子。
 */
#include <cstdio>
#include <cstring>

#include "fft_conv1d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace
{
// 返回 >= n 的最小 4 的幂（最小为 4）
inline uint32_t NextPow4(uint32_t n)
{
    uint32_t v = 4;
    while (v < n && v <= (1U << 30))
    {
        v <<= 2;
    }
    return v;
}

// n 为 4 的幂时返回 sqrt(n)
inline uint32_t IsqrtPow4(uint32_t n)
{
    uint32_t r = 1;
    while (r * r < n)
    {
        r <<= 1;
    }
    return r;
}

inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}
} // namespace

namespace optiling
{

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // ---------------- 1. 取 shape 并校验 ----------------
    // 说明：tiling 失败只会向上暴露一个 "ret is -1"，不带原因，
    // 所以每一处校验都必须自己打印出失败理由，否则定位成本极高。
    auto xTensor = context->GetInputTensor(0);
    auto kTensor = context->GetInputTensor(1);
    if (xTensor == nullptr || kTensor == nullptr)
    {
        printf("[fft_conv1d tiling] 取输入 tensor 失败\n");
        return ge::GRAPH_FAILED;
    }
    auto xShape = xTensor->GetOriginShape();
    auto kShape = kTensor->GetOriginShape();

    // input 必须是 [B,H,L]，kernel 必须是 [H,K]
    if (xShape.GetDimNum() != 3 || kShape.GetDimNum() != 2)
    {
        printf("[fft_conv1d tiling] 维度错误: x 应为 3 维(实际 %zu)，kernel 应为 2 维(实际 %zu)\n",
               xShape.GetDimNum(), kShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    const uint32_t B = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t H = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t L = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t Hk = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t K = static_cast<uint32_t>(kShape.GetDim(1));

    if (B < 1 || H < 1 || L < 1 || K < 1)
    {
        printf("[fft_conv1d tiling] B,H,L,K 必须 >=1，实际 B=%u H=%u L=%u K=%u\n", B, H, L, K);
        return ge::GRAPH_FAILED;
    }
    if (H != Hk)
    {
        printf("[fft_conv1d tiling] 通道数不一致: x 的 H=%u, kernel 的 H=%u\n", H, Hk);
        return ge::GRAPH_FAILED;
    }
    if (K > L)
    {
        printf("[fft_conv1d tiling] 约束要求 K<=L，实际 K=%u L=%u\n", K, L);
        return ge::GRAPH_FAILED;
    }
    // v1 约束：线性卷积长度不得超过 N_fft 上限
    if (L + K - 1 > FFT_CONV1D_MAX_NFFT)
    {
        printf("[fft_conv1d tiling] v1 约束 L+K-1<=%u，实际 %u (L=%u K=%u)\n",
               FFT_CONV1D_MAX_NFFT, L + K - 1, L, K);
        return ge::GRAPH_FAILED;
    }

    // ---------------- 2. FFT 长度与分解 ----------------
    // v1 取 N 为 4 的幂 => N1 = N2 = sqrt(N)，D1 与 D2 是同一个矩阵，
    // 且四步分解中 12 次 GEMM 形状统一为 (N1, N1, N1)，共用一套 TCubeTiling。
    // 取舍理由见 docs/03_ascendc_v1.md §2。
    const uint32_t need = (L + K - 1) < 2 ? 2 : (L + K - 1);
    const uint32_t nFft = NextPow4(need);
    const uint32_t nRadix = IsqrtPow4(nFft);

    // ---------------- 3. 算法分派 ----------------
    // K 较小时 direct 更快（分界点分析见 docs/02_algorithm_design.md §9）
    const uint32_t algo = (K < FFT_CONV1D_FFT_MIN_K) ? FFT_CONV1D_ALGO_DIRECT
                                                     : FFT_CONV1D_ALGO_FFT;

    // ---------------- 4. 多核切分：按 R = B*H 行切 ----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    if (aicNum == 0)
    {
        aicNum = 1;
    }
    const uint32_t totalRows = B * H;
    uint32_t usedCoreNum = (totalRows < aicNum) ? totalRows : aicNum;
    if (usedCoreNum == 0)
    {
        usedCoreNum = 1;
    }
    const uint32_t rowsPerCore = CeilDiv(totalRows, usedCoreNum);

    // direct 路径的输出分块长度。移位副本方案下 tile 与 K 无耦合，直接取上限即可。
    const uint32_t tileLen = (L < FFT_CONV1D_DIRECT_TILE) ? L : FFT_CONV1D_DIRECT_TILE;

    // ---------------- 5. Matmul tiling（单核视角，每个核算自己的 GEMM）----------------
    // 只有 FFT 路径才需要 cube tiling。DIRECT 路径是纯 Vector 实现，
    // 且此时 N1 可能小到 2/8（例如 L=16,K=3 => N=64,N1=8），
    // 小于 fp32 Cube 的最小分形，GetTiling 必然返回 -1 —— 不能无条件调用。
    FftConv1dTilingData tilingData;
    if (algo == FFT_CONV1D_ALGO_FFT)
    {
        // FFT 路径下 K>=64 且 K<=L，故 L+K-1>=127 => N>=256 => N1>=16，形状合法
        MatmulApiTiling mmTiling(ascendcPlatform);
        mmTiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        mmTiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        mmTiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        mmTiling.SetShape(static_cast<int32_t>(nRadix), static_cast<int32_t>(nRadix),
                          static_cast<int32_t>(nRadix));
        mmTiling.SetOrgShape(static_cast<int32_t>(nRadix), static_cast<int32_t>(nRadix),
                             static_cast<int32_t>(nRadix));
        mmTiling.SetBias(false);
        mmTiling.SetBufferSpace(-1, -1, -1);
        if (mmTiling.GetTiling(tilingData.cubeTiling) == -1)
        {
            printf("[fft_conv1d tiling] MatmulApiTiling GetTiling 失败: N1=%u (N=%u)\n",
                   nRadix, nFft);
            return ge::GRAPH_FAILED;
        }
    }
    else
    {
        // DIRECT 路径不用 cube，但 tilingData 是栈对象，
        // 清零避免把未初始化字节写进 tiling blob（tiling 会被缓存/比对）
        memset(&tilingData.cubeTiling, 0, sizeof(TCubeTiling));
    }

    // ---------------- 6. 填 TilingData ----------------
    tilingData.set_batch(B);
    tilingData.set_channel(H);
    tilingData.set_seqLen(L);
    tilingData.set_kernelLen(K);
    tilingData.set_totalRows(totalRows);
    tilingData.set_algo(algo);
    tilingData.set_nFft(nFft);
    tilingData.set_nRadix(nRadix);
    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_tileLen(tileLen);
    tilingData.set_rowsPerCore(rowsPerCore);

    printf("[fft_conv1d tiling] B=%u H=%u L=%u K=%u | algo=%s N=%u N1=%u "
           "cores=%u rowsPerCore=%u tileLen=%u\n",
           B, H, L, K,
           (algo == FFT_CONV1D_ALGO_FFT) ? "FFT" : "DIRECT",
           nFft, nRadix, usedCoreNum, rowsPerCore, tileLen);

    context->SetBlockDim(usedCoreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    // ---------------- 7. workspace ----------------
    // FFT 路径布局（单位：float）：
    //   [0]  Dr  [N]  Di  [2N] Dn(=-Di)  [3N] DrS(=Dr/N)  [4N] DiS(=Di/N)
    //   [5N] Tr  [6N] Ti
    //   [7N]       Kfr  : H*N
    //   [7N+HN]    Kfi  : H*N
    //   [7N+2HN]   每核 scratch : usedCoreNum * SCRATCH_BUFS * N
    // direct 路径布局：每核一份零前缀输入行 (K-1+L)，按 8 对齐
    size_t userWorkspace = 0;
    if (algo == FFT_CONV1D_ALGO_FFT)
    {
        const size_t n = static_cast<size_t>(nFft);
        userWorkspace = (FFT_CONV1D_TABLES * n + 2 * static_cast<size_t>(H) * n +
                         static_cast<size_t>(usedCoreNum) * FFT_CONV1D_SCRATCH_BUFS * n) *
                        sizeof(float);
    }
    else
    {
        const size_t rowLen = ((static_cast<size_t>(K) - 1 + L + 7) / 8) * 8;
        userWorkspace = static_cast<size_t>(usedCoreNum) * rowLen * sizeof(float);
    }
    const size_t sysWorkspace = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspace + sysWorkspace;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge
{
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // output 与 input 同 shape：[B,H,L]
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr)
    {
        return GRAPH_FAILED;
    }
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops
{
class FftConv1d : public OpDef
{
  public:
    explicit FftConv1d(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("kernel")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FftConv1d);
} // namespace ops
