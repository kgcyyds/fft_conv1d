/**
 * fft_conv1d host 侧：shape 校验、tiling、算子注册
 *
 * 算子语义（以数学公式为准，详见 docs/01_semantics_and_fft_derivation.md）：
 *   output[b,h,t] = sum_{j=0}^{K-1} input[b,h,t-j] * kernel[h,j],  t-j<0 时 input 视为 0
 *   即因果 depthwise 卷积，每个通道独立，不做通道间求和。
 *
 * 属性 flip_kernel：
 *   0（默认）= math 语义，与上式一致
 *   1        = corr 语义，等价于把 kernel 时域翻转，用于复现
 *              F.conv1d(F.pad(x,(K-1,0)), kernel.unsqueeze(1), groups=H) 的行为
 */
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
    auto xTensor = context->GetInputTensor(0);
    auto kTensor = context->GetInputTensor(1);
    if (xTensor == nullptr || kTensor == nullptr)
    {
        return ge::GRAPH_FAILED;
    }
    auto xShape = xTensor->GetOriginShape();
    auto kShape = kTensor->GetOriginShape();

    // input 必须是 [B,H,L]，kernel 必须是 [H,K]
    if (xShape.GetDimNum() != 3 || kShape.GetDimNum() != 2)
    {
        return ge::GRAPH_FAILED;
    }

    const uint32_t B = static_cast<uint32_t>(xShape.GetDim(0));
    const uint32_t H = static_cast<uint32_t>(xShape.GetDim(1));
    const uint32_t L = static_cast<uint32_t>(xShape.GetDim(2));
    const uint32_t Hk = static_cast<uint32_t>(kShape.GetDim(0));
    const uint32_t K = static_cast<uint32_t>(kShape.GetDim(1));

    if (B < 1 || H < 1 || L < 1 || K < 1)
    {
        return ge::GRAPH_FAILED; // B,H,L,K >= 1
    }
    if (H != Hk)
    {
        return ge::GRAPH_FAILED; // 通道数必须一致
    }
    if (K > L)
    {
        return ge::GRAPH_FAILED; // v1 约束：K <= L
    }
    // v1 约束：线性卷积长度不得超过 N_fft 上限
    if (L + K - 1 > FFT_CONV1D_MAX_NFFT)
    {
        return ge::GRAPH_FAILED;
    }

    // // ---------------- 2. 属性 ----------------
    // uint32_t flipKernel = 0;
    // auto attrs = context->GetAttrs();
    // if (attrs != nullptr && attrs->GetAttrNum() > 0)
    // {
    //     const int64_t *p = attrs->GetAttrPointer<int64_t>(0);
    //     if (p != nullptr)
    //     {
    //         flipKernel = (*p != 0) ? 1U : 0U;
    //     }
    // }

    // ---------------- 3. FFT 长度与分解 ----------------
    // v1 取 N 为 4 的幂 => N1 = N2 = sqrt(N)，D1 与 D2 是同一个矩阵，
    // 且四步分解中 12 次 GEMM 形状统一为 (N1, N1, N1)，共用一套 TCubeTiling。
    // 取舍理由见 docs/03_ascendc_v1.md §2。
    const uint32_t need = (L + K - 1) < 2 ? 2 : (L + K - 1);
    const uint32_t nFft = NextPow4(need);
    const uint32_t nRadix = IsqrtPow4(nFft);

    // ---------------- 4. 算法分派 ----------------
    // K 较小时 direct 更快（分界点分析见 docs/02_algorithm_design.md §9）
    const uint32_t algo = (K < FFT_CONV1D_FFT_MIN_K) ? FFT_CONV1D_ALGO_DIRECT
                                                     : FFT_CONV1D_ALGO_FFT;

    // ---------------- 5. 多核切分：按 R = B*H 行切 ----------------
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
    // 调试期强制单核，排除多核相关因素（见 fft_conv1d_tiling.h 的开关说明）
    if (FFT_CONV1D_FORCE_SINGLE_CORE != 0)
    {
        usedCoreNum = 1;
    }
    const uint32_t rowsPerCore = CeilDiv(totalRows, usedCoreNum);

    // direct 路径的输出分块长度；必须 >= K-1，否则首块的零前缀逻辑不成立
    uint32_t tileLen = (L < FFT_CONV1D_DIRECT_TILE) ? L : FFT_CONV1D_DIRECT_TILE;
    if (tileLen < K)
    {
        tileLen = K; // K <= L，所以这一步不会超过 L
    }

    // ---------------- 6. Matmul tiling（单核视角，每个核算自己的 GEMM）----------------
    FftConv1dTilingData tilingData;
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
        return ge::GRAPH_FAILED;
    }

    // ---------------- 7. 填 TilingData ----------------
    tilingData.set_batch(B);
    tilingData.set_channel(H);
    tilingData.set_seqLen(L);
    tilingData.set_kernelLen(K);
    tilingData.set_totalRows(totalRows);
    tilingData.set_algo(algo);
    // tilingData.set_flipKernel(flipKernel);
    tilingData.set_nFft(nFft);
    tilingData.set_nRadix(nRadix);
    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_tileLen(tileLen);
    tilingData.set_rowsPerCore(rowsPerCore);

    context->SetBlockDim(usedCoreNum);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    // ---------------- 8. workspace ----------------
    // FFT 路径布局（单位：float）：
    //   [0]        Dr   : N            (N1*N1 == N)
    //   [N]        Di   : N
    //   [2N]       Tr   : N
    //   [3N]       Ti   : N
    //   [4N]       Kfr  : H*N
    //   [4N+HN]    Kfi  : H*N
    //   [4N+2HN]   每核 scratch : usedCoreNum * SCRATCH_BUFS * N
    // direct 路径布局：每核一份零前缀输入行 (K-1+L)，按 8 对齐
    size_t userWorkspace = 0;
    if (algo == FFT_CONV1D_ALGO_FFT)
    {
        const size_t n = static_cast<size_t>(nFft);
        userWorkspace = (4 * n + 2 * static_cast<size_t>(H) * n +
                         static_cast<size_t>(usedCoreNum) * FFT_CONV1D_SCRATCH_BUFS * n) *
                        sizeof(float);
    }
    else
    {
        // DIRECT 是纯 Vector 路径，kernel 里按 GetBlockIdx() 分片，
        // 而 MIX 模式下 AIV 的 GetBlockIdx() 范围是 [0, blockDim*2)，
        // 所以必须按 blockDim*2 份分配，否则最后一半 AIV 会越界写。
        // （当前之所以没暴露，是越界部分正好落进了系统 workspace 区。）
        const size_t rowLen = ((static_cast<size_t>(K) - 1 + L + 7) / 8) * 8;
        userWorkspace = 2 * static_cast<size_t>(usedCoreNum) * rowLen * sizeof(float);
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

        // 0 = math 语义（默认，与数学公式一致）；1 = corr 语义（kernel 时域翻转）
        // this->Attr("flip_kernel").AttrType(OPTIONAL).Int(0);

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(FftConv1d);
} // namespace ops
