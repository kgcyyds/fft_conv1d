/**
 * fft_conv1d host 侧：shape 校验、tiling、算子注册
 *
 * 算子语义（以数学公式为准，详见 docs/01_semantics_and_fft_derivation.md）：
 *   output[b,h,t] = sum_{j=0}^{K-1} input[b,h,t-j] * kernel[h,j],  t-j<0 时 input 视为 0
 *   即因果 depthwise 卷积，每个通道独立，不做通道间求和。
 *
 * 注意：这是数学意义的卷积，不是 cross-correlation。
 */
#include <cstdio>
#include <cstring>

#include "fft_conv1d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"


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

    // ---------------- 2. FFT 长度与分解 ----------------
    // N 取 4 的幂 => N1 = N2 = sqrt(N)，D1 与 D2 是同一个矩阵。
    const uint32_t need = (L + K - 1) < 2 ? 2 : (L + K - 1);
    const uint32_t nFft = NextPow4(need);
    const uint32_t nRadix = IsqrtPow4(nFft);

    // ---------------- 3. 算法分派 ----------------
    // FFT 需同时满足：K 足够大（否则 direct 更快）且 N_fft 不超过 UB 容量上限。
    // 任一不满足就走 DIRECT —— DIRECT 对任意 shape 都数值正确，只是更慢，
    // 因此算子支持的 shape 范围不因这个上限而缩小。
    // 三路分派：
    //   K < 64                     -> DIRECT（小 K 时直接卷积更快）
    //   N <= 1024                  -> FFT（UB 常驻，已验证的基线）
    //   1024 < N <= 16384          -> FFT_GM（UB 占用与 N 无关）
    //   N > 16384                  -> DIRECT（兜底，数值仍然正确）
    uint32_t algo = FFT_CONV1D_ALGO_DIRECT;
    if (K >= FFT_CONV1D_FFT_MIN_K && need <= FFT_CONV1D_MAX_NFFT)
    {
        algo = (need <= FFT_CONV1D_MAX_NFFT_UB) ? FFT_CONV1D_ALGO_FFT
                                                : FFT_CONV1D_ALGO_FFT_GM;
    }

    // ---------------- 4. 多核切分 ----------------
    // FFT 按通道切（每核独占若干通道，kernel 频谱每通道只算一次）；
    // DIRECT 按行切。两者都无跨核共享，不需要任何同步。
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0)
    {
        coreNum = 1;
    }
    const uint32_t totalRows = B * H;
    const uint32_t splitUnit = (algo != FFT_CONV1D_ALGO_DIRECT) ? H : totalRows;
    uint32_t usedCoreNum = (splitUnit < coreNum) ? splitUnit : coreNum;
    if (usedCoreNum == 0)
    {
        usedCoreNum = 1;
    }
    if (FFT_CONV1D_FORCE_SINGLE_CORE != 0)
    {
        usedCoreNum = 1; // 调试期强制单核
    }
    const uint32_t rowsPerCore = CeilDiv(totalRows, usedCoreNum);

    // direct 路径的输出分块长度
    uint32_t tileLen = (L < FFT_CONV1D_DIRECT_TILE) ? L : FFT_CONV1D_DIRECT_TILE;
    if (tileLen < K)
    {
        tileLen = K; // K <= L，不会超过 L
    }

    FftConv1dTilingData tilingData;
    // 注意：不要对 tilingData.cubeTiling 做 memset。TILING_DATA_FIELD_DEF_STRUCT
    // 生成的成员布局由框架宏决定，不保证等于 sizeof(TCubeTiling)，
    // 手工 memset 可能越界写并导致 host 侧 segfault。
    //
    // Cube 版矩阵乘需要 TCubeTiling。只在 FFT 路径求解：此时 N1 ∈ {16,32}，
    // 不会再出现当初 DIRECT 路径下 N1=8 小于 fp32 分形导致 GetTiling 返回 -1 的情况。
    // kernel 侧 FFT_CONV1D_USE_CUBE=0 时这份 tiling 不被使用，求解失败也只是回退，
    // 不让整个算子失败。
    if (algo != FFT_CONV1D_ALGO_DIRECT)
    {
        // 操作数位置必须与 kernel 侧 MatmulType **逐个字段一致**，否则 Matmul 会按
        // 错误的搬运路径和缓冲划分工作，结果错误但不报错。
        //
        // 重设计后两条 FFT 路径用的是同一套配置：A/B 在 VECOUT、C 在 VECIN
        // （GM 版的“GM”指的是数据的后备存储，Cube 本身只碰 UB）。
        matmul_tiling::MatmulApiTiling mmT(ascendcPlatform);
        mmT.SetAType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT);
        mmT.SetBType(matmul_tiling::TPosition::VECOUT, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT);
        mmT.SetCType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT);
        mmT.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT);
        const int32_t s1 = static_cast<int32_t>(nRadix);
        mmT.SetShape(s1, s1, s1);
        mmT.SetOrgShape(s1, s1, s1);
        mmT.SetBias(false);
        mmT.SetBufferSpace(-1, -1, -1);
        if (mmT.GetTiling(tilingData.cubeTiling) == -1)
        {
            printf("[fft_conv1d tiling] 警告: N1=%u 的 Cube tiling 求解失败"
                   "（USE_CUBE=0 时无影响）\n", nRadix);
        }
    }

    printf("[fft_conv1d tiling] B=%u H=%u L=%u K=%u | algo=%s N=%u N1=%u cores=%u tileLen=%u\n",
           B, H, L, K, (algo == FFT_CONV1D_ALGO_FFT) ? "FFT-UB" : ((algo == FFT_CONV1D_ALGO_FFT_GM) ? "FFT-GM" : "DIRECT"),
           nFft, nRadix, usedCoreNum, tileLen);

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
    // FFT 路径：全部数据常驻 UB，不需要用户 workspace
    // DIRECT 路径：每核一份零前缀输入行 (K-1+L)，按 8 对齐
    size_t userWorkspace = 0;
    if (algo == FFT_CONV1D_ALGO_FFT)
    {
        userWorkspace = 0; // UB 常驻，不需要用户 workspace
    }
    else if (algo == FFT_CONV1D_ALGO_FFT_GM)
    {
        // 每核 12 个长度 N 的缓冲：Dr Di Tr Ti Kfr Kfi Xr Xi Yr Yi Zr Zi
        userWorkspace = static_cast<size_t>(usedCoreNum) * FFT_CONV1D_GM_BUFS *
                        static_cast<size_t>(nFft) * sizeof(float);
    }
    else
    {
        // 每核一份，2 倍余量（AIV_ONLY 下 blockDim 即核数，2x 纯属保险）
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
