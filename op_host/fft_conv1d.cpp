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
    // 三个分支的边界都由容量推导（见 fft_conv1d_tiling.h）：
    //   K < 64                        -> DIRECT   小 K 时直接卷积更快
    //   N_fft <= 1024                 -> FFT-UB   12 个 N 长缓冲常驻 UB（48KB）
    //   1024 < N_fft <= 4096          -> FFT-GM   数据放 GM，整块经 UB 中转（96KB）
    //   N_fft > 4096                  -> DIRECT   兜底，数值仍然正确
    // 条件直接写在 nFft 上（而不是 need）：nFft 才是决定缓冲大小的量。
    uint32_t algo = FFT_CONV1D_ALGO_DIRECT;
    if (K >= FFT_CONV1D_FFT_MIN_K)
    {
        if (nFft <= FFT_CONV1D_MAX_NFFT_UB)
        {
            algo = FFT_CONV1D_ALGO_FFT;
        }
        else if (nFft <= FFT_CONV1D_MAX_NFFT_GM)
        {
            algo = FFT_CONV1D_ALGO_FFT_GM;
        }
    }

    // ---------------- 4. 多核切分 ----------------
    // FFT 按通道切（每核独占若干通道，kernel 频谱每通道只算一次）；
    // DIRECT 按行切。blockDim 是 MIX 1:2 组合数；每组只由 sub-block 0
    // 执行用户数据流，sub-block 1 仅完成 KFC 客户端生命周期。
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicCoreNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    const uint32_t aivGroupNum = aivCoreNum / 2;
    uint32_t coreNum = (aicCoreNum < aivGroupNum) ? aicCoreNum : aivGroupNum;
    if (coreNum == 0)
    {
        // 平台信息缺失时只保留单组合兜底。
        coreNum = 1;
    }
    const uint32_t totalRows = B * H;
    const uint32_t splitUnit = (algo != FFT_CONV1D_ALGO_DIRECT) ? H : totalRows;
    uint32_t usedCoreNum = (splitUnit < coreNum) ? splitUnit : coreNum;
    if (usedCoreNum == 0)
    {
        usedCoreNum = 1;
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
    // Cube 版矩阵乘需要 TCubeTiling。只在 FFT 路径求解：此时 N1 ∈ {16,32,64}，
    // 不会再出现当初 DIRECT 路径下 N1=8 小于 fp32 分形导致 GetTiling 返回 -1 的情况。
    // FFT-GM 恒使用这份 tiling，因此求解失败必须直接返回错误。
    if (algo != FFT_CONV1D_ALGO_DIRECT)
    {
        // 操作数位置必须与 kernel 侧 MatmulType **逐个字段一致**，否则 Matmul 会按
        // 错误的搬运路径和缓冲划分工作，结果错误但不报错。
        //
        //   UB 路径  -> A/B 在 VECOUT、C 在 VECIN（Cube 只碰 UB）
        //   GM 路径  -> A/B/C 全在 GM（与 AscendC-S4 已验证的配置一致）
        const bool gmPath = (algo == FFT_CONV1D_ALGO_FFT_GM);
        const auto posAB = gmPath ? matmul_tiling::TPosition::GM
                                  : matmul_tiling::TPosition::VECOUT;
        const auto posC = gmPath ? matmul_tiling::TPosition::GM
                                 : matmul_tiling::TPosition::VECIN;
        matmul_tiling::MatmulApiTiling mmT(ascendcPlatform);
        mmT.SetAType(posAB, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        mmT.SetBType(posAB, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        mmT.SetCType(posC, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        mmT.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                        matmul_tiling::DataType::DT_FLOAT);
        const int32_t s1 = static_cast<int32_t>(nRadix);
        mmT.SetShape(s1, s1, s1);
        mmT.SetOrgShape(s1, s1, s1);
        mmT.SetBias(false);
        int32_t matmulUbBudget = -1;
        if (gmPath)
        {
            // FFT-GM 自有 UB：
            //   6 * N * fp32 + 2 * AlignUp8(N1) * fp32 + 8 KiB shared tmp。
            // N=4096、N1=64 时为 107008B。SetBufferSpace 的第三个参数是
            // Matmul 实际可用 UB，不能继续传 -1（表示整块 UB），否则 tiling 会
            // 与 kernel 的 TPipe 缓冲重复规划同一片 UB。
            constexpr uint64_t kGmFullBuffers = 6;
            constexpr uint64_t kGmTmpBytes = 8 * 1024;
            constexpr int32_t kCompatMatmulUbBudget = 48 * 1024;
            const uint64_t alignedRadix = (static_cast<uint64_t>(nRadix) + 7U) & ~7ULL;
            const uint64_t gmOwnedUb =
                kGmFullBuffers * static_cast<uint64_t>(nFft) * sizeof(float) +
                2U * alignedRadix * sizeof(float) + kGmTmpBytes;

            uint64_t ubSize = 0;
            ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
            if (ubSize == 0)
            {
                // 部分旧版 platform stub 不返回 UB 大小；沿用已验证过的保守预算。
                matmulUbBudget = kCompatMatmulUbBudget;
            }
            else if (ubSize <= gmOwnedUb)
            {
                printf("[fft_conv1d tiling] FFT-GM UB 不足: total=%llu owned=%llu\n",
                       static_cast<unsigned long long>(ubSize),
                       static_cast<unsigned long long>(gmOwnedUb));
                return ge::GRAPH_FAILED;
            }
            else
            {
                const uint64_t availableUb = (ubSize - gmOwnedUb) & ~31ULL;
                matmulUbBudget = static_cast<int32_t>(
                    availableUb > 0x7fffffffULL ? 0x7fffffffULL : availableUb);
            }
        }
        mmT.SetBufferSpace(-1, -1, matmulUbBudget);
        if (mmT.GetTiling(tilingData.cubeTiling) == -1)
        {
            // 不能只警告后继续：kernel 会拿着未初始化的 cubeTiling 去跑，
            // 直接挂死并表现为 ACL stream synchronize failed (507015)。
            printf("[fft_conv1d tiling] Cube tiling 求解失败: N1=%u (N=%u)\n", nRadix, nFft);
            return ge::GRAPH_FAILED;
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
    // FFT-UB 路径：全部数据常驻 UB，不需要用户 workspace
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
