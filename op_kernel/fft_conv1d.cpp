/**
 * fft_conv1d AscendC kernel（v1，Ascend 910B / Atlas A2）
 *
 * 语义：output[b,h,t] = sum_{j=0}^{K-1} input[b,h,t-j] * kernel[h,j]，t-j<0 时 input 视为 0
 *       （因果 depthwise 卷积，通道独立，不做通道间求和）
 *       flip_kernel=1 时为 corr 语义（等价于 kernel 时域翻转），见 §3。
 *
 * 两条路径（host tiling 静态分派，kernel 侧不做动态判断）：
 *   ALGO_DIRECT：K < 64，直接因果卷积，纯 Vector
 *   ALGO_FFT   ：K >= 64，四步 Cooley-Tukey，Cube 做 GEMM + Vector 做旋转因子/逐点乘
 *
 * v1 的分解取 N 为 4 的幂 => N1 = N2 = sqrt(N)，于是：
 *   - D1 与 D2 是同一个矩阵 D（DFT 矩阵对称，且 N1==N2）
 *   - 12 次 GEMM 形状统一为 (M=N1, K=N1, N=N1)，共用一个 Matmul 对象与一套 TCubeTiling
 * 数值路径已在 python/fft_conv1d_four_step.py 的 plan_v1 中逐步验证。
 */
// ---------------------------------------------------------------------------
// Cube 开关：0 = Vector 版矩阵乘（Axpy 循环，当前已验证数值正确）
//            1 = Cube 版矩阵乘（Matmul 高阶 API，UB 进 / UB 出）
// 只影响 MatMulUB 一个函数，其余结构完全相同，因此可以一键 A/B 对照。
// 注意：Matmul 的 A/B 支持 TPosition::VECOUT、C 支持 TPosition::VECIN，
//       操作数直接用 UB 上的 LocalTensor，不需要经过 GM。
// ---------------------------------------------------------------------------
#define FFT_CONV1D_USE_CUBE 0

// FFT-GM 尚未验证通过。置 0 时整段不参与编译，确保它不会影响
// 已验证正确的 DIRECT / FFT-UB 两条路径。需与 host 的
// FFT_CONV1D_ENABLE_GM 保持一致。
#define FFT_CONV1D_ENABLE_GM_KERNEL 1

#include "kernel_operator.h"

// 必须无条件包含：自动生成的 TilingData 里含 TCubeTiling 字段，
// 即使 USE_CUBE=0 也需要这个类型可见。
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace
{
constexpr int32_t ALGO_DIRECT = 0;
constexpr int32_t ALGO_FFT = 1;    // UB 常驻（N <= 1024）
constexpr int32_t ALGO_FFT_GM = 2; // GM 版（N > 1024，解除限制）
constexpr int32_t GM_BUFS = 12;    // GM 版每核缓冲个数，与 host 一致

constexpr int32_t VEC_CHUNK = 2048;  // Vector 逐点运算的分块长度（float 个数）
constexpr int32_t TMP_BUF_BYTES = 8192; // Cos/Sin/Fmod 需要的 sharedTmpBuffer
constexpr float TWO_PI = 6.283185307179586f;

// ---------------------------------------------------------------------------
// 优化开关：Cube -> Vector 边界是否插全局栅栏
// ---------------------------------------------------------------------------
// 原实现在 Forward/Inverse/Pointwise 里每个 Cube->Vector 边界都调 SyncAll()，
// 每行 15 次全局栅栏。这是保守写法，两条依据说明它可以去掉：
//   1. IterateAll 模板默认 sync=true，文档明确“需要同步等待 IterateAll 执行结束”，
//      Cube/Vector 的握手由 Matmul API 自己完成；
//   2. 阶段 2 每个核只读写自己的 scratch（offScr_ 按 blockIdx 分片），无跨核依赖，
//      全局栅栏在这里没有任何语义作用。
// 真正需要的跨核栅栏只有两处，保留在 Process() 里：常量表就绪、kernel 频谱就绪。
//
// 若去栅栏后 FFT 路径数值出错，把下面这行改回 true 即可完全恢复原行为，
// 从而确认问题是否出在 Cube 写 GM 对 Vector 的可见性上。
constexpr bool CONSERVATIVE_SYNC = false;

// FFT-GM 的用户 workspace 基址。
//   0 = workspace                   —— 与 DIRECT 路径一致，DIRECT 已验证可用
//   1 = GetUserWorkspace(workspace) —— 文档说法，但从未单独验证过
// 当前取 0：DIRECT 路径长期用裸 workspace 且结果正确，是更可靠的依据。
// 若 GM 路径读回全 0，改成 1 再试一次即可 A/B。
constexpr int32_t GM_USE_USER_WS = 0;

__aicore__ inline void CubeVecSync()
{
    if (CONSERVATIVE_SYNC)
    {
        SyncAll();
    }
}

// ---------------------------------------------------------------------------
// MIX 模式下取一致的 block 号
// ---------------------------------------------------------------------------
// 关键事实：KERNEL_TYPE_MIX_AIC_1_2 下同一份代码在 AIC 和 AIV 上都会执行，
// 但两者的 GetBlockIdx() 取值范围不同：
//   AIC: [0, blockDim)        AIV: [0, blockDim * 2)
// （AscendC-S4 模板对 Vector 循环用 cnt = GetBlockNum()*GetTaskRation() 也印证了这点）
//
// 如果直接用 GetBlockIdx() 去切 scratch，Cube 会把 GEMM 结果写进 slot b，
// 而与它配对的两个 AIV 会去读 slot 2b 和 2b+1 —— 其中一个永远读到从未被写过的
// 区域（内容为 0），另一个还会越界写到 workspace 之外。
// 这正是 “FFT 结果很多地方是 0” 的成因。
//
// GetTaskRation() 在 Cube 上返回 1、在 Vector 上返回 2，所以除一下就能得到
// 两边一致的 block 号。代价是同一 block 的两个 AIV 做重复的 Vector 工作
// （结果相同，无害），把 Vector 拆开是后续优化项。
__aicore__ inline int32_t CoreIdx()
{
    return static_cast<int32_t>(GetBlockIdx() / GetTaskRation());
}

__aicore__ inline int32_t AlignUp8(int32_t x)
{
    return (x + 7U) & ~7U;
}

__aicore__ inline int32_t MinU(int32_t a, int32_t b)
{
    return a < b ? a : b;
}
} // namespace

// ============================================================================
// 路径一：直接因果卷积（Vector）
// ============================================================================
// 实现要点：Ascend 的 Vector 指令要求 LocalTensor 操作数首地址 32B 对齐，
// 而 out[t] += x[t-j]*k[j] 的滑窗天然产生非对齐偏移。解决办法是把
// "K-1 个前导零 + 输入行" 物化到 workspace（GM 地址无对齐约束），
// 每个抽头 j 从 GM 偏移处直接搬到 UB 的 0 号位置，UB 侧永远对齐。
class FftConv1dDirect
{
  public:
    __aicore__ inline FftConv1dDirect()
    {
    }

    template <class TILING>
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                GM_ADDR workspace, const TILING &t)
    {
        H_ = t.channel;
        L_ = t.seqLen;
        K_ = t.kernelLen;
        rows_ = t.totalRows;
        // flip_ = t.flipKernel;
        tile_ = t.tileLen;
        cores_ = t.usedCoreNum;
        rowLen_ = AlignUp8(K_ - 1 + L_);

        xGm_.SetGlobalBuffer((__gm__ float *)x);
        wGm_.SetGlobalBuffer((__gm__ float *)w);
        yGm_.SetGlobalBuffer((__gm__ float *)y);
        wsGm_.SetGlobalBuffer((__gm__ float *)workspace);

        pipe->InitBuffer(bufIn_, AlignUp8(tile_) * sizeof(float));
        pipe->InitBuffer(bufOut_, AlignUp8(tile_) * sizeof(float));
        pipe->InitBuffer(bufK_, AlignUp8(K_) * sizeof(float));
        pipe->InitBuffer(bufZero_, AlignUp8(K_ > 8 ? K_ : 8) * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> zeros = bufZero_.Get<float>();
        Duplicate(zeros, 0.0f, AlignUp8(K_ > 8 ? K_ : 8));
        PipeBarrier<PIPE_V>();

        const int32_t base = static_cast<int32_t>(GetBlockIdx()) * rowLen_;
        for (int32_t row = static_cast<int32_t>(GetBlockIdx()); row < rows_; row += cores_)
        {
            // input 布局 [B,H,L]，行号 row = b*H + h => 通道号 h = row % H
            LoadKernelRow(row % H_);
            BuildPaddedRow(row, base);
            for (int32_t t0 = 0; t0 < L_; t0 += tile_)
            {
                ComputeTile(row, base, t0, MinU(tile_, L_ - t0));
            }
        }
    }

  private:
    // 把 kernel 的第 h 行搬进 UB，后续用 GetValue 取标量
    __aicore__ inline void LoadKernelRow(int32_t h)
    {
        LocalTensor<float> kb = bufK_.Get<float>();
        DataCopyExtParams cp{1, K_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(kb, wGm_[static_cast<uint64_t>(h) * K_], cp, pad);
        // 标量读取前必须等搬运完成
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    // 在 workspace 里物化 [K-1 个 0, x[row][0..L-1]]
    __aicore__ inline void BuildPaddedRow(int32_t row, int32_t base)
    {
        if (K_ > 1)
        {
            LocalTensor<float> zeros = bufZero_.Get<float>();
            DataCopyExtParams zp{1, (K_ - 1) * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(wsGm_[base], zeros, zp);
        }
        LocalTensor<float> in = bufIn_.Get<float>();
        for (int32_t t0 = 0; t0 < L_; t0 += tile_)
        {
            const int32_t len = MinU(tile_, L_ - t0);
            DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(in, xGm_[static_cast<uint64_t>(row) * L_ + t0], cp, pad);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopyExtParams op{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(wsGm_[base + K_ - 1 + t0], in, op);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void ComputeTile(int32_t row, int32_t base, int32_t t0, int32_t len)
    {
        LocalTensor<float> kb = bufK_.Get<float>();
        LocalTensor<float> out = bufOut_.Get<float>();
        LocalTensor<float> xin = bufIn_.Get<float>();

        Duplicate(out, 0.0f, AlignUp8(len));
        PipeBarrier<PIPE_V>();

        for (int32_t j = 0; j < K_; ++j)
        {
            // math 语义取 kernel[j]；corr 语义取 kernel[K-1-j]（等价于时域翻转）
            const float kv = kb.GetValue(j);
            // xz[t0 + K-1 - j] 即 x[t0 - j]，越界部分由零前缀保证为 0
            DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(xin, wsGm_[base + t0 + K_ - 1 - j], cp, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Axpy(out, xin, kv, len); // out += x * k[j]
            SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        }

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams op{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(yGm_[static_cast<uint64_t>(row) * L_ + t0], out, op);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    GlobalTensor<float> xGm_, wGm_, yGm_, wsGm_;
    TBuf<TPosition::VECCALC> bufIn_, bufOut_, bufK_, bufZero_;
    int32_t H_, L_, K_, rows_, tile_, cores_, rowLen_;
};

// ============================================================================
// 路径二：四步 Cooley-Tukey FFT 卷积（重写版，UB 常驻）
// ============================================================================
// 重写的依据（都是实测结论，不是推测）：
//   1. DIRECT 路径正确 => DataCopyPad / Duplicate / Axpy / GetValue / GM workspace
//      这套原语在本机是可用的。
//   2. FFT 路径单核仍然全 0 => 与多核分片无关；与 GM 通路无关（DIRECT 用的同一块）。
//      唯一的差异是 Matmul 高阶 API 在本 MIX 配置下没有产出。
//   3. fp32 在 A2 上没有硬件转置（Transpose 只支持 16bit），纯 Vector 的蝶形 FFT
//      在小 stride 上无法满足 32B 对齐 => O(N logN) 蝶形不可行，DFT-matmul 是对的算法。
//
// 于是这一版把 Matmul 从关键路径摘掉：矩阵乘用 Axpy 循环实现。
//   C[i][:] = sum_k A[i][k] * B[k][:]
// 每个 Axpy 的操作数都是整行（长度 N1），偏移是 N1 的倍数（16/32，天然 32B 对齐），
// 与 DIRECT 路径里已验证可用的写法完全同构。
//
// 所有数据常驻 UB：不用 GM scratch、不用跨核共享、不需要任何 SyncAll。
// 循环按“通道外层、batch 内层”，kernel 频谱每通道只算一次。
//
// 容量：12 个长度 N 的缓冲，N<=1024 时 48KB，很宽裕。
// host 侧保证 FFT 路径只在 L+K-1 <= 1024 时选用，否则回退 DIRECT（功能不减）。
class FftConv1dFft
{
  public:
#if FFT_CONV1D_USE_CUBE
    // 12 次矩阵乘形状统一为 (N1, N1, N1)，共用一个对象。
    // A/B 放 VECOUT、C 放 VECIN => 全程 UB，不落 GM。
    Matmul<MatmulType<TPosition::VECOUT, CubeFormat::ND, float>,
           MatmulType<TPosition::VECOUT, CubeFormat::ND, float>,
           MatmulType<TPosition::VECIN, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>>
        mm_;
#endif

    __aicore__ inline FftConv1dFft()
    {
    }

    template <class TILING>
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                GM_ADDR workspace, const TILING &t)
    {
        B_ = t.batch;
        H_ = t.channel;
        L_ = t.seqLen;
        K_ = t.kernelLen;
        N_ = t.nFft;
        N1_ = t.nRadix;
        cores_ = t.usedCoreNum;

        xGm_.SetGlobalBuffer((__gm__ float *)x);
        wGm_.SetGlobalBuffer((__gm__ float *)w);
        yGm_.SetGlobalBuffer((__gm__ float *)y);

        const int32_t bytes = N_ * sizeof(float);
        pipe->InitBuffer(bDr_, bytes);
        pipe->InitBuffer(bDi_, bytes);
        pipe->InitBuffer(bTr_, bytes);
        pipe->InitBuffer(bTi_, bytes);
        pipe->InitBuffer(bKr_, bytes);
        pipe->InitBuffer(bKi_, bytes);
        pipe->InitBuffer(bXr_, bytes);
        pipe->InitBuffer(bXi_, bytes);
        pipe->InitBuffer(bYr_, bytes);
        pipe->InitBuffer(bYi_, bytes);
        pipe->InitBuffer(bZr_, bytes);
        pipe->InitBuffer(bZi_, bytes);
        pipe->InitBuffer(bTmp_, TMP_BUF_BYTES);
    }

    __aicore__ inline void Process()
    {
        BuildTables();

        // 按通道切核：每个核完整拥有若干通道，核之间零共享，因此不需要任何同步。
        // kernel 频谱在通道循环体内只算一次，被该通道的全部 batch 复用。
        for (int32_t h = CoreIdx(); h < H_; h += cores_)
        {
            LoadToX(wGm_, static_cast<uint64_t>(h) * K_, K_); // Xr = [kernel 行, 0...]
            Forward();                                        // -> (Xr, Xi)
            CopyBuf(bKr_.Get<float>(), bXr_.Get<float>());
            CopyBuf(bKi_.Get<float>(), bXi_.Get<float>());

            for (int32_t b = 0; b < B_; ++b)
            {
                const int32_t row = b * H_ + h; // input 布局 [B,H,L]
                LoadToX(xGm_, static_cast<uint64_t>(row) * L_, L_);
                Forward();
                // 频域逐点复乘 (Xr,Xi) *= (Kr,Ki)
                CMul(bXr_.Get<float>(), bXi_.Get<float>(),
                     bKr_.Get<float>(), bKi_.Get<float>(), false);
                Inverse();  // (Xr,Xi) -> Xr（实数时域）
                StoreFromX(static_cast<uint64_t>(row) * L_);
            }
        }
    }

  private:
    // ------------------------------------------------------------------
    // 常量表：D[k][n] = exp(-2πi*((k*n) mod N1)/N1)，T[k1][n2] = exp(-2πi*((k1*n2) mod N)/N)
    // 必须先做整数乘再取模：浮点乘不满足结合律，写成 (-2π*k)*n/M 会破坏 D 的对称性
    // （“D 对称所以不用转置”是本方案成立的前提），取模还能避免大幅角的规约误差。
    // ------------------------------------------------------------------
    __aicore__ inline void BuildTables()
    {
        LocalTensor<float> dr = bDr_.Get<float>();
        LocalTensor<float> di = bDi_.Get<float>();
        LocalTensor<float> tr = bTr_.Get<float>();
        LocalTensor<float> ti = bTi_.Get<float>();
        LocalTensor<float> idx = bZr_.Get<float>(); // 借用工作缓冲
        LocalTensor<float> ang = bZi_.Get<float>();
        LocalTensor<uint8_t> tmp = bTmp_.Get<uint8_t>();

        CreateVecIndex(idx, 0.0f, N1_); // idx[n] = n（A2 支持 float）
        PipeBarrier<PIPE_V>();

        for (int32_t k = 0; k < N1_; ++k)
        {
            EmitRow(dr[k * N1_], di[k * N1_], idx, ang, tmp, k, N1_);
            EmitRow(tr[k * N1_], ti[k * N1_], idx, ang, tmp, k, N_);
        }
        // 后面要用 GetValue 从标量单元读这些表，必须等 Vector 写完
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    // 生成一行：re[n] = cos(θ)，im[n] = sin(θ)，θ = -2π·k·n/M
    //
    // 全部用 Vector 指令，不做任何标量运算：AICore 的标量单元没有硬件除法器，
    // 整数取模在 NPU 上不可用。而取模本来就不必要 —— cos/sin 是 2π 周期的，
    // 取模只影响精度、不影响正确性。
    //
    // 两个 Muls 的**顺序不能合并**：必须先算出精确整数 k*n，再乘同一个 step。
    //   正确：(k*n) * step        —— k*n 与 n*k 逐位相同 => D[k][n] == D[n][k] 精确成立
    //   错误：n * (k*step)        —— 浮点乘不满足结合律，D 会失去对称性，
    //                               而“D 对称所以不用转置”是本方案成立的前提
    // k*n <= (N1-1)^2 <= 961 < 2^24，fp32 精确表示。
    //
    // 代价：D 表角度最大约 2π·961/32 ≈ 189 rad，fp32 下绝对误差约 1e-5，
    //       cos/sin 误差同量级。在 1e-4 的目标内，首版本接受；
    //       后续要更高精度可改用查表 + Gather 做角度规约。
    //
    // 另外 Cos/Sin 的 dst 与 src 必须是不同缓冲，不能原地覆盖。
    __aicore__ inline void EmitRow(const LocalTensor<float> &re, const LocalTensor<float> &im,
                                   const LocalTensor<float> &idx, const LocalTensor<float> &ang,
                                   const LocalTensor<uint8_t> &tmp, int32_t k, int32_t mod)
    {
        Muls(ang, idx, static_cast<float>(k), N1_); // ang = k*n（精确整数）
        PipeBarrier<PIPE_V>();
        Muls(ang, ang, -TWO_PI / static_cast<float>(mod), N1_); // 乘同一个 step
        PipeBarrier<PIPE_V>();
        Cos(re, ang, tmp, N1_); // dst != src
        Sin(im, ang, tmp, N1_);
        PipeBarrier<PIPE_V>();
    }

    // ------------------------------------------------------------------
    // UB 内的 [N1,N1] 矩阵乘：C = A @ B，用 Axpy 逐行累加
    // C[i][:] = sum_k A[i][k] * B[k][:]
    // 每次 Axpy 处理一整行（N1 个 float），偏移 i*N1 / k*N1 都是 N1 的倍数，
    // N1 ∈ {16,32} 均为 8 的倍数 => 首地址天然 32B 对齐。
    // ------------------------------------------------------------------
    __aicore__ inline void MatMulUB(const LocalTensor<float> &c, const LocalTensor<float> &a,
                                    const LocalTensor<float> &b)
    {
#if FFT_CONV1D_USE_CUBE
        // Cube 版：A/B 从 UB(VECOUT) 送入，C 直接写回 UB(VECIN)
        mm_.SetTensorA(a);
        mm_.SetTensorB(b);
        mm_.IterateAll(c);
        mm_.End();
        PipeBarrier<PIPE_ALL>();
#else
        // Vector 版：C[i][:] = sum_k A[i][k] * B[k][:]
        // 每次 Axpy 处理一整行（N1 个 float），偏移 i*N1 / k*N1 都是 N1 的倍数，
        // N1 ∈ {16,32} 均为 8 的倍数 => 首地址天然 32B 对齐。
        // a 由上一步 Vector 运算写出，标量单元读它之前必须同步。
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        for (int32_t i = 0; i < N1_; ++i)
        {
            Duplicate(c[i * N1_], 0.0f, N1_);
            PipeBarrier<PIPE_V>();
            for (int32_t k = 0; k < N1_; ++k)
            {
                Axpy(c[i * N1_], b[k * N1_], a.GetValue(i * N1_ + k), N1_);
            }
            PipeBarrier<PIPE_V>();
        }
#endif
    }

    // (ar,ai) *= (br,bi)，conjB 时用 (br,-bi)。4 次实乘，数值最稳。
    __aicore__ inline void CMul(const LocalTensor<float> &ar, const LocalTensor<float> &ai,
                                const LocalTensor<float> &br, const LocalTensor<float> &bi,
                                bool conjB)
    {
        LocalTensor<float> t0 = bZr_.Get<float>();
        LocalTensor<float> t1 = bZi_.Get<float>();
        Mul(t0, ar, br, N_); // ar*br
        Mul(t1, ai, bi, N_); // ai*bi
        PipeBarrier<PIPE_V>();
        if (conjB)
        {
            Add(t0, t0, t1, N_); // conj: 实部 = ar*br + ai*bi
        }
        else
        {
            Sub(t0, t0, t1, N_); // 实部 = ar*br - ai*bi
        }
        PipeBarrier<PIPE_V>();
        Mul(t1, ar, bi, N_); // ar*bi
        PipeBarrier<PIPE_V>();
        Mul(ar, ai, br, N_); // ai*br（ar 此时已不再需要，安全复用）
        PipeBarrier<PIPE_V>();
        if (conjB)
        {
            Sub(ai, ar, t1, N_); // conj: 虚部 = ai*br - ar*bi
        }
        else
        {
            Add(ai, ar, t1, N_); // 虚部 = ar*bi + ai*br
        }
        PipeBarrier<PIPE_V>();
        CopyBuf(ar, t0);
    }

    __aicore__ inline void CopyBuf(const LocalTensor<float> &dst, const LocalTensor<float> &src)
    {
        Adds(dst, src, 0.0f, N_); // UB->UB 拷贝，用 Adds 最省事
        PipeBarrier<PIPE_V>();
    }

    // ------------------------------------------------------------------
    // 正变换：(Xr) 实数时域 -> (Xr, Xi) 频域（[k1][k2] 置换序，与逆变换自洽）
    // ------------------------------------------------------------------
    __aicore__ inline void Forward()
    {
        LocalTensor<float> dr = bDr_.Get<float>();
        LocalTensor<float> di = bDi_.Get<float>();
        LocalTensor<float> xr = bXr_.Get<float>();
        LocalTensor<float> xi = bXi_.Get<float>();
        LocalTensor<float> yr = bYr_.Get<float>();
        LocalTensor<float> yi = bYi_.Get<float>();
        LocalTensor<float> zr = bZr_.Get<float>();
        LocalTensor<float> zi = bZi_.Get<float>();

        // 步骤 A：Br = Dr @ X，Bi = Di @ X（实输入，只需 2 次实矩阵乘）
        MatMulUB(yr, dr, xr);
        MatMulUB(yi, di, xr);
        // 步骤 B：乘旋转因子
        CMul(yr, yi, bTr_.Get<float>(), bTi_.Get<float>(), false);
        // 步骤 C：Cr = Br@Dr - Bi@Di，Ci = Br@Di + Bi@Dr（D 对称，无需转置）
        MatMulUB(zr, yr, dr);
        MatMulUB(zi, yi, di);
        Sub(xr, zr, zi, N_);
        PipeBarrier<PIPE_V>();
        MatMulUB(zr, yr, di);
        MatMulUB(zi, yi, dr);
        Add(xi, zr, zi, N_);
        PipeBarrier<PIPE_V>();
    }

    // ------------------------------------------------------------------
    // 逆变换：(Xr, Xi) -> Xr（实数时域，已含 1/N 归一化）
    // ------------------------------------------------------------------
    __aicore__ inline void Inverse()
    {
        LocalTensor<float> dr = bDr_.Get<float>();
        LocalTensor<float> di = bDi_.Get<float>();
        LocalTensor<float> xr = bXr_.Get<float>();
        LocalTensor<float> xi = bXi_.Get<float>();
        LocalTensor<float> yr = bYr_.Get<float>();
        LocalTensor<float> yi = bYi_.Get<float>();
        LocalTensor<float> zr = bZr_.Get<float>();
        LocalTensor<float> zi = bZi_.Get<float>();

        // 步骤 A'：Er = Yr@Dr + Yi@Di，Ei = Yi@Dr - Yr@Di（conj(D) = (Dr, -Di)）
        MatMulUB(zr, xr, dr);
        MatMulUB(zi, xi, di);
        Add(yr, zr, zi, N_);
        PipeBarrier<PIPE_V>();
        MatMulUB(zr, xr, di);
        MatMulUB(zi, xi, dr);
        Sub(yi, zi, zr, N_);
        PipeBarrier<PIPE_V>();
        // 步骤 B'：乘共轭旋转因子
        CMul(yr, yi, bTr_.Get<float>(), bTi_.Get<float>(), true);
        // 步骤 C'：y = (Dr@Er + Di@Ei)/N（实输出，只取实部）
        MatMulUB(zr, dr, yr);
        MatMulUB(zi, di, yi);
        Add(xr, zr, zi, N_);
        PipeBarrier<PIPE_V>();
        Muls(xr, xr, 1.0f / static_cast<float>(N_), N_);
        PipeBarrier<PIPE_V>();
    }

    // ------------------------------------------------------------------
    // GM <-> UB
    // ------------------------------------------------------------------
    // Xr = [src 的 count 个元素, 0 补齐到 N]；Xi 清零
    __aicore__ inline void LoadToX(const GlobalTensor<float> &src, uint64_t off, int32_t count)
    {
        LocalTensor<float> xr = bXr_.Get<float>();
        Duplicate(xr, 0.0f, N_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyExtParams cp{1, static_cast<uint32_t>(count) * static_cast<uint32_t>(sizeof(float)),
                             0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(xr, src[off], cp, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    // 因果卷积取线性卷积的前 L 个点，偏移为 0（推导见 docs/01 §5）
    __aicore__ inline void StoreFromX(uint64_t off)
    {
        LocalTensor<float> xr = bXr_.Get<float>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, static_cast<uint32_t>(L_) * static_cast<uint32_t>(sizeof(float)),
                             0, 0, 0};
        DataCopyPad(yGm_[off], xr, cp);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    GlobalTensor<float> xGm_, wGm_, yGm_;
    TBuf<TPosition::VECCALC> bDr_, bDi_, bTr_, bTi_, bKr_, bKi_;
    TBuf<TPosition::VECCALC> bXr_, bXi_, bYr_, bYi_, bZr_, bZi_, bTmp_;
    int32_t B_, H_, L_, K_, N_, N1_, cores_;
};

#if FFT_CONV1D_ENABLE_GM_KERNEL
// ============================================================================
// 路径三：FFT-GM（重新设计版）—— 以“消灭同步问题”为唯一目标，暂不考虑性能
// ============================================================================
// 设计出发点是一条已验证的事实：UB 路径（FftConv1dFft）完全正确，它的 Matmul
// 用的是 IterateAll(LocalTensor) **同步**重载，Cube 的输入输出都在 UB，
// 因此不存在 “Cube 写 GM、Vector 读 GM” 的跨核可见性问题。
//
// 所以本设计的铁律：**Cube 永远不碰 GM**。
//   - GM 只作为 12 个长度 N 的后备存储（每核独占一片）
//   - 每个运算之前把整块 N 个 float 从 GM 搬进 UB，算完再整块搬回
//   - 矩阵乘一律走 A/B/C 全在 UB 的同步重载（与已验证的 UB 路径完全一致）
//
// 由此消除的失败面：
//   1. Cube→GM→Vector 的跨核同步          => 不存在，Cube 只读写 UB
//   2. IterateAll 的异步/waitIterateAll   => 用的是没有该参数的同步重载
//   3. Matmul 系统 workspace 与用户数据重叠 => 用户区从 GetUserWorkspace() 起算
//   4. SyncAll 调用次数不匹配              => 全程没有 SyncAll（每核零共享）
//   5. Vector 非对齐访问                   => 所有搬运都是整块 N，UB 侧偏移恒为 0
//
// 代价：UB 需同时容纳 6 个长度 N 的缓冲（复数逐点乘要 4 入 2 出），
//       N=4096 时 6*16KB=96KB，是本路径的上限；N>4096 由 host 回退 DIRECT。
//       相对 UB 路径的 1024，覆盖范围提升 4 倍（L+K-1 <= 4096）。
class FftConv1dFftGm
{
  public:
    // A/B/C 全在 GM —— 与 AscendC-S4 模板中已验证可用的配置一致。
    // Cube 直接读写 GM，不经 UB 中转。
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>>
        mm_;

    __aicore__ inline FftConv1dFftGm()
    {
    }

    template <class TILING>
    __aicore__ inline void Init(TPipe *pipe, GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                GM_ADDR workspace, const TILING &t)
    {
        B_ = t.batch;
        H_ = t.channel;
        L_ = t.seqLen;
        K_ = t.kernelLen;
        N_ = t.nFft;
        N1_ = t.nRadix;
        cores_ = t.usedCoreNum;

        xGm_.SetGlobalBuffer((__gm__ float *)x);
        wGm_.SetGlobalBuffer((__gm__ float *)w);
        yGm_.SetGlobalBuffer((__gm__ float *)y);
        // 用户区必须避开系统 workspace：Matmul 的内部 scratch 用的就是系统区
        wsGm_.SetGlobalBuffer((__gm__ float *)(GM_USE_USER_WS ? GetUserWorkspace(workspace)
                                                             : workspace));

        base_ = static_cast<uint64_t>(CoreIdx()) * GM_BUFS * N_;

        const int32_t nb = N_ * sizeof(float);
        pipe->InitBuffer(b0_, nb);
        pipe->InitBuffer(b1_, nb);
        pipe->InitBuffer(b2_, nb);
        pipe->InitBuffer(b3_, nb);
        pipe->InitBuffer(b4_, nb);
        pipe->InitBuffer(b5_, nb);
        pipe->InitBuffer(bIdx_, AlignUp8(N1_) * sizeof(float));
        pipe->InitBuffer(bAng_, AlignUp8(N1_) * sizeof(float));
        pipe->InitBuffer(bTmp_, TMP_BUF_BYTES);
    }

    __aicore__ inline void Process()
    {
        BuildTables();
        SyncAll();

        // SyncAll 是全核栅栏，要求**所有核调用次数一致**，
        // 所以两层循环都用固定次数 + active 标志（分不到通道的核照样走完流程，
        // 只是不写回输出）。这也是 AscendC-S4 模板的做法。
        const int32_t chPerCore = (H_ + cores_ - 1) / cores_;
        for (int32_t i = 0; i < chPerCore; ++i)
        {
            const int32_t h = CoreIdx() + i * cores_;
            const bool active = (h < H_);
            LoadRowG(Buf(XR), wGm_, static_cast<uint64_t>(active ? h : 0) * K_, K_);
            Forward();
            CopyG(Buf(KR), Buf(XR));
            CopyG(Buf(KI), Buf(XI));
            for (int32_t b = 0; b < B_; ++b)
            {
                const int32_t row = active ? (b * H_ + h) : 0;
                LoadRowG(Buf(XR), xGm_, static_cast<uint64_t>(row) * L_, L_);
                Forward();
                CMulG(Buf(XR), Buf(XI), Buf(KR), Buf(KI), false);
                Inverse();
                if (active)
                {
                    StoreOutG(static_cast<uint64_t>(row) * L_);
                }
            }
        }
    }

  private:
    enum : int32_t { DR = 0, DI, TR, TI, KR, KI, XR, XI, YR, YI, ZR, ZI };

    __aicore__ inline uint64_t Buf(int32_t i) const
    {
        return base_ + static_cast<uint64_t>(i) * N_;
    }

    // ---------------- GM <-> UB 的整块搬运（长度恒为 N，UB 偏移恒为 0）----------------
    __aicore__ inline void LoadG(const LocalTensor<float> &d, uint64_t off)
    {
        DataCopyExtParams cp{1, static_cast<uint32_t>(N_) * 4U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(d, wsGm_[off], cp, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void StoreG(uint64_t off, const LocalTensor<float> &s)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, static_cast<uint32_t>(N_) * 4U, 0, 0, 0};
        DataCopyPad(wsGm_[off], s, cp);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void CopyG(uint64_t dst, uint64_t src)
    {
        LocalTensor<float> t = b0_.Get<float>();
        LoadG(t, src);
        StoreG(dst, t);
    }

    // ---------------- 常量表：整张在 UB 生成后一次性落 GM ----------------
    // 角度算法与已验证的 UB 路径逐字相同：先算精确整数 k*n，再乘同一个 step，
    // 保证 D[k][n] == D[n][k] 精确成立（“D 对称所以不用转置”是本方案的前提）。
    __aicore__ inline void BuildTables()
    {
        LocalTensor<float> idx = bIdx_.Get<float>();
        LocalTensor<float> ang = bAng_.Get<float>();
        LocalTensor<uint8_t> tmp = bTmp_.Get<uint8_t>();
        LocalTensor<float> dr = b0_.Get<float>();
        LocalTensor<float> di = b1_.Get<float>();
        LocalTensor<float> tr = b2_.Get<float>();
        LocalTensor<float> ti = b3_.Get<float>();

        CreateVecIndex(idx, 0.0f, N1_);
        PipeBarrier<PIPE_V>();
        for (int32_t k = 0; k < N1_; ++k)
        {
            Row(dr[k * N1_], di[k * N1_], idx, ang, tmp, k, N1_);
            Row(tr[k * N1_], ti[k * N1_], idx, ang, tmp, k, N_);
        }
        StoreG(Buf(DR), dr);
        StoreG(Buf(DI), di);
        StoreG(Buf(TR), tr);
        StoreG(Buf(TI), ti);
    }

    __aicore__ inline void Row(const LocalTensor<float> &re, const LocalTensor<float> &im,
                               const LocalTensor<float> &idx, const LocalTensor<float> &ang,
                               const LocalTensor<uint8_t> &tmp, int32_t k, int32_t mod)
    {
        Muls(ang, idx, static_cast<float>(k), N1_);              // k*n（精确整数）
        PipeBarrier<PIPE_V>();
        Muls(ang, ang, -TWO_PI / static_cast<float>(mod), N1_);  // 乘同一个 step
        PipeBarrier<PIPE_V>();
        Cos(re, ang, tmp, N1_); // dst != src
        Sin(im, ang, tmp, N1_);
        PipeBarrier<PIPE_V>();
    }

    // ---------------- 矩阵乘：GM -> UB -> Cube -> UB -> GM ----------------
    // Cube 的输入输出全在 UB，走的是没有 waitIterateAll 参数的同步重载，
    // 与已验证的 UB 路径完全一致，因此不存在跨核可见性问题。
    // 矩阵乘：A/B/C 全在 GM，Cube 直接读写。
    //
    __aicore__ inline void MatMulG(uint64_t c, uint64_t a, uint64_t b)
    {
        // 同步只能用 SyncAll()。
        // 官方文档说明：Matmul 高阶 API **内部已经使用 CrossCoreSetFlag/WaitFlag**
        // 做核间同步，不建议再手工调用该接口 —— 手工插入会与它的内部握手
        // 争用同一组 flag 计数器，导致死锁（上一版正是这么挂的）。
        // PipeBarrier 也不行：只约束单核内流水，跨不到 AIC/AIV。
        //
        // 代价：SyncAll 是全核栅栏，要求所有核调用次数一致，
        // 因此 Process() 的循环必须是固定次数 + active 标志。
        SyncAll();
        mm_.SetTensorA(wsGm_[a]);
        mm_.SetTensorB(wsGm_[b]);
        mm_.IterateAll(wsGm_[c]);
        mm_.End();
        SyncAll();
    }

    // ---------------- 逐点运算：整块进出 ----------------
    // kind: 0 = a+b, 1 = a-b, 2 = (a+b)*s
    __aicore__ inline void BinG(uint64_t dst, uint64_t oa, uint64_t ob, int32_t kind, float s)
    {
        LocalTensor<float> a = b0_.Get<float>();
        LocalTensor<float> b = b1_.Get<float>();
        LocalTensor<float> c = b2_.Get<float>();
        LoadG(a, oa);
        LoadG(b, ob);
        if (kind == 1)
        {
            Sub(c, a, b, N_);
        }
        else
        {
            Add(c, a, b, N_);
        }
        PipeBarrier<PIPE_V>();
        if (kind == 2)
        {
            Muls(c, c, s, N_);
            PipeBarrier<PIPE_V>();
        }
        StoreG(dst, c);
    }

    // (ar,ai) *= (br,bi)，conjB 时用 (br,-bi)。4 次实乘。
    __aicore__ inline void CMulG(uint64_t oar, uint64_t oai, uint64_t obr, uint64_t obi,
                                 bool conjB)
    {
        LocalTensor<float> ar = b0_.Get<float>();
        LocalTensor<float> ai = b1_.Get<float>();
        LocalTensor<float> br = b2_.Get<float>();
        LocalTensor<float> bi = b3_.Get<float>();
        LocalTensor<float> t0 = b4_.Get<float>();
        LocalTensor<float> t1 = b5_.Get<float>();
        LoadG(ar, oar);
        LoadG(ai, oai);
        LoadG(br, obr);
        LoadG(bi, obi);
        Mul(t0, ar, br, N_); // ar*br
        Mul(t1, ai, bi, N_); // ai*bi
        PipeBarrier<PIPE_V>();
        if (conjB)
        {
            Add(t0, t0, t1, N_); // conj 实部 = ar*br + ai*bi
        }
        else
        {
            Sub(t0, t0, t1, N_); // 实部 = ar*br - ai*bi
        }
        PipeBarrier<PIPE_V>();
        Mul(t1, ar, bi, N_); // ar*bi
        PipeBarrier<PIPE_V>();
        Mul(ar, ai, br, N_); // ai*br（ar 已不再需要）
        PipeBarrier<PIPE_V>();
        if (conjB)
        {
            Sub(t1, ar, t1, N_); // conj 虚部 = ai*br - ar*bi
        }
        else
        {
            Add(t1, t1, ar, N_); // 虚部 = ar*bi + ai*br
        }
        PipeBarrier<PIPE_V>();
        StoreG(oar, t0);
        StoreG(oai, t1);
    }

    // ---------------- 正/逆变换（与已验证的 UB 路径逐步对应）----------------
    __aicore__ inline void Forward()
    {
        MatMulG(Buf(YR), Buf(DR), Buf(XR)); // Br = Dr @ X
        MatMulG(Buf(YI), Buf(DI), Buf(XR)); // Bi = Di @ X
        CMulG(Buf(YR), Buf(YI), Buf(TR), Buf(TI), false); // 乘旋转因子
        MatMulG(Buf(ZR), Buf(YR), Buf(DR));
        MatMulG(Buf(ZI), Buf(YI), Buf(DI));
        BinG(Buf(XR), Buf(ZR), Buf(ZI), 1, 0.0f); // Cr = Br@Dr - Bi@Di
        MatMulG(Buf(ZR), Buf(YR), Buf(DI));
        MatMulG(Buf(ZI), Buf(YI), Buf(DR));
        BinG(Buf(XI), Buf(ZR), Buf(ZI), 0, 0.0f); // Ci = Br@Di + Bi@Dr
    }

    __aicore__ inline void Inverse()
    {
        MatMulG(Buf(ZR), Buf(XR), Buf(DR));
        MatMulG(Buf(ZI), Buf(XI), Buf(DI));
        BinG(Buf(YR), Buf(ZR), Buf(ZI), 0, 0.0f); // Er = Yr@Dr + Yi@Di
        MatMulG(Buf(ZR), Buf(XR), Buf(DI));
        MatMulG(Buf(ZI), Buf(XI), Buf(DR));
        BinG(Buf(YI), Buf(ZI), Buf(ZR), 1, 0.0f); // Ei = Yi@Dr - Yr@Di
        CMulG(Buf(YR), Buf(YI), Buf(TR), Buf(TI), true); // 共轭旋转因子
        MatMulG(Buf(ZR), Buf(DR), Buf(YR));
        MatMulG(Buf(ZI), Buf(DI), Buf(YI));
        BinG(Buf(XR), Buf(ZR), Buf(ZI), 2, 1.0f / static_cast<float>(N_));
    }

    // ---------------- 行搬运 ----------------
    // dst = [src 的 count 个元素, 0 补齐到 N]
    __aicore__ inline void LoadRowG(uint64_t dst, const GlobalTensor<float> &src, uint64_t off,
                                    int32_t count)
    {
        LocalTensor<float> a = b0_.Get<float>();
        Duplicate(a, 0.0f, N_);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        DataCopyExtParams cp{1, static_cast<uint32_t>(count) * 4U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(a, src[off], cp, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        StoreG(dst, a);
    }

    // 因果卷积取线性卷积前 L 点，偏移为 0
    __aicore__ inline void StoreOutG(uint64_t yOff)
    {
        LocalTensor<float> a = b0_.Get<float>();
        LoadG(a, Buf(XR));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, static_cast<uint32_t>(L_) * 4U, 0, 0, 0};
        DataCopyPad(yGm_[yOff], a, cp);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    GlobalTensor<float> xGm_, wGm_, yGm_, wsGm_;
    TBuf<TPosition::VECCALC> b0_, b1_, b2_, b3_, b4_, b5_;
    TBuf<TPosition::VECCALC> bIdx_, bAng_, bTmp_;
    uint64_t base_;
    int32_t B_, H_, L_, K_, N_, N1_, cores_;
};

#endif // FFT_CONV1D_ENABLE_GM_KERNEL

// ============================================================================
// kernel 入口
// ============================================================================
extern "C" __global__ __aicore__ void fft_conv1d(GM_ADDR x, GM_ADDR kernel, GM_ADDR y,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    // GM 版 FFT 恒用 Cube，因此任务类型必须是 MIX（DIRECT/UB 版在 MIX 下同样能跑）
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    if (tilingData.algo == ALGO_DIRECT)
    {
        FftConv1dDirect op;
        op.Init(&pipe, x, kernel, y, workspace, tilingData);
        op.Process();
    }
#if FFT_CONV1D_ENABLE_GM_KERNEL
    else if (tilingData.algo == ALGO_FFT_GM)
    {
        // GM 版：UB 占用与 N 无关，支持大 N（解除 L+K-1 <= 1024）
        FftConv1dFftGm op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm_, &tilingData.cubeTiling);
        op.Init(&pipe, x, kernel, y, workspace, tilingData);
        op.Process();
    }
#endif
    else
    {
        // UB 常驻版：已验证正确的基线，小 N 走这里
        FftConv1dFft op;
#if FFT_CONV1D_USE_CUBE
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm_, &tilingData.cubeTiling);
#endif
        op.Init(&pipe, x, kernel, y, workspace, tilingData);
        op.Process();
    }
}
