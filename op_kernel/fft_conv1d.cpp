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
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace
{
constexpr uint32_t ALGO_DIRECT = 0;
constexpr uint32_t ALGO_FFT = 1;

constexpr uint32_t SCRATCH_BUFS = 10; // 与 host 侧 FFT_CONV1D_SCRATCH_BUFS 保持一致
constexpr uint32_t VEC_CHUNK = 2048;  // Vector 逐点运算的分块长度（float 个数）
constexpr uint32_t TMP_BUF_BYTES = 8192; // Cos/Sin/Fmod 需要的 sharedTmpBuffer
constexpr float TWO_PI = 6.283185307179586f;

__aicore__ inline uint32_t AlignUp8(uint32_t x)
{
    return (x + 7U) & ~7U;
}

__aicore__ inline uint32_t MinU(uint32_t a, uint32_t b)
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
        flip_ = t.flipKernel;
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

        const uint32_t base = static_cast<uint32_t>(GetBlockIdx()) * rowLen_;
        for (uint32_t row = static_cast<uint32_t>(GetBlockIdx()); row < rows_; row += cores_)
        {
            // input 布局 [B,H,L]，行号 row = b*H + h => 通道号 h = row % H
            LoadKernelRow(row % H_);
            BuildPaddedRow(row, base);
            for (uint32_t t0 = 0; t0 < L_; t0 += tile_)
            {
                ComputeTile(row, base, t0, MinU(tile_, L_ - t0));
            }
        }
    }

  private:
    // 把 kernel 的第 h 行搬进 UB，后续用 GetValue 取标量
    __aicore__ inline void LoadKernelRow(uint32_t h)
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
    __aicore__ inline void BuildPaddedRow(uint32_t row, uint32_t base)
    {
        if (K_ > 1)
        {
            LocalTensor<float> zeros = bufZero_.Get<float>();
            DataCopyExtParams zp{1, (K_ - 1) * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(wsGm_[base], zeros, zp);
        }
        LocalTensor<float> in = bufIn_.Get<float>();
        for (uint32_t t0 = 0; t0 < L_; t0 += tile_)
        {
            const uint32_t len = MinU(tile_, L_ - t0);
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

    __aicore__ inline void ComputeTile(uint32_t row, uint32_t base, uint32_t t0, uint32_t len)
    {
        LocalTensor<float> kb = bufK_.Get<float>();
        LocalTensor<float> out = bufOut_.Get<float>();
        LocalTensor<float> xin = bufIn_.Get<float>();

        Duplicate(out, 0.0f, AlignUp8(len));
        PipeBarrier<PIPE_V>();

        for (uint32_t j = 0; j < K_; ++j)
        {
            // math 语义取 kernel[j]；corr 语义取 kernel[K-1-j]（等价于时域翻转）
            const float kv = kb.GetValue(flip_ != 0 ? (K_ - 1 - j) : j);
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
    uint32_t H_, L_, K_, rows_, flip_, tile_, cores_, rowLen_;
};

// ============================================================================
// 路径二：四步 Cooley-Tukey FFT 卷积（Cube GEMM + Vector 逐点）
// ============================================================================
class FftConv1dFft
{
  public:
    // 12 次 GEMM 形状统一为 (N1, N1, N1)，共用这一个对象
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>>
        mm_;

    __aicore__ inline FftConv1dFft()
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
        flip_ = t.flipKernel;
        N_ = t.nFft;
        N1_ = t.nRadix;
        cores_ = t.usedCoreNum;

        xGm_.SetGlobalBuffer((__gm__ float *)x);
        wGm_.SetGlobalBuffer((__gm__ float *)w);
        yGm_.SetGlobalBuffer((__gm__ float *)y);
        wsGm_.SetGlobalBuffer((__gm__ float *)workspace);

        // workspace 布局（float 为单位），与 host 侧一致
        offDr_ = 0;
        offDi_ = N_;
        offTr_ = 2ULL * N_;
        offTi_ = 3ULL * N_;
        offKfr_ = 4ULL * N_;
        offKfi_ = 4ULL * N_ + static_cast<uint64_t>(H_) * N_;
        offScr_ = 4ULL * N_ + 2ULL * H_ * N_ +
                  static_cast<uint64_t>(GetBlockIdx()) * SCRATCH_BUFS * N_;

        const uint32_t chunk = AlignUp8(VEC_CHUNK) * sizeof(float);
        pipe->InitBuffer(bA_, chunk);
        pipe->InitBuffer(bB_, chunk);
        pipe->InitBuffer(bC_, chunk);
        pipe->InitBuffer(bD_, chunk);
        pipe->InitBuffer(bE_, chunk);
        pipe->InitBuffer(bF_, chunk);
        pipe->InitBuffer(bTmp_, TMP_BUF_BYTES);
    }

    __aicore__ inline void Process()
    {
        BuildTables();
        SyncAll();

        // 阶段 1：每个通道的 kernel 频谱（写入 workspace，供所有 batch 复用）
        const uint32_t chPerCore = (H_ + cores_ - 1) / cores_;
        for (uint32_t i = 0; i < chPerCore; ++i)
        {
            const uint32_t h = static_cast<uint32_t>(GetBlockIdx()) + i * cores_;
            const bool active = (h < H_);
            PrepareKernelRow(active ? h : 0, active);
            Forward(); // scr0 -> (scr5, scr6)
            if (active)
            {
                CopyRange(offKfr_ + static_cast<uint64_t>(h) * N_, Scr(5), N_);
                // corr 语义：用 conj(K̂) 代替翻转 kernel（等价性已在 python 侧用
                // float64 验证到 1e-14，见 docs/03_ascendc_v1.md §3）
                if (flip_ != 0)
                {
                    NegRange(offKfi_ + static_cast<uint64_t>(h) * N_, Scr(6), N_);
                }
                else
                {
                    CopyRange(offKfi_ + static_cast<uint64_t>(h) * N_, Scr(6), N_);
                }
            }
            SyncAll();
        }

        // 阶段 2：逐行做 FFT 卷积
        const uint32_t rowsPerCore = (rows_ + cores_ - 1) / cores_;
        for (uint32_t i = 0; i < rowsPerCore; ++i)
        {
            const uint32_t row = static_cast<uint32_t>(GetBlockIdx()) + i * cores_;
            const bool active = (row < rows_);
            PrepareDataRow(active ? row : 0, active);
            Forward();                       // scr0 -> (scr5, scr6)
            PointwiseWithKernel(active ? (row % H_) : 0);
            Inverse();                       // (scr5,scr6) -> scr9
            if (active)
            {
                StoreOutput(row);
            }
            SyncAll();
        }
    }

  private:
    __aicore__ inline uint64_t Scr(uint32_t idx) const
    {
        return offScr_ + static_cast<uint64_t>(idx) * N_;
    }

    // ---------------- 常量表生成 ----------------
    // D[k][n] = exp(-2πi·((k·n) mod N1)/N1)，k,n ∈ [0,N1)
    // T[k1][n2] = exp(-2πi·((k1·n2) mod N)/N)，存放在 k1*N1 + n2
    //
    // 必须"先做整数乘再取模"：浮点乘法不满足结合律，若写成 (-2π·k)·n/M 会让
    // 本应对称的 DFT 矩阵出现 ~1e-14 不对称，破坏"D 对称所以不用转置"的前提；
    // 取模同时把角度压回 [-2π,0)，避免大幅角三角函数的规约误差。
    // （这个坑是在 python 原型阶段实测踩到并修掉的）
    __aicore__ inline void BuildTables()
    {
        LocalTensor<float> idx = bA_.Get<float>();
        LocalTensor<float> prod = bB_.Get<float>();
        LocalTensor<float> modv = bC_.Get<float>();
        LocalTensor<float> ang = bD_.Get<float>();
        LocalTensor<float> re = bE_.Get<float>();
        LocalTensor<float> im = bF_.Get<float>();
        LocalTensor<uint8_t> tmp = bTmp_.Get<uint8_t>();

        const uint32_t len = AlignUp8(N1_);
        CreateVecIndex(idx, 0.0f, len); // idx[n] = n
        PipeBarrier<PIPE_V>();

        for (uint32_t k = static_cast<uint32_t>(GetBlockIdx()); k < N1_; k += cores_)
        {
            // ---- D 的第 k 行，模 N1 ----
            EmitRow(idx, prod, modv, ang, re, im, tmp, len, k,
                    static_cast<float>(N1_), offDr_ + static_cast<uint64_t>(k) * N1_,
                    offDi_ + static_cast<uint64_t>(k) * N1_);
            // ---- T 的第 k 行，模 N ----
            EmitRow(idx, prod, modv, ang, re, im, tmp, len, k,
                    static_cast<float>(N_), offTr_ + static_cast<uint64_t>(k) * N1_,
                    offTi_ + static_cast<uint64_t>(k) * N1_);
        }
    }

    __aicore__ inline void EmitRow(const LocalTensor<float> &idx, const LocalTensor<float> &prod,
                                   const LocalTensor<float> &modv, const LocalTensor<float> &ang,
                                   const LocalTensor<float> &re, const LocalTensor<float> &im,
                                   const LocalTensor<uint8_t> &tmp, uint32_t len, uint32_t k,
                                   float mod, uint64_t offR, uint64_t offI)
    {
        Muls(prod, idx, static_cast<float>(k), len); // prod = k*n（整数，fp32 精确表示）
        PipeBarrier<PIPE_V>();
        Duplicate(modv, mod, len);
        PipeBarrier<PIPE_V>();
        Fmod(prod, prod, modv, tmp, len); // prod = (k*n) mod M ∈ [0, M)
        PipeBarrier<PIPE_V>();
        Muls(ang, prod, -TWO_PI / mod, len); // 角度落在 (-2π, 0]
        PipeBarrier<PIPE_V>();
        Cos(re, ang, tmp, len);
        Sin(im, ang, tmp, len);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, N1_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm_[offR], re, cp);
        DataCopyPad(wsGm_[offI], im, cp);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    // ---------------- 输入准备 ----------------
    // scr0 = [行数据, 0, ..., 0]，长度 N
    __aicore__ inline void FillZero(uint64_t off, uint32_t count)
    {
        LocalTensor<float> z = bA_.Get<float>();
        Duplicate(z, 0.0f, AlignUp8(VEC_CHUNK));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        for (uint32_t o = 0; o < count; o += VEC_CHUNK)
        {
            const uint32_t len = MinU(VEC_CHUNK, count - o);
            DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(wsGm_[off + o], z, cp);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    __aicore__ inline void GmToGm(uint64_t dstOff, const GlobalTensor<float> &src,
                                  uint64_t srcOff, uint32_t count)
    {
        LocalTensor<float> b = bA_.Get<float>();
        for (uint32_t o = 0; o < count; o += VEC_CHUNK)
        {
            const uint32_t len = MinU(VEC_CHUNK, count - o);
            DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(b, src[srcOff + o], cp, pad);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopyPad(wsGm_[dstOff + o], b, cp);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void PrepareKernelRow(uint32_t h, bool active)
    {
        FillZero(Scr(0), N_);
        if (active)
        {
            // flip_ 语义不在这里做翻转：改用 conj(K̂) + 输出循环移位，见 Process()
            GmToGm(Scr(0), wGm_, static_cast<uint64_t>(h) * K_, K_);
        }
    }

    __aicore__ inline void PrepareDataRow(uint32_t row, bool active)
    {
        FillZero(Scr(0), N_);
        if (active)
        {
            GmToGm(Scr(0), xGm_, static_cast<uint64_t>(row) * L_, L_);
        }
    }

    // ---------------- GEMM 封装 ----------------
    __aicore__ inline void Gemm(uint64_t oC, uint64_t oA, uint64_t oB)
    {
        mm_.SetTensorA(wsGm_[oA]);
        mm_.SetTensorB(wsGm_[oB]);
        mm_.IterateAll(wsGm_[oC]);
        mm_.End();
    }

    // ---------------- Vector 逐点运算（全部经 UB 分块）----------------
    // dst = a + b
    __aicore__ inline void AddTo(uint64_t oDst, uint64_t oA, uint64_t oB)
    {
        BinaryOp(oDst, oA, oB, 0);
    }
    // dst = a - b
    __aicore__ inline void SubTo(uint64_t oDst, uint64_t oA, uint64_t oB)
    {
        BinaryOp(oDst, oA, oB, 1);
    }

    __aicore__ inline void BinaryOp(uint64_t oDst, uint64_t oA, uint64_t oB, uint32_t kind)
    {
        LocalTensor<float> a = bA_.Get<float>();
        LocalTensor<float> b = bB_.Get<float>();
        LocalTensor<float> c = bC_.Get<float>();
        for (uint32_t o = 0; o < N_; o += VEC_CHUNK)
        {
            const uint32_t len = MinU(VEC_CHUNK, N_ - o);
            LoadTwo(a, b, oA + o, oB + o, len);
            if (kind == 0)
            {
                Add(c, a, b, len);
            }
            else
            {
                Sub(c, a, b, len);
            }
            StoreOne(c, oDst + o, len);
        }
    }

    // dst = (a + b) * s
    __aicore__ inline void AddScaleTo(uint64_t oDst, uint64_t oA, uint64_t oB, float s)
    {
        LocalTensor<float> a = bA_.Get<float>();
        LocalTensor<float> b = bB_.Get<float>();
        LocalTensor<float> c = bC_.Get<float>();
        for (uint32_t o = 0; o < N_; o += VEC_CHUNK)
        {
            const uint32_t len = MinU(VEC_CHUNK, N_ - o);
            LoadTwo(a, b, oA + o, oB + o, len);
            Add(c, a, b, len);
            PipeBarrier<PIPE_V>();
            Muls(c, c, s, len);
            StoreOne(c, oDst + o, len);
        }
    }

    __aicore__ inline void CopyRange(uint64_t oDst, uint64_t oSrc, uint32_t count)
    {
        LocalTensor<float> a = bA_.Get<float>();
        for (uint32_t o = 0; o < count; o += VEC_CHUNK)
        {
            const uint32_t len = MinU(VEC_CHUNK, count - o);
            LoadOne(a, oSrc + o, len);
            StoreOne(a, oDst + o, len);
        }
    }

    __aicore__ inline void NegRange(uint64_t oDst, uint64_t oSrc, uint32_t count)
    {
        LocalTensor<float> a = bA_.Get<float>();
        for (uint32_t o = 0; o < count; o += VEC_CHUNK)
        {
            const uint32_t len = MinU(VEC_CHUNK, count - o);
            LoadOne(a, oSrc + o, len);
            Muls(a, a, -1.0f, len);
            StoreOne(a, oDst + o, len);
        }
    }

    // (ar, ai) = (ar, ai) * (br, bi)，conjB 时用 (br, -bi)。4 次实乘，见设计文档 §6.4
    __aicore__ inline void ComplexMulInPlace(uint64_t oAr, uint64_t oAi, uint64_t oBr,
                                             uint64_t oBi, bool conjB)
    {
        LocalTensor<float> ar = bA_.Get<float>();
        LocalTensor<float> ai = bB_.Get<float>();
        LocalTensor<float> br = bC_.Get<float>();
        LocalTensor<float> bi = bD_.Get<float>();
        LocalTensor<float> tr = bE_.Get<float>();
        LocalTensor<float> ti = bF_.Get<float>();

        for (uint32_t o = 0; o < N_; o += VEC_CHUNK)
        {
            const uint32_t len = MinU(VEC_CHUNK, N_ - o);
            LoadTwo(ar, ai, oAr + o, oAi + o, len);
            LoadTwo(br, bi, oBr + o, oBi + o, len);
            if (conjB)
            {
                Muls(bi, bi, -1.0f, len);
                PipeBarrier<PIPE_V>();
            }
            // 实部：ar*br - ai*bi
            Mul(tr, ar, br, len);
            Mul(ti, ai, bi, len);
            PipeBarrier<PIPE_V>();
            Sub(tr, tr, ti, len);
            PipeBarrier<PIPE_V>();
            // 虚部：ar*bi + ai*br（ar 在算完 ar*bi 后即失效，可安全复用为临时量）
            Mul(ti, ar, bi, len);
            PipeBarrier<PIPE_V>();
            Mul(ar, ai, br, len);
            PipeBarrier<PIPE_V>();
            Add(ti, ti, ar, len);
            StoreOne(tr, oAr + o, len);
            StoreOne(ti, oAi + o, len);
        }
    }

    __aicore__ inline void LoadOne(const LocalTensor<float> &d, uint64_t off, uint32_t len)
    {
        DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(d, wsGm_[off], cp, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void LoadTwo(const LocalTensor<float> &d0, const LocalTensor<float> &d1,
                                   uint64_t o0, uint64_t o1, uint32_t len)
    {
        DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(d0, wsGm_[o0], cp, pad);
        DataCopyPad(d1, wsGm_[o1], cp, pad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void StoreOne(const LocalTensor<float> &s, uint64_t off, uint32_t len)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm_[off], s, cp);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    // ---------------- 正变换：scr0（实，已补零）-> (scr5, scr6) ----------------
    __aicore__ inline void Forward()
    {
        // 步骤 A：Br = Dr @ xmat, Bi = Di @ xmat（实输入 => 只要 2 次实数 GEMM）
        Gemm(Scr(1), offDr_, Scr(0));
        Gemm(Scr(2), offDi_, Scr(0));
        SyncAll();
        // 步骤 B：乘旋转因子 W_N^{k1·n2}
        ComplexMulInPlace(Scr(1), Scr(2), offTr_, offTi_, false);
        SyncAll();
        // 步骤 C：Cr = Br@D - Bi@Di，Ci = Br@Di + Bi@Dr（D 对称，无需转置）
        Gemm(Scr(3), Scr(1), offDr_);
        Gemm(Scr(4), Scr(2), offDi_);
        SyncAll();
        SubTo(Scr(5), Scr(3), Scr(4));
        SyncAll();
        Gemm(Scr(3), Scr(1), offDi_);
        Gemm(Scr(4), Scr(2), offDr_);
        SyncAll();
        AddTo(Scr(6), Scr(3), Scr(4));
        SyncAll();
    }

    // ---------------- 频域逐点乘 ----------------
    __aicore__ inline void PointwiseWithKernel(uint32_t h)
    {
        ComplexMulInPlace(Scr(5), Scr(6), offKfr_ + static_cast<uint64_t>(h) * N_,
                          offKfi_ + static_cast<uint64_t>(h) * N_, false);
        SyncAll();
    }

    // ---------------- 逆变换：(scr5, scr6) -> scr9（实） ----------------
    __aicore__ inline void Inverse()
    {
        // 步骤 A'：Er = Yr@Dr + Yi@Di，Ei = Yi@Dr - Yr@Di（conj(D) = (Dr, -Di)）
        Gemm(Scr(3), Scr(5), offDr_);
        Gemm(Scr(4), Scr(6), offDi_);
        SyncAll();
        AddTo(Scr(7), Scr(3), Scr(4));
        SyncAll();
        Gemm(Scr(3), Scr(5), offDi_);
        Gemm(Scr(4), Scr(6), offDr_);
        SyncAll();
        SubTo(Scr(8), Scr(4), Scr(3));
        SyncAll();
        // 步骤 B'：乘共轭旋转因子
        ComplexMulInPlace(Scr(7), Scr(8), offTr_, offTi_, true);
        SyncAll();
        // 步骤 C'：y = (Dr@Er + Di@Ei)/N（实输出 => 只取实部，2 次实数 GEMM）
        Gemm(Scr(3), offDr_, Scr(7));
        Gemm(Scr(4), offDi_, Scr(8));
        SyncAll();
        AddScaleTo(Scr(9), Scr(3), Scr(4), 1.0f / static_cast<float>(N_));
        SyncAll();
    }

    // ---------------- 输出裁剪 ----------------
    // math 语义：y[0..L-1] = ifft[0..L-1]（因果卷积取线性卷积前 L 点，偏移为 0）
    // corr 语义：等价于循环相关右移 K-1，输出由两段拼成
    //            y[0..K-2]   = ifft[N-K+1..N-1]
    //            y[K-1..L-1] = ifft[0..L-K]
    __aicore__ inline void StoreOutput(uint32_t row)
    {
        const uint64_t yBase = static_cast<uint64_t>(row) * L_;
        if (flip_ == 0)
        {
            WsToOut(yBase, Scr(9), L_);
        }
        else
        {
            if (K_ > 1)
            {
                WsToOut(yBase, Scr(9) + (N_ - K_ + 1), K_ - 1);
            }
            WsToOut(yBase + (K_ - 1), Scr(9), L_ - K_ + 1);
        }
    }

    __aicore__ inline void WsToOut(uint64_t yOff, uint64_t wsOff, uint32_t count)
    {
        LocalTensor<float> a = bA_.Get<float>();
        for (uint32_t o = 0; o < count; o += VEC_CHUNK)
        {
            const uint32_t len = MinU(VEC_CHUNK, count - o);
            DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(a, wsGm_[wsOff + o], cp, pad);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopyPad(yGm_[yOff + o], a, cp);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    GlobalTensor<float> xGm_, wGm_, yGm_, wsGm_;
    TBuf<TPosition::VECCALC> bA_, bB_, bC_, bD_, bE_, bF_, bTmp_;
    uint32_t H_, L_, K_, rows_, flip_, N_, N1_, cores_;
    uint64_t offDr_, offDi_, offTr_, offTi_, offKfr_, offKfi_, offScr_;
};

// ============================================================================
// kernel 入口
// ============================================================================
extern "C" __global__ __aicore__ void fft_conv1d(GM_ADDR x, GM_ADDR kernel, GM_ADDR y,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    TPipe pipe;
    if (tilingData.algo == ALGO_DIRECT)
    {
        FftConv1dDirect op;
        op.Init(&pipe, x, kernel, y, workspace, tilingData);
        op.Process();
    }
    else
    {
        FftConv1dFft op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm_, &tilingData.cubeTiling);
        op.Init(&pipe, x, kernel, y, workspace, tilingData);
        op.Process();
    }
}
