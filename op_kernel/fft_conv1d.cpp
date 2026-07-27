/**
 * fft_conv1d AscendC kernel（v2，Ascend 910B / Atlas A2）
 *
 * 语义（唯一语义）：
 *   output[b,h,t] = sum_{j=0}^{K-1} input[b,h,t-j] * kernel[h,j]，t-j<0 时 input 视为 0
 *   因果 depthwise 卷积，通道独立，不做通道间求和。数学卷积，非 cross-correlation。
 *
 * 两条路径（host tiling 静态分派）：
 *   ALGO_DIRECT：K < 64，直接因果卷积，纯 Vector
 *   ALGO_FFT   ：K >= 64，四步 Cooley-Tukey，Cube 做 GEMM + Vector 做旋转因子/逐点乘
 *
 * ---------------------------------------------------------------------------
 * v2 相对 v1 的优化（每一项都标注了依据）
 * ---------------------------------------------------------------------------
 * [O1] 去掉逐行 SyncAll。依据：`IterateAll` 模板默认 sync=true，文档明确
 *      "需要同步等待 IterateAll 执行结束"，Cube/Vector 握手由 API 自己完成。
 *      且阶段 2 每个核只访问自己的 scratch，无跨核依赖。
 *      全 kernel 只保留 2 次 SyncAll（常量表就绪、kernel 频谱就绪），
 *      v1 是每行 15 次全局栅栏。
 *
 * [O2] 用 IterateAll 的 enAtomic=1（AtomicAdd）做累加，消除全部 Vector 合并 pass。
 *      Cr = Br@Dr - Bi@Di 写成 Gemm(Cr,Br,Dr,0) + Gemm(Cr,Bi,Dn,1)，其中 Dn = -Di；
 *      末级 y = (Dr@Er + Di@Ei)/N 写成 Gemm(y,DrS,Er,0) + Gemm(y,DiS,Ei,1)，
 *      其中 DrS = Dr/N、DiS = Di/N，把归一化也折进常量表。
 *      => 每行少 5 个 Vector 全长 pass、少 2 个 scratch 缓冲、少约 15N 字节 GM 往返。
 *      注意首次调用必须 enAtomic=0（覆盖写），否则会累加到上一行的残留值上。
 *
 * [O3] 只清零补零区 [L, N) 而不是整个 [0, N)，省掉一遍 N 长度的 GM 写。
 *
 * [O4] direct 路径用 8 份移位副本消除 Vector 非对齐访问：
 *      抽头 j 的源偏移是 K-1-j，逐 j 变化必然非 32B 对齐，v1 因此每个抽头都单独
 *      从 GM 读一整块（K 次读放大）。注意偏移同余 8 的抽头之间相差 8 个 float = 32B，
 *      天然对齐，故只需预载 8 份移位副本，K 次 GM 读降为 min(8,K) 次。
 *
 * [O5] ComplexMul 一次性发起 4 个 DataCopyPad 再同步，而不是两两同步。
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace
{
constexpr uint32_t ALGO_DIRECT = 0;
constexpr uint32_t ALGO_FFT = 1;

constexpr uint32_t SCRATCH_BUFS = 8;  // 与 host 侧 FFT_CONV1D_SCRATCH_BUFS 一致
constexpr uint32_t VEC_CHUNK = 2048;  // Vector 逐点运算的分块长度（float 个数）
constexpr uint32_t TMP_BUF_BYTES = 8192; // Cos/Sin/Fmod 需要的 sharedTmpBuffer
constexpr uint32_t SHIFT_COPIES = 8;  // direct 路径移位副本数（32B / 4B = 8）
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
// 路径一：直接因果卷积（纯 Vector）
// ============================================================================
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
        tile_ = t.tileLen;
        cores_ = t.usedCoreNum;
        rowLen_ = AlignUp8(K_ - 1 + L_);
        nShift_ = MinU(SHIFT_COPIES, K_);
        shiftCap_ = AlignUp8(tile_ + K_);

        xGm_.SetGlobalBuffer((__gm__ float *)x);
        wGm_.SetGlobalBuffer((__gm__ float *)w);
        yGm_.SetGlobalBuffer((__gm__ float *)y);
        wsGm_.SetGlobalBuffer((__gm__ float *)workspace);

        // [O4] nShift_ 份移位副本，每份容量 tile+K
        pipe->InitBuffer(bufShift_, nShift_ * shiftCap_ * sizeof(float));
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
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0); // 标量读取前必须等搬运完成
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    // 在 workspace 里物化 [K-1 个 0, x[row][0..L-1]]，
    // 使所有抽头的取数都退化成 GM 上的普通偏移（GM 地址无对齐约束）
    __aicore__ inline void BuildPaddedRow(uint32_t row, uint32_t base)
    {
        if (K_ > 1)
        {
            LocalTensor<float> zeros = bufZero_.Get<float>();
            DataCopyExtParams zp{1, (K_ - 1) * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(wsGm_[base], zeros, zp);
        }
        LocalTensor<float> buf = bufShift_.Get<float>();
        for (uint32_t t0 = 0; t0 < L_; t0 += tile_)
        {
            const uint32_t len = MinU(tile_, L_ - t0);
            DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(buf, xGm_[static_cast<uint64_t>(row) * L_ + t0], cp, pad);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopyPad(wsGm_[base + K_ - 1 + t0], buf, cp);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    __aicore__ inline void ComputeTile(uint32_t row, uint32_t base, uint32_t t0, uint32_t len)
    {
        LocalTensor<float> kb = bufK_.Get<float>();
        LocalTensor<float> out = bufOut_.Get<float>();
        LocalTensor<float> shift = bufShift_.Get<float>();

        // [O4] 预载 nShift_ 份移位副本：copy r 覆盖 xz[t0+r ...]
        // 抽头 j 的源在 xz 里的起点是 t0+K-1-j，落在副本 r=(K-1-j)%8 的
        // 第 (K-1-j-r) 个元素处，而 (K-1-j-r) 恒为 8 的倍数 => UB 侧 32B 对齐。
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        for (uint32_t r = 0; r < nShift_; ++r)
        {
            // 该副本需要覆盖的最大抽头偏移
            const uint32_t need = len + (K_ - 1 - r);
            DataCopyExtParams cp{1, need * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(shift[r * shiftCap_], wsGm_[base + t0 + r], cp, pad);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        Duplicate(out, 0.0f, AlignUp8(len));
        PipeBarrier<PIPE_V>();
        for (uint32_t j = 0; j < K_; ++j)
        {
            const uint32_t d = K_ - 1 - j;      // 源相对 t0 的偏移
            const uint32_t r = d % nShift_;     // 落在哪份副本
            const uint32_t off = r * shiftCap_ + (d - r); // (d-r) 是 nShift_ 的倍数
            Axpy(out, shift[off], kb.GetValue(j), len); // out += x[t0+..-j] * k[j]
        }

        // 1) V -> MTE3：out 算完才能搬出
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams op{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(yGm_[static_cast<uint64_t>(row) * L_ + t0], out, op);
        // 2) V -> MTE2：shift 被 Axpy 读完，下一轮才能重载移位副本
        SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        // 3) MTE3 -> V：out 搬完，下一轮才能 Duplicate 重写
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    GlobalTensor<float> xGm_, wGm_, yGm_, wsGm_;
    TBuf<TPosition::VECCALC> bufShift_, bufOut_, bufK_, bufZero_;
    uint32_t H_, L_, K_, rows_, tile_, cores_, rowLen_, nShift_, shiftCap_;
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
        N_ = t.nFft;
        N1_ = t.nRadix;
        cores_ = t.usedCoreNum;

        xGm_.SetGlobalBuffer((__gm__ float *)x);
        wGm_.SetGlobalBuffer((__gm__ float *)w);
        yGm_.SetGlobalBuffer((__gm__ float *)y);
        wsGm_.SetGlobalBuffer((__gm__ float *)workspace);

        // 常量表：Dr, Di, Dn(=-Di), DrS(=Dr/N), DiS(=Di/N), Tr, Ti
        offDr_ = 0;
        offDi_ = N_;
        offDn_ = 2ULL * N_;
        offDrS_ = 3ULL * N_;
        offDiS_ = 4ULL * N_;
        offTr_ = 5ULL * N_;
        offTi_ = 6ULL * N_;
        offKfr_ = 7ULL * N_;
        offKfi_ = 7ULL * N_ + static_cast<uint64_t>(H_) * N_;
        offScr_ = 7ULL * N_ + 2ULL * H_ * N_ +
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
        SyncAll(); // 跨核：所有核都要读全部常量表

        // 阶段 1：每个通道的 kernel 频谱（写 workspace，供所有 batch 复用）
        for (uint32_t h = static_cast<uint32_t>(GetBlockIdx()); h < H_; h += cores_)
        {
            PrepareRow(wGm_, static_cast<uint64_t>(h) * K_, K_);
            Forward(); // scr0 -> (scr3, scr4)
            CopyRange(offKfr_ + static_cast<uint64_t>(h) * N_, Scr(3), N_);
            CopyRange(offKfi_ + static_cast<uint64_t>(h) * N_, Scr(4), N_);
        }
        SyncAll(); // 跨核：核 A 写的 Kf[h] 会被核 B 读

        // 阶段 2：逐行做 FFT 卷积。每个核只碰自己的 scratch，无跨核依赖 => 无栅栏 [O1]
        for (uint32_t row = static_cast<uint32_t>(GetBlockIdx()); row < rows_; row += cores_)
        {
            PrepareRow(xGm_, static_cast<uint64_t>(row) * L_, L_);
            Forward();                       // scr0 -> (scr3, scr4)
            PointwiseWithKernel(row % H_);   // (scr3,scr4) *= Kf[h]
            Inverse();                       // (scr3,scr4) -> scr7
            WsToOut(static_cast<uint64_t>(row) * L_, Scr(7), L_); // 因果裁剪：取前 L 点
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
            const uint64_t rowOff = static_cast<uint64_t>(k) * N1_;

            // ---- D 的第 k 行（模 N1），同时派生 Dn / DrS / DiS ----
            Angle(idx, prod, modv, ang, tmp, len, k, static_cast<float>(N1_));
            Cos(re, ang, tmp, len);
            Sin(im, ang, tmp, len);
            PipeBarrier<PIPE_V>();
            Store(re, offDr_ + rowOff);
            Store(im, offDi_ + rowOff);
            // Dn = -Di，供 AtomicAdd 表达减法 [O2]
            Muls(prod, im, -1.0f, len);
            PipeBarrier<PIPE_V>();
            Store(prod, offDn_ + rowOff);
            // DrS = Dr/N、DiS = Di/N，把 IRFFT 的 1/N 归一化折进常量表 [O2]
            const float inv = 1.0f / static_cast<float>(N_);
            Muls(prod, re, inv, len);
            PipeBarrier<PIPE_V>();
            Store(prod, offDrS_ + rowOff);
            Muls(prod, im, inv, len);
            PipeBarrier<PIPE_V>();
            Store(prod, offDiS_ + rowOff);

            // ---- T 的第 k 行（模 N）----
            Angle(idx, prod, modv, ang, tmp, len, k, static_cast<float>(N_));
            Cos(re, ang, tmp, len);
            Sin(im, ang, tmp, len);
            PipeBarrier<PIPE_V>();
            Store(re, offTr_ + rowOff);
            Store(im, offTi_ + rowOff);
        }
    }

    // ang[n] = -2π·((k·n) mod mod)/mod
    __aicore__ inline void Angle(const LocalTensor<float> &idx, const LocalTensor<float> &prod,
                                 const LocalTensor<float> &modv, const LocalTensor<float> &ang,
                                 const LocalTensor<uint8_t> &tmp, uint32_t len, uint32_t k,
                                 float mod)
    {
        Muls(prod, idx, static_cast<float>(k), len); // k*n（整数，fp32 精确表示）
        PipeBarrier<PIPE_V>();
        Duplicate(modv, mod, len);
        PipeBarrier<PIPE_V>();
        Fmod(prod, prod, modv, tmp, len); // (k*n) mod M ∈ [0, M)
        PipeBarrier<PIPE_V>();
        Muls(ang, prod, -TWO_PI / mod, len); // 角度落在 (-2π, 0]
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Store(const LocalTensor<float> &src, uint64_t off)
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyExtParams cp{1, N1_ * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(wsGm_[off], src, cp);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    // ---------------- 输入准备 ----------------
    // scr0 = [src 的 count 个元素, 0, ..., 0]，总长 N
    // [O3] 只清 [count, N) 这段补零区，不再整段清零
    __aicore__ inline void PrepareRow(const GlobalTensor<float> &src, uint64_t srcOff,
                                      uint32_t count)
    {
        LocalTensor<float> b = bA_.Get<float>();
        // 先把有效数据搬过去
        for (uint32_t o = 0; o < count; o += VEC_CHUNK)
        {
            const uint32_t len = MinU(VEC_CHUNK, count - o);
            DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(b, src[srcOff + o], cp, pad);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopyPad(wsGm_[Scr(0) + o], b, cp);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        // 再补零尾部
        Duplicate(b, 0.0f, AlignUp8(VEC_CHUNK));
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        for (uint32_t o = count; o < N_; o += VEC_CHUNK)
        {
            const uint32_t len = MinU(VEC_CHUNK, N_ - o);
            DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(wsGm_[Scr(0) + o], b, cp);
        }
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }

    // ---------------- GEMM 封装 ----------------
    // atomic=0 覆盖写，atomic=1 AtomicAdd 累加 [O2]
    __aicore__ inline void Gemm(uint64_t oC, uint64_t oA, uint64_t oB, uint8_t atomic)
    {
        mm_.SetTensorA(wsGm_[oA]);
        mm_.SetTensorB(wsGm_[oB]);
        mm_.IterateAll(wsGm_[oC], atomic); // 默认 sync=true，返回即完成 [O1]
        mm_.End();
    }

    // ---------------- Vector 逐点运算 ----------------
    __aicore__ inline void CopyRange(uint64_t oDst, uint64_t oSrc, uint32_t count)
    {
        LocalTensor<float> a = bA_.Get<float>();
        for (uint32_t o = 0; o < count; o += VEC_CHUNK)
        {
            const uint32_t len = MinU(VEC_CHUNK, count - o);
            DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            DataCopyPad(a, wsGm_[oSrc + o], cp, pad);
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            DataCopyPad(wsGm_[oDst + o], a, cp);
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
    }

    // (ar, ai) = (ar, ai) * (br, bi)，conjB 时用 (br, -bi)。4 次实乘。
    // 选 4 乘而非 Karatsuba 3 乘：Vector 的乘加吞吐不是瓶颈（瓶颈在搬运），
    // 3 乘要多 2 个临时缓冲和 5 次加减，UB 压力与指令数反而更差。
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
            DataCopyExtParams cp{1, len * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            // [O5] 4 个搬运一起发，只同步一次
            DataCopyPad(ar, wsGm_[oAr + o], cp, pad);
            DataCopyPad(ai, wsGm_[oAi + o], cp, pad);
            DataCopyPad(br, wsGm_[oBr + o], cp, pad);
            DataCopyPad(bi, wsGm_[oBi + o], cp, pad);
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

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
            // 虚部：ar*bi + ai*br（ar 算完 ar*bi 后即失效，可复用为临时量）
            Mul(ti, ar, bi, len);
            PipeBarrier<PIPE_V>();
            Mul(ar, ai, br, len);
            PipeBarrier<PIPE_V>();
            Add(ti, ti, ar, len);

            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(wsGm_[oAr + o], tr, cp);
            DataCopyPad(wsGm_[oAi + o], ti, cp);
            SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

    // ---------------- 正变换：scr0（实，已补零）-> (scr3, scr4) ----------------
    __aicore__ inline void Forward()
    {
        // 步骤 A：Br = Dr @ xmat, Bi = Di @ xmat（实输入 => 只要 2 次实数 GEMM）
        Gemm(Scr(1), offDr_, Scr(0), 0);
        Gemm(Scr(2), offDi_, Scr(0), 0);
        // 步骤 B：乘旋转因子 W_N^{k1·n2}
        ComplexMulInPlace(Scr(1), Scr(2), offTr_, offTi_, false);
        // 步骤 C：Cr = Br@Dr - Bi@Di，Ci = Br@Di + Bi@Dr（D 对称，无需转置）
        // 减法用 Dn = -Di 配合 AtomicAdd 表达，省掉 Vector 合并 pass [O2]
        Gemm(Scr(3), Scr(1), offDr_, 0);
        Gemm(Scr(3), Scr(2), offDn_, 1);
        Gemm(Scr(4), Scr(1), offDi_, 0);
        Gemm(Scr(4), Scr(2), offDr_, 1);
    }

    // ---------------- 频域逐点乘 ----------------
    __aicore__ inline void PointwiseWithKernel(uint32_t h)
    {
        ComplexMulInPlace(Scr(3), Scr(4), offKfr_ + static_cast<uint64_t>(h) * N_,
                          offKfi_ + static_cast<uint64_t>(h) * N_, false);
    }

    // ---------------- 逆变换：(scr3, scr4) -> scr7（实） ----------------
    __aicore__ inline void Inverse()
    {
        // 步骤 A'：Er = Yr@Dr + Yi@Di，Ei = Yi@Dr - Yr@Di（conj(D) = (Dr, -Di)）
        Gemm(Scr(5), Scr(3), offDr_, 0);
        Gemm(Scr(5), Scr(4), offDi_, 1);
        Gemm(Scr(6), Scr(4), offDr_, 0);
        Gemm(Scr(6), Scr(3), offDn_, 1);
        // 步骤 B'：乘共轭旋转因子
        ComplexMulInPlace(Scr(5), Scr(6), offTr_, offTi_, true);
        // 步骤 C'：y = (Dr@Er + Di@Ei)/N，用 DrS/DiS 把 1/N 折进常量表 [O2]
        Gemm(Scr(7), offDrS_, Scr(5), 0);
        Gemm(Scr(7), offDiS_, Scr(6), 1);
    }

    // ---------------- 输出裁剪：取线性卷积前 L 点，偏移为 0 ----------------
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
    uint32_t H_, L_, K_, rows_, N_, N1_, cores_;
    uint64_t offDr_, offDi_, offDn_, offDrS_, offDiS_, offTr_, offTi_;
    uint64_t offKfr_, offKfi_, offScr_;
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
