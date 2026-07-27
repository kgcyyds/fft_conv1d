# fft_conv1d 阶段 3/5：AscendC 实现与优化记录

> 对应代码：`op_host/fft_conv1d.cpp`、`op_host/fft_conv1d_tiling.h`、`op_kernel/fft_conv1d.cpp`
> 逻辑守护测试：`tests/test_stage3_kernel_logic.py`（229 个用例全过，无需 NPU）

---

## 1. 算子接口（v2 起唯一语义，无属性）

```
输入  x      : [B, H, L]  float32  ND 连续
      kernel : [H, K]     float32  ND 连续
输出  y      : [B, H, L]  float32  ND 连续
属性  无
```

```
y[b,h,t] = Σ_{j=0}^{K-1} x[b,h,t-j] · kernel[h,j]，  t-j < 0 时 x 视为 0
```

**这是数学卷积，不是 cross-correlation。** 若调用方的参考实现写成

```python
F.conv1d(F.pad(x, (K-1, 0)), kernel.unsqueeze(1), groups=H)
```

那算的是 kernel 时域翻转后的结果，需要调用方自行 `kernel.flip(-1)` 后再调用本算子。
v1 曾提供 `flip_kernel` 属性覆盖这两种语义，v2 按需求确认结果删除，接口只留输入输出。

### 约束（host 校验，失败时打印具体原因）

| 约束 | 值 |
|---|---|
| dtype | float32 |
| 维度 | x `[B,H,L]`、kernel `[H,K]`，H 必须一致 |
| 取值 | `B,H,L,K ≥ 1`，`K ≤ L` |
| 长度 | `L + K - 1 ≤ 16384`（`FFT_CONV1D_MAX_NFFT`） |
| 内存 | 连续（contiguous） |

---

## 2. 两条路径

host tiling 静态分派，kernel 侧不做动态判断：

| 条件 | 路径 | 单元 |
|---|---|---|
| `K < 64` | 直接因果卷积 | 纯 Vector |
| `K ≥ 64` | 四步 Cooley-Tukey FFT 卷积 | Cube GEMM + Vector |

分界点 64 目前是按阶段 2 的代价模型估的，**必须用 msprof 实测标定**（依赖 Cube/Vector 算力比 ρ，那个数只能实测）。

### FFT 分解的取舍

`N` 取 **4 的幂**（`N = 4^⌈log4(L+K-1)⌉`），于是 `N1 = N2 = √N`：

- `D1` 与 `D2` 是同一个矩阵（DFT 矩阵对称 + `N1 == N2`），常量表减半
- **12 次 GEMM 形状统一为 `(N1, N1, N1)`**，只需一个 Matmul 对象、一套 `TCubeTiling`

代价：`N` 最多浪费 2 倍，MAC 数约为最优解的 1.5~4 倍（见 §5 待办）。
换来的是 host/kernel 代码量和出错面大幅下降 —— 这是首个可上机版本的正确取舍。
结构上仍是四步 Cooley-Tukey，后续放开是**优化而非重写**。

---

## 3. v2 优化项（每项都有依据与验证）

### [O1] 去掉逐行 `SyncAll`：每行 15 次 → 全 kernel 2 次

v1 在每个 Cube→Vector 边界都插了 `SyncAll()` 全局栅栏，因为当时无法上机确认
`IterateAll` 之后 Cube 写 GM、Vector 再读 GM 的跨单元同步语义。

依据（CANN 文档 `5.2.1.23 IterateAll`）：

```cpp
template <bool sync = true> __aicore__ inline void IterateAll(
    const GlobalTensor<DstT>& gm, uint8_t enAtomic = 0, ...)
```

> **同步：** 需要同步等待 IterateAll 执行结束

默认 `sync = true`，握手由 API 自己完成。加上阶段 2 每个核只访问自己的 scratch、
无跨核依赖，逐行栅栏纯属浪费。v2 只保留 2 次 `SyncAll`：

1. 常量表生成完毕（所有核都要读全部表）
2. kernel 频谱生成完毕（核 A 写的 `Kf[h]` 会被核 B 读）

副产品：不再需要"所有核循环次数一致"的约束，两个循环都回归自然的
`for (i = blockIdx; i < n; i += cores)` 跨步写法，空闲核不再空转。

### [O2] 用 `IterateAll(enAtomic=1)` 累加，消除全部 Vector 合并 pass

`enAtomic = 1` 是 AtomicAdd 累加（文档同章节参数表）。于是

```
Cr = Br@Dr - Bi@Di   →   Gemm(Cr, Br, Dr, atomic=0); Gemm(Cr, Bi, Dn, atomic=1)
y  = (Dr@Er + Di@Ei)/N →  Gemm(y, DrS, Er, atomic=0); Gemm(y, DiS, Ei, atomic=1)
```

新增 3 张常量表把减法和归一化都吃进去：

| 表 | 定义 | 作用 |
|---|---|---|
| `Dn` | `-Di` | 让只会累加的 AtomicAdd 表达减法 |
| `DrS` | `Dr / N` | 把 IRFFT 的 `1/N` 折进常量表 |
| `DiS` | `Di / N` | 同上 |

> 首次调用必须 `atomic=0`（覆盖写），否则会累加到上一行的残留值上。

**等价性是逐位精确的**，不是近似：取负 `(-a)·b = -(a·b)` 精确；`N` 为 2 的幂
所以 `Dr/N` 也精确。`test_atomic_refactor_is_bit_exact` 用 `rtol=0, atol=0` 守护，
实测 v2 与 v1 差异 **0.0e+00**。

收益：每行 Vector 全长 pass **8 → 3**（只剩 2 次旋转因子 + 1 次频域逐点乘），
scratch 缓冲 **10 → 8**，少约 15N 字节 GM 往返。

### [O3] 只清补零区

`PrepareRow` 只清 `[count, N)`，不再整段清 `[0, N)`，省一遍 N 长度的 GM 写。

### [O4] direct 路径：8 份移位副本，GM 读 `K` 次 → `min(8,K)` 次

抽头 `j` 的源偏移是 `K-1-j`，逐 `j` 变化必然不满足 Vector 要求的 32B 对齐，
v1 因此每个抽头都单独从 GM 读一整块（K 倍读放大）。

关键观察：**偏移同余 8 的抽头之间相差 8 个 float = 32B，天然对齐**。
所以只需预载 `min(8,K)` 份移位副本，副本 `r` 覆盖 `xz[t0+r ...]`，
抽头 `j` 落在副本 `r = (K-1-j) % 8` 的第 `(K-1-j-r)` 个元素处，而该偏移恒为 8 的倍数。

`K=63` 时 GM 读次数 63 → 8。`test_direct_shift_copy_is_exact` 逐位（`rtol=0, atol=0`）
守护下标正确性，并用断言覆盖 32B 对齐与 workspace 越界。

### [O5] ComplexMul 批量发搬运

4 个 `DataCopyPad` 一起发再统一同步，而不是两两同步。

---

## 4. workspace 布局

FFT 路径（单位：float）：

```
[0]   Dr   : N        (N1*N1 == N)
[N]   Di   : N
[2N]  Dn   : N        = -Di
[3N]  DrS  : N        = Dr / N
[4N]  DiS  : N        = Di / N
[5N]  Tr   : N
[6N]  Ti   : N
[7N]      Kfr : H*N
[7N+HN]   Kfi : H*N
[7N+2HN]  每核 scratch : usedCoreNum * 8 * N
          （0:xmat 1:Br 2:Bi 3:Cr 4:Ci 5:Er 6:Ei 7:yout）
```

direct 路径：每核一份零前缀输入行 `AlignUp8(K-1+L)`。

**旋转因子生成**：必须"先做整数乘再取模"（`-2π·((k·n) mod M)/M`）。
写成 `(-2π·k)·n/M` 这种浮点连乘会因为乘法不满足结合律，让本应对称的 DFT 矩阵
出现 ~1e-14 不对称，破坏"D 对称所以不用转置"的前提。取模同时把角度压回 `(-2π, 0]`，
避免大幅角三角函数的规约误差。这个坑是在 Python 原型阶段实测踩到并修掉的。

---

## 5. 尚未做的优化（按性价比排序）

| # | 项 | 预期 | 说明 |
|---|---|---|---|
| 1 | 中间结果留 UB | ~10x | 现在算术强度约 3.4 MAC/byte，需 30+ 才算力受限。`N=4096` 时频谱两平面才 32KB，可全程驻 UB；Matmul 的 A 矩阵设 `TPosition::VECOUT` 直接喂 Cube |
| 2 | 换回 2 的幂 + 非方形 `(N1,N2)` | 1.5–4.0x | 代价是 3 种 GEMM 形状、3 套 `TCubeTiling` |
| 3 | 开启稀疏跳算 | 1.33x | `R1_in/R1_ker/R1_out`，需 `SetTail` 变形状 |
| 4 | Hermitian 对称 | ~2x（中间三段） | 当前算全 N 点复数谱，未利用实信号共轭对称 |
| 5 | double buffer | 2–3x | 现在 Load/Store 都是 SetFlag 后立即 WaitFlag，全串行 |
| 6 | `GetSubBlockIdx` 拆分 Vector | ~2x（Vector 段） | 现在两个 AIV 做完全相同的工作 |
| 7 | batch 配对打包 | ~2x（B 为偶数） | `x = a + i·b`，已在 `fft_conv1d_four_step_packed` 实现并测过 |
| 8 | 复数 GEMM 改块矩阵 | GEMM 次数减半 | `[Ar|Ai] @ [[Br,Bi],[-Bi,Br]]`，需实虚部横向相邻的布局 |
| 9 | 常量表跨 launch 缓存 | 固定开销 | 现在每次 launch 重算 `N1` 行 Cos/Sin/Fmod |
| 10 | fp16 / bf16 + fp32 累加 | — | 旋转因子仍用 fp32 生成 |
| 11 | 分界点标定 | — | `K < 64` 是估的，要用 msprof 实测 |
| 12 | `R < 核数` 时按 k1 跨核 | — | 阶段 2 §3.2 已设计，默认关闭 |

**推进方式**：先用 msprof 看时间实际花在哪（别照这张表猜第 1、5 项谁占大头），
再逐项做，每项单独提交并保留优化前后的性能与正确性数据。

---

## 6. 验证

不需要 NPU 的逻辑守护（本机可跑）：

```bash
cd FFT_CONV1D && python3 -m pytest tests/ -q
```

上机端到端：

```bash
bash scripts/run_test.sh direct   # K<64
bash scripts/run_test.sh fft      # K>=64
```

判定标准：`assert_close(rtol=5e-3, atol=5e-3)` 必过，且相对最大误差 `< 1e-4`。
未达标时 `compare_data.py` 会打印超标行数与最差位置来定位来源，**不放宽阈值**。

---

## 7. 本轮改动的未验证项

以下是 v2 引入的新逻辑，Python 侧已逐位验证等价性，但**尚未在 910B 上跑过**，
首次上机应优先确认：

1. **`[O1]` 去掉逐行 `SyncAll` 后 FFT 路径是否仍数值正确。**
   若 FFT 挂而 DIRECT 正常，说明 `IterateAll` 的默认同步没覆盖到我们的用法，
   回退方式：在 `Gemm()` 之后、下一个 `ComplexMulInPlace()` 之前补 `SyncAll()`。
2. **`[O2]` `enAtomic=1` 在 910B fp32 + GM 输出下的行为。**
   若结果偏大且随机，检查是不是第一次 `Gemm` 的 `atomic` 参数被误设成了 1。
3. **`[O4]` direct 路径的移位副本。**
   下标逻辑已逐位验证，风险在 UB 容量：`8 × (tile+K) × 4B`，
   `tile=1024, K=63` 约 35KB，若 `InitBuffer` 失败请调小 `FFT_CONV1D_DIRECT_TILE`。
