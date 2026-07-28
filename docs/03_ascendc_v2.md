# fft_conv1d 当前 AscendC 实现

> 对应源码：`op_host/fft_conv1d.cpp`、`op_host/fft_conv1d_tiling.h`、
> `op_kernel/fft_conv1d.cpp`
>
> 目标环境：Ascend 910B / Atlas A2，CANN 8.5

## 1. 接口与语义

```text
输入  x      : [B, H, L]  float32  ND 连续
      kernel : [H, K]     float32  ND 连续
输出  y      : [B, H, L]  float32  ND 连续
属性  无
```

```text
y[b,h,t] = sum(j=0..K-1) x[b,h,t-j] * kernel[h,j]
x[b,h,t-j] = 0, 当 t-j < 0
```

这是 causal depthwise 数学卷积，不是 `torch.nn.functional.conv1d` 默认执行的
cross-correlation。设备接口只实现上述 math 语义。

Host 校验：

- `x` 为 3 维、`kernel` 为 2 维，且两个输入的 `H` 相同；
- `B,H,L,K >= 1`；
- `K <= L`；
- 当前注册的数据类型为 `float32`。

FFT 容量不是算子 shape 的硬上限；FFT 不适用时会回退 DIRECT。

## 2. 三路分派

定义：

```text
need = max(2, L + K - 1)
N    = 不小于 need 的最小 4 的幂
N1   = N2 = sqrt(N)
```

Host 静态分派：

| 条件 | 路径 | 当前执行方式 |
|---|---|---|
| `K < 64` | DIRECT | 主 AIV 上直接因果卷积 |
| `K >= 64 && need <= 1024` | FFT-UB | 数据常驻 UB；当前 `FFT_CONV1D_USE_CUBE=0`，矩阵乘为 Vector Axpy 基线 |
| `K >= 64 && 1024 < need <= 4096` | FFT-GM | `N=4096`；Cube 执行 GM Matmul，Vector 执行旋转因子与逐点运算 |
| `need > 4096` | DIRECT 回退 | 保持数值覆盖，性能较慢 |

`FFT_CONV1D_ENABLE_GM` 与 `FFT_CONV1D_ENABLE_GM_KERNEL` 必须同时启用或同时关闭。

## 3. Host tiling 与核切分

kernel 固定声明为 `KERNEL_TYPE_MIX_AIC_1_2`。一个逻辑组包含 1 个 AIC 和
2 个 AIV，所以 Host 可用逻辑组数为：

```text
min(GetCoreNumAic(), GetCoreNumAiv() / 2)
```

- DIRECT 按 `B*H` 行切；
- FFT-UB、FFT-GM 按通道 `H` 切，每个逻辑核处理所分通道的全部 batch；
- `blockDim` 等于实际使用的逻辑组数。

FFT-GM 的 Matmul tiling 与 kernel 类型一致：

```text
A: GM / ND / float
B: GM / ND / float
C: GM / ND / float
shape: (N1, N1, N1)
```

FFT-GM 自有临时 UB 为：

```text
6 * N * sizeof(float)
+ 2 * AlignUp8(N1) * sizeof(float)
+ 8192
```

`N=4096, N1=64` 时共 `107008` 字节。Host 在 Matmul tiling 中只提供扣除
这部分后的 UB 预算，避免 Matmul 与用户 `TPipe` 同时按整块 UB 规划。

## 4. MIX 1:2 与 KFC 生命周期

在 1:2 模式下，AIV 的 `GetBlockIdx()` 是展开后的索引。AIC、两个 AIV 使用同一
逻辑核号：

```cpp
CoreIdx = (GetBlockIdx() - GetSubBlockIdx()) / GetTaskRation();
```

每组只允许 `subBlockIdx == 0` 的主 AIV 执行用户 Vector/DMA 数据流。

- DIRECT、非 Cube 的 FFT-UB：AIC 和第二个 AIV 直接退出；
- FFT-GM：AIC 和两个 AIV 都必须先执行 `REGIST_MATMUL_OBJ`；
- 第二个 AIV 注册后立即退出，其栈上 KFC client 析构并发送 `SERVICE_QUIT`；
- 主 AIV 发起全部 Matmul，完成后析构并发送另一个 quit；
- AIC 在 `REGIST_MATMUL_OBJ` 展开的 server 循环中服务消息，收到两个 quit 后返回。

第二个 AIV 不能在 `REGIST_MATMUL_OBJ` 之前退出，否则 CANN 8.5 的 KFC server
会一直等待缺失的 client 生命周期。

同步版 `IterateAll(GlobalTensor)` 完成 AIV→AIC→AIV 的消息握手和结果等待。
FFT-GM 不再叠加外层 `SyncAll` 或手工 CrossCore flag，以免与 Matmul 内部 flag
冲突，也不要求不同逻辑核执行相同次数的 Matmul。

## 5. FFT-GM workspace 与数据流

Host 总 workspace：

```text
[ Matmul/KFC 系统区 | 用户区 ]
```

系统区大小由 `GetLibApiWorkSpaceSize()` 给出。FFT-GM 用户区大小为：

```text
usedCoreNum * 12 * N * sizeof(float)
```

kernel 入口的 `workspace` 指向整块空间首址。Matmul 使用
`GetSysWorkSpacePtr()`，FFT-GM 必须且只调用一次
`GetUserWorkspace(workspace)` 定位用户区；直接从 offset 0 写 scratch 会覆盖
KFC 消息队列。

每个逻辑核独占 12 个长度 `N` 的 float 槽：

```text
Dr, Di, Tr, Ti, Kr, Ki, Xr, Xi, Yr, Yi, Zr, Zi
```

处理顺序：

1. 主 AIV 在 UB 生成本核的 `D` 与 twiddle 表并写入本核 GM 槽；
2. 按 `h = CoreIdx, CoreIdx + usedCoreNum, ...` 计算并缓存该通道的 kernel 频谱；
3. 对该通道的每个 batch 执行输入 FFT、复数逐点乘、IFFT；
4. IFFT 在最后一步乘 `1/N`；
5. causal 输出直接裁剪线性卷积的 `[0:L]`。

Cube 的 A/B/C 都在 GM；Vector 运算前把所需平面搬到 UB，完成后写回 GM。
`LoadG` 在覆盖复用 UB 前建立 `V_MTE2` 依赖，`StoreG` 建立
`V_MTE3` 与 `MTE3_MTE2` 依赖。写给下一次 Matmul 的 GM 数据与 KFC 消息共用
同一 MTE3 pipe，后发的 KFC 消息保证 AIC 不会早于前序 GM 写入读取输入。

## 6. 其他路径的 workspace

- FFT-UB：无用户 workspace；
- DIRECT：Host 当前申请
  `2 * usedCoreNum * AlignUp8(K-1+L) * sizeof(float)`，kernel 每个主 AIV使用一行
  零前缀缓冲。

## 7. 验证

本地静态与 CPU 回归：

```bash
python3 -B -m pytest -q
bash -n scripts/run_test.sh
python3 -B -m py_compile scripts/gen_data.py scripts/fft_conv1d_dispatch.py
git diff --check
```

上机按路径运行：

```bash
bash scripts/run_test.sh direct
bash scripts/run_test.sh fft
bash scripts/run_test.sh fftgm
```

CPU 数学测试和源码合同只能验证公式、分派及静态约束，不能证明以下目标行为：

- CANN 8.5 Host/kernel clean build；
- MIX 1:2 的实际 KFC 生命周期；
- GM Matmul 与 MTE 可见性；
- workspace ABI；
- 910B 端到端精度、稳定性与性能。

FFT-GM 合入前仍需在目标 CANN 8.5 环境 clean build，并在真实 910B 上至少覆盖
`need=1025`、非 8 对齐 shape、`need=4096` 边界、多通道和多 batch。
