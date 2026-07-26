# fft_conv1d 阶段 1：语义定义、kernel 方向与裁剪位置推导

> 状态：阶段 1 已完成并全部通过验证（87 个 pytest 用例）。
> 对应代码：`python/fft_conv1d_reference.py`、`tests/test_stage1_semantics.py`

---

## 0. 结论速览（先看这三条）

| # | 结论 |
|---|---|
| 1 | **题目给出的 PyTorch 参考片段与题目给出的数学公式不等价**，两者相差一个 kernel 时间翻转。已用 float64 精确验证（误差 1.1e-16）。按题目规定"以数学公式为准"，本项目以数学公式为权威语义，同时保留 `mode="corr"` 复现原片段。 |
| 2 | **在 math 语义下，FFT 路径不需要翻转 kernel**（频域乘法实现的就是数学卷积）；若要复现原片段的 correlation 语义，则必须在时域把 kernel 翻转后再补零做 RFFT。 |
| 3 | **裁剪区间恒为 `[0 : L]`，无任何偏移**，且与 `N_fft` 的具体取值无关（只要 `N_fft >= L+K-1`）。已用 N=64/128/256/1024 交叉验证。 |

---

## 1. 算子数学语义

输入 `input ∈ R^{B×H×L}`，`kernel ∈ R^{H×K}`，输出 `output ∈ R^{B×H×L}`。

每个 `(b,h)` 独立（depthwise，无通道间求和）：

```
output[b,h,t] = Σ_{j=0}^{K-1} input[b,h,t-j] · kernel[h,j],   t = 0,1,...,L-1
input[b,h,τ] = 0   当 τ < 0
```

记为 **causal depthwise convolution**。注意这是**数学卷积**：`kernel[h,0]` 是"当前抽头"，`kernel[h,K-1]` 是"最旧抽头"。

**冲激响应判据**（区分方向的最强判据）：令 `input[b,h,t] = δ[t]`，则

```
output[b,h,t] = kernel[h,t]  (t < K),  0  (t >= K)
```

即输出**原样**是 kernel，不是翻转的 kernel。这一条被 `test_impulse_response_layout` 强制检查。

---

## 2. 题目参考片段的语义验证（存在歧义，此处澄清）

### 2.1 逐步展开

题目片段：

```python
x = F.pad(input, (K - 1, 0))          # 左侧补 K-1 个 0，长度 L+K-1
weight = kernel.unsqueeze(1)          # [H,1,K]，未翻转
return F.conv1d(x, weight, groups=H)  # 输出长度 (L+K-1)-K+1 = L  ✔
```

`F.pad(input,(K-1,0))` 定义了

```
x[b,h,p] = input[b,h, p-(K-1)],   p = 0..L+K-2   （p < K-1 时为 0）
```

`F.conv1d` 执行的是 **cross-correlation**：

```
out[b,h,t] = Σ_{j=0}^{K-1} x[b,h,t+j] · weight[h,0,j]
           = Σ_{j=0}^{K-1} input[b,h, t+j-K+1] · kernel[h,j]
```

令 `j' = K-1-j`（则 `j = K-1-j'`，`t+j-K+1 = t-j'`）：

```
out[b,h,t] = Σ_{j'=0}^{K-1} input[b,h,t-j'] · kernel[h, K-1-j']        (★)
```

### 2.2 与数学公式对比

| | 与 `input[t]` 相乘的抽头 | 与 `input[t-K+1]` 相乘的抽头 |
|---|---|---|
| 数学公式 | `kernel[h,0]` | `kernel[h,K-1]` |
| 题目片段 (★) | `kernel[h,K-1]` | `kernel[h,0]` |

**两者恰好互为 kernel 的时间翻转**，仅当 `K == 1` 或 kernel 时间对称时才相等。

实测（float64，`K=4`）：

```
kernel      : [-0.7193 -0.4033 -0.5966  0.182 ]
题目片段输出 : [ 0.2805 -0.9728 -0.8431  0.4134  0.5533  1.7303  0.9365  1.2561]
数学公式输出 : [-1.1084 -0.4105  0.7661  0.9255  1.7973  0.7076  1.0245 -0.1284]
corr(翻转kernel后按数学公式算) == 题目片段，最大误差 1.11e-16   ✔ 精确一致
```

冲激响应更直观：

```
input = δ[t] 时
数学公式输出  : [-0.7193 -0.4033 -0.5966  0.182  0 0 0 0]   <- kernel 原样
题目片段输出  : [ 0.182  -0.5966 -0.4033 -0.7193 0 0 0 0]   <- kernel 翻转
```

### 2.3 处理方式

按题目规定"**如发现语义歧义，以数学公式为准**"：

- 默认语义 `mode="math"`：`output[t] = Σ_j input[t-j]·kernel[j]`（本算子的正式定义）。
- 备用语义 `mode="corr"`：`output[t] = Σ_j input[t-j]·kernel[K-1-j]`，与题目原片段**逐比特对齐**。

`mode="corr"` 在实现上就是先做 `kernel.flip(-1)`，再走完全相同的 math 路径 —— 因此**两种语义共享同一份 NPU kernel，只在 host 侧 tiling/搬运阶段决定 kernel 是否反向读取**，不产生任何额外算力开销（后续 AscendC 实现中，翻转可以直接融合进 kernel 从 GM 搬到 UB 时的 stride/反序拷贝，或者融合进 kernel 的 DFT 矩阵）。

> 如果你确认想要的就是题目片段的行为，请在阶段 2 明确告诉我把默认 mode 改成 `corr`；
> 反过来，如果要让原片段与数学公式一致，正确的修法是 `weight = kernel.flip(-1).unsqueeze(1)`。

---

## 3. FFT 实现：为什么 math 语义下不需要翻转 kernel

线性卷积定义：

```
(x * k)[n] = Σ_m x[m]·k[n-m] = Σ_j x[n-j]·k[j]
```

这正是我们的数学公式（在 `n < L` 范围内）。而卷积定理给出

```
DFT_N{ x ⊛ k } = DFT_N{x} · DFT_N{k}      (⊛ 为长度 N 的循环卷积)
```

**频域逐点相乘天然实现的是卷积（不是相关）**，所以：

- `mode="math"`：kernel **不翻转**，直接右侧补零做 RFFT。
- `mode="corr"`：kernel **必须在时域翻转**（`k'[j] = k[K-1-j]`），然后补零。

补充说明为什么不用"共轭代替翻转"：`conj(K̂)` 对应的是 `Σ_j x[t+j]k[j]`（无因果延迟的相关），其有效结果落在循环缓冲的尾部（负索引处），裁剪区间会变成 `y[0]` 与 `y[N-K+1:N]` 的拼接，既不连续也不利于 DataCopy。**时域翻转 + 统一裁剪 `[0:L]`** 在 Ascend 上明显更友好（一次反序搬运 vs. 分段搬运 + 额外的取共轭 Vector 指令），因此选用时域翻转。

---

## 4. FFT 长度选择

```
N_fft >= L + K - 1
首版本： N_fft = 2^⌈log2(L+K-1)⌉ ，且强制 N_fft >= 2
```

**为什么必须 `>= L+K-1`**：长度 N 的循环卷积为

```
y_circ[n] = Σ_{m=0}^{N-1} x_pad[m]·k_pad[(n-m) mod N]
```

`x_pad` 的支撑集是 `[0, L-1]`，`k_pad` 的支撑集是 `[0, K-1]`，二者线性卷积 `y_lin` 的支撑集是 `[0, L+K-2]`，长度 `L+K-1`。混叠项来自把 `y_lin[n+N]`（`n+N >= N`）折叠回 `n`；只要 `L+K-2 < N`，`y_lin` 在 `[N, 2N-2]` 上恒为 0，**无任何混叠**，于是

```
y_circ[n] = y_lin[n],  n = 0..N-1
```

`test_n_fft_too_small_causes_aliasing` 反向验证了这一点：取 `N=16 < L+K-1=20` 时结果确实与直接卷积不符。

**为什么首版本取 2 的幂**：
1. Cooley–Tukey `N = N1·N2` 分解自由度最大（2 的幂可任意拆成 radix-16/32 的组合，如 `1024 = 32×32`、`4096 = 64×64`），便于后续把小规模 DFT 映射成 Cube 的 Matmul。
2. 频点数 `N/2+1` 中的 `N/2` 为 2 的幂，Vector 处理 `1..N/2-1` 的主体区间天然对齐 32B/256B。
3. 旋转因子表可以按层次共享，存储量最小。

**代价**：`L+K-1` 略超过 2 的幂时会浪费近 2 倍算力（如 `L=257,K=17 → 273 → N=512`）。阶段 2 会评估混合基长度（形如 `2^a·3^b·5^c` 的最小可用长度，例如 273 → 288 = 2^5·3^2）能否在 Ascend 上跑得更快 —— 前提是 radix-3/5 的小 DFT 也能高效映射到 Vector/Cube，否则 2 的幂的规整性收益更大。这一取舍留到阶段 2 用实测数据定，不在阶段 1 拍板。

**上限约束**：首版本计划限制 `N_fft <= 8192`（UB 容量约束，阶段 2 精算后写死到 host 校验里）。超出上限走分块卷积（overlap-save）或 direct 路径。

---

## 5. 裁剪区间推导（核心，不凭经验）

设 `y_lin = x ⊛_lin k`，支撑集 `[0, L+K-2]`。目标输出定义为

```
output[t] = Σ_{j=0}^{K-1} input[t-j]·kernel[j],  t = 0..L-1,  input[τ<0] = 0
```

而线性卷积在 `t` 处的值为

```
y_lin[t] = Σ_{j=0}^{min(t, K-1)} x_pad[t-j]·k[j]
```

对 `t ∈ [0, L-1]`：
- `j > t` 的项在 `y_lin` 中根本不存在（下标为负，不参与求和）；在 output 定义中对应 `input[t-j]=0`。**二者等价。**
- `t-j <= t <= L-1`，所以只会取到 `x_pad` 中来自真实 input 的部分，右侧补的零不参与。**无污染。**

因此

```
output[b,h,t] = y_lin[b,h,t],   t = 0..L-1
=> output = y_lin[..., 0:L]
```

**裁剪区间 = `[0 : L]`，偏移量为 0。**

三点补充：

1. **为什么不是 `[K-1 : K-1+L]`**：那是"左侧 pad K-1 后再做 full 卷积"的写法对应的位置。我们的 FFT 路径中 input **没有**左侧补零（左侧补零已经被"因果+零初值"的定义吸收了），只在**右侧**补零到 `N_fft`。右侧补零不改变时间原点，所以偏移为 0。若误用 `[K-1 : K-1+L]`，输出会整体提前 `K-1` 个样本，前几个因果爬升点会丢失。
2. **丢弃的部分**：`y_lin[L : L+K-2]` 是卷积的"拖尾"（tail），对应 `t >= L` 的输出，按题目定义（输出长度固定为 L）直接丢弃。
3. **与 N_fft 无关**：`y_circ[n] = y_lin[n]` 对所有 `n < N` 成立，所以换任何合法 `N_fft` 裁剪位置都不变 —— `test_larger_n_fft_gives_same_result` 用 N=64/128/256/1024 验证通过。

---

## 6. RFFT / IRFFT 的实数域展开（对齐未来 AscendC 实现）

实信号只需保留 `M = N/2+1` 个频点。`fft_conv1d_manual_dft` 完全不用 `torch.fft`，用实数矩阵乘 + 实虚分离复乘 + 显式 IRFFT 公式复算了一遍，与直接卷积对齐（误差见 §7），说明后续 AscendC 按这套公式实现即可。

**前向（RFFT，无归一化）**，`θ = 2πkn/N`：

```
X[k] = Σ_n x[n]·e^{-iθ} = Σ_n x[n](cosθ - i·sinθ)
Xr = x · Cᵀ ,   Xi = -(x · Sᵀ) ,   C[k,n]=cos(2πkn/N), S[k,n]=sin(2πkn/N),  k=0..M-1
```

即一次 `[.., N] × [N, M]` 的**实数矩阵乘**（两次，分别出实部/虚部）—— 这正是后续可以喂给 Cube/Matmul API 的形式。

**逐频点复乘**（`Y = X·K̂`）：

```
Yr = Xr·Kr - Xi·Ki
Yi = Xr·Ki + Xi·Kr
```

- `k=0`（DC）与 `k=N/2`（Nyquist）对实信号虚部恒为 0，乘积虚部也恒为 0 → 代码中显式清零，硬件上可直接省掉这两点的虚部通路。
- `k=1..N/2-1` 为一般复数；`k=N/2+1..N-1` 由共轭对称 `X[N-k]=conj(X[k])` 决定，**不计算、不存储**（省一半算力和一半 UB）。

**逆变换（IRFFT，backward 归一化 1/N，与 `torch.fft.irfft` 一致）**：

```
y[n] = (1/N)·[ Yr[0] + (-1)^n·Yr[N/2] + 2·Σ_{k=1}^{N/2-1} ( Yr[k]·cos(2πkn/N) - Yi[k]·sin(2πkn/N) ) ]
```

推导：`Y[k]e^{iθ} + Y[N-k]e^{-iθ} = 2·Re(Y[k]e^{iθ}) = 2(Yr[k]cosθ - Yi[k]sinθ)`；`k=0` 与 `k=N/2` 自共轭故权重为 1，`e^{iπn} = (-1)^n`。

实现上等价于给频点乘一个权重向量 `w = [1, 2, 2, ..., 2, 1]`（长度 M）后再做两次实数矩阵乘：`y = (Yr·w)·Ci - (Yi·w)·Si`，同样是 Cube 友好的形式。

**归一化系数只出现一次**（`1/N`，放在 IRFFT 末尾）。也可以预先折叠进 IRFFT 的余弦/正弦矩阵或者 `w` 向量里，省掉一遍 Vector 的 muls —— 但要注意 fp16/bf16 下先缩放会影响动态范围，低精度版本再定。

**只算前 L 个时间点的优化**：IRFFT 的输出我们只要 `n = 0..L-1`，所以逆变换矩阵可以只取 `[M, L]` 而不是 `[M, N]`，直接省掉约 `(N-L)/N` 的 IRFFT 算力（`L=257,N=512` 时省 50%）。这一点在朴素蝶形 FFT 里做不到，但在 **DFT-Matmul 形式下是白送的**，是把 IRFFT 映射到 Cube 的一个额外理由，阶段 2 展开。

---

## 7. 数值验证结果（实测，float32）

环境：torch 2.2.2 / CPU。基准 = float64 直接卷积。

### 7.1 要求覆盖的 shape

| B,H,L,K | N_fft | FFT vs direct (max abs) | 手写DFT vs direct | 相对误差 |
|---|---|---|---|---|
| 1,1,16,3 | 32 | 1.19e-07 | 1.19e-07 | 8.4e-08 |
| 2,3,31,7 | 64 | 9.54e-07 | 9.54e-07 | 1.2e-07 |
| 4,8,64,1 | 64 | 9.54e-07 | 1.19e-06 | 1.4e-07 |
| 1,5,100,32 | 256 | 2.38e-06 | 4.77e-06 | 1.4e-07 |
| 2,16,257,17 | 512 | 2.86e-06 | 5.25e-06 | 1.6e-07 |

### 7.2 边界 shape

| B,H,L,K | N_fft | FFT vs direct | 说明 |
|---|---|---|---|
| 1,1,1,1 | 2 | 0.0 | 最小退化情形 |
| 1,1,8,8 | 16 | 2.38e-07 | K == L |
| 3,2,30,3 | 32 | 9.54e-07 | L+K-1 恰为 2 的幂（不进位） |
| 2,4,128,64 | 256 | 4.77e-06 | 大 K |
| 1,3,513,5 | 1024 | 9.54e-07 | L 非 2 的幂 |
| 5,7,33,33 | 128 | 1.91e-06 | K == L 且都非 2 的幂 |

### 7.3 误差随 N 的增长（fp32 FFT）

| L | K | N_fft | max abs err | 相对误差 |
|---|---|---|---|---|
| 64 | 8 | 128 | 9.5e-07 | 1.2e-07 |
| 256 | 16 | 512 | 2.4e-06 | 1.6e-07 |
| 1024 | 64 | 2048 | 5.7e-06 | 1.8e-07 |
| 4096 | 128 | 8192 | 8.6e-06 | 1.8e-07 |
| 16384 | 256 | 32768 | 1.9e-05 | 2.5e-07 |

**相对误差稳定在 1e-7 ~ 2.5e-7 量级**（约 `O(√log N)·ε_fp32` 的理论行为），距离阶段 4 的目标 `1e-4` 有约 400 倍余量。绝对误差随信号能量线性增长，因此阶段 4 的判定应以**相对误差**为准（测试里已按 `max(1, |ref|_max)` 归一）。

> 注：手写 DFT 矩阵版误差略大于 `torch.fft`（约 2 倍），因为它是 O(N²) 的一次性长累加，而 FFT 是 O(N log N) 的分层累加、累加链更短。这提示 **AscendC 实现应优先用 Cooley–Tukey 分解（分层短累加），而不是一次大 DFT 矩阵乘** —— 除了算力，精度上也更优。

### 7.4 复数乘法 3-mul vs 4-mul

`test_complex_mul_variants_agree` 验证两者在 fp32 下 1e-5 内一致，走完整算子路径也都满足 5e-3。初步判断（详细分析留到阶段 2）：**Ascend 910B 上应选 4-mul**。理由是 Vector 单元的乘加吞吐远不是瓶颈（瓶颈在 GM↔UB 搬运和 UB 容量），而 3-mul 需要 5 次加减 + 2 个额外临时 buffer，UB 压力和指令数反而更差；且 `t3-t1-t2` 在 fp16/bf16 下有明显的抵消误差风险。

---

## 8. 测试覆盖清单

`tests/test_stage1_semantics.py`，**87 passed**：

| 测试 | 目的 |
|---|---|
| `test_original_snippet_is_correlation` | float64 精确证明题目片段 == corr 语义 |
| `test_original_snippet_differs_from_math_formula_when_K_gt_1` | 证明歧义真实存在（K=1 时二者一致，K>1 时不一致） |
| `test_direct_torch_matches_naive` | direct 两种写法自洽（两种 mode 各测一遍） |
| `test_impulse_response_layout` | **kernel 方向判据**：冲激响应必须是未翻转的 kernel |
| `test_causality_no_future_leak` | **因果性**：改动 input 后半段不影响输出前半段 |
| `test_shift_property` | **裁剪位置判据**：input 右移 s，输出必须同样右移 s 且前 s 点为 0 |
| `test_fft_matches_direct_fp32` | FFT vs direct，两种 mode × 11 shape |
| `test_manual_dft_matches_direct_fp32` | 手写 DFT/复乘/IRFFT 路径 vs direct |
| `test_complex_mul_variants_agree` | 3-mul / 4-mul 一致性 |
| `test_next_pow2_and_choose_n_fft` | FFT 长度选择逻辑与边界（含 L+K-1 恰为 2 的幂不进位） |
| `test_n_fft_too_small_causes_aliasing` | 反向证明 N >= L+K-1 是必要条件 |
| `test_larger_n_fft_gives_same_result` | 裁剪位置与 N_fft 无关 |
| `test_random_shapes_sweep` | 60 组随机 (B,H,L,K) 对拍，最大相对误差 < 1e-4 |
| `test_shape_validation` | shape 约束校验（H 不匹配 / K>L / 维度错误） |

---

## 9. 已固化的约束（将写入算子校验与文档，不做隐式假设）

首版本硬约束：

| 约束 | 值 | 校验位置 |
|---|---|---|
| dtype | float32 | host shape/dtype 校验 |
| 维度 | input `[B,H,L]`，kernel `[H,K]`，H 必须一致 | `check_shapes` |
| 取值范围 | `B,H,L,K >= 1`，`K <= L` | `check_shapes` |
| N_fft | `2^⌈log2(L+K-1)⌉`，且 `>= 2` | `choose_n_fft` |
| N_fft 上限 | 计划 8192（阶段 2 按 UB 容量精算后写死） | 待定 |
| 内存 | 连续（contiguous），地址按硬件要求对齐 | 待阶段 3 |
| 语义 mode | 默认 `math`；`corr` 用于复现题目原片段 | 算子属性 |

允许但需注意：`L`、`K`、`L+K-1` 均**不要求**是 2 的幂（已测 L=31/100/257/513、K=3/7/17/32/33）；`B*H` 不要求能被核数整除（阶段 2 处理尾块）。

---

## 10. 阶段 1 完成，待确认后进入阶段 2

**需要你拍板的一个问题**：默认语义取 `math`（题目数学公式，kernel 不翻转）还是 `corr`（题目 PyTorch 片段，kernel 翻转）？我目前按题目"以数学公式为准"的规定默认 `math`，两种都已实现且都通过验证，改默认值只是改一个常量。

确认后阶段 2 将输出：FFT 分解方案（Cooley–Tukey `N=N1·N2` 的 radix 选择 + Vector/Cube 分工权衡）、数据布局、多核切分（`B*H` 维优先、尾块与 `B*H < 核数` 的处理）、UB/L1/L0 数据流与 tile 大小、旋转因子生成与存储、复杂度与 GM 搬运量分析、direct conv 与 FFT conv 的性能分界点估算。
