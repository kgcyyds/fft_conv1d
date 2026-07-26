/**
 * fft_conv1d 单算子调用示例（aclnn 接口）
 *
 * 用法：
 *   ./fft_conv1d_test <data_dir> <B> <H> <L> <K> <flip_kernel>
 * 读取 <data_dir>/x.bin、<data_dir>/kernel.bin，输出写 <data_dir>/npu_out.bin
 *
 * aclnn 接口名由算子定义自动生成：算子 FftConv1d -> aclnnFftConv1d
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_fft_conv1d.h"

#define CHECK(expr)                                                                      \
    do                                                                                   \
    {                                                                                    \
        auto _ret = (expr);                                                              \
        if (_ret != 0)                                                                   \
        {                                                                                \
            std::printf("[ERROR] %s:%d  %s failed, ret=%d\n", __FILE__, __LINE__, #expr, \
                        static_cast<int>(_ret));                                         \
            return -1;                                                                   \
        }                                                                                \
    } while (0)

static bool ReadBin(const std::string &path, std::vector<float> &buf, size_t count)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
    {
        std::printf("[ERROR] 打不开 %s\n", path.c_str());
        return false;
    }
    buf.resize(count);
    f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<size_t>(f.gcount()) != count * sizeof(float))
    {
        std::printf("[ERROR] %s 大小不符，期望 %zu 个 float\n", path.c_str(), count);
        return false;
    }
    return true;
}

static bool WriteBin(const std::string &path, const std::vector<float> &buf)
{
    std::ofstream f(path, std::ios::binary);
    if (!f)
    {
        return false;
    }
    f.write(reinterpret_cast<const char *>(buf.data()),
            static_cast<std::streamsize>(buf.size() * sizeof(float)));
    return true;
}

// 按 ND 连续布局创建 aclTensor
static aclTensor *MakeTensor(const std::vector<int64_t> &shape, void *devPtr)
{
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, strides.data(), 0,
                           aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), devPtr);
}

int main(int argc, char **argv)
{
    if (argc < 7)
    {
        std::printf("用法: %s <data_dir> <B> <H> <L> <K> <flip_kernel>\n", argv[0]);
        return -1;
    }
    const std::string dir = argv[1];
    const int64_t B = std::atoll(argv[2]);
    const int64_t H = std::atoll(argv[3]);
    const int64_t L = std::atoll(argv[4]);
    const int64_t K = std::atoll(argv[5]);
    const int64_t flipKernel = std::atoll(argv[6]);

    const size_t xCount = static_cast<size_t>(B * H * L);
    const size_t kCount = static_cast<size_t>(H * K);
    const size_t yCount = xCount;

    std::vector<float> xHost, kHost, yHost(yCount, 0.0f);
    if (!ReadBin(dir + "/x.bin", xHost, xCount) || !ReadBin(dir + "/kernel.bin", kHost, kCount))
    {
        return -1;
    }

    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream));

    void *xDev = nullptr;
    void *kDev = nullptr;
    void *yDev = nullptr;
    CHECK(aclrtMalloc(&xDev, xCount * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&kDev, kCount * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&yDev, yCount * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(xDev, xCount * sizeof(float), xHost.data(), xCount * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemcpy(kDev, kCount * sizeof(float), kHost.data(), kCount * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE));

    aclTensor *xT = MakeTensor({B, H, L}, xDev);
    aclTensor *kT = MakeTensor({H, K}, kDev);
    aclTensor *yT = MakeTensor({B, H, L}, yDev);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK(aclnnFftConv1dGetWorkspaceSize(xT, kT, flipKernel, yT, &workspaceSize, &executor));

    void *workspace = nullptr;
    if (workspaceSize > 0)
    {
        CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    std::printf("[run] B=%lld H=%lld L=%lld K=%lld flip=%lld workspace=%llu bytes\n",
                static_cast<long long>(B), static_cast<long long>(H), static_cast<long long>(L),
                static_cast<long long>(K), static_cast<long long>(flipKernel),
                static_cast<unsigned long long>(workspaceSize));

    CHECK(aclnnFftConv1d(workspace, workspaceSize, executor, stream));
    CHECK(aclrtSynchronizeStream(stream));

    CHECK(aclrtMemcpy(yHost.data(), yCount * sizeof(float), yDev, yCount * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST));
    if (!WriteBin(dir + "/npu_out.bin", yHost))
    {
        std::printf("[ERROR] 写 npu_out.bin 失败\n");
        return -1;
    }
    std::printf("[run] 输出已写入 %s/npu_out.bin\n", dir.c_str());

    aclDestroyTensor(xT);
    aclDestroyTensor(kT);
    aclDestroyTensor(yT);
    if (workspace != nullptr)
    {
        aclrtFree(workspace);
    }
    aclrtFree(xDev);
    aclrtFree(kDev);
    aclrtFree(yDev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
