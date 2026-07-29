"""Static contracts for the current FFT-GM implementation.

These tests deliberately avoid CANN.  They do not prove NPU correctness; they
keep the host dispatch, test runner, and the synchronization design from
silently drifting apart again.
"""

from pathlib import Path
import re
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fft_conv1d_dispatch import (  # noqa: E402
    FFT_MAX_NFFT_GM,
    FFT_MAX_NFFT_UB,
    FFT_MIN_K,
    select_algorithm,
)


def _source(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _shell_function_body(source, name):
    match = re.search(
        rf"^{re.escape(name)}\(\)\s*\{{\s*$"
        rf"(?P<body>.*?)"
        rf"^\}}\s*$",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"scripts/run_test.sh 中缺少 {name}()"
    return match.group("body")


def _literal_run_cases(body):
    return [
        tuple(map(int, values))
        for values in re.findall(
            r"^\s*run_case\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)(?:\s|$)",
            body,
            flags=re.MULTILINE,
        )
    ]


def _function_slice(source, signature):
    start = source.index(signature)
    body_start = source.index("{", start + len(signature))
    depth = 0
    for index in range(body_start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"{signature} 的函数体括号不完整")


def _without_cpp_comments(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//.*", "", source)


def _fft_gm_source():
    source = _source("op_kernel/fft_conv1d.cpp")
    start = source.index("class FftConv1dFftGm")
    end = source.index("// kernel 入口", start)
    return source[start:end]


@pytest.mark.parametrize(
    "L,K,expected_algo,expected_nfft",
    [
        (1024, 63, "DIRECT", 4096),  # K 阈值优先
        (961, 64, "FFT-UB", 1024),   # need = 1024
        (962, 64, "FFT-GM", 4096),   # need = 1025
        (2049, 2048, "FFT-GM", 4096),  # need = 4096
        (2050, 2048, "DIRECT", 16384),  # need = 4097，回退 direct
    ],
)
def test_dispatch_boundaries(L, K, expected_algo, expected_nfft):
    assert (FFT_MIN_K, FFT_MAX_NFFT_UB, FFT_MAX_NFFT_GM) == (64, 1024, 4096)
    assert select_algorithm(L, K) == (expected_algo, expected_nfft)


@pytest.mark.parametrize(
    "function_name,expected_algo",
    [
        ("run_direct", "DIRECT"),
        ("run_fft", "FFT-UB"),
        ("run_fft_gm", "FFT-GM"),
    ],
)
def test_run_test_groups_only_contain_the_named_algorithm(function_name, expected_algo):
    source = _source("scripts/run_test.sh")
    cases = _literal_run_cases(_shell_function_body(source, function_name))
    assert cases, f"{function_name}() 没有测试用例"

    mismatches = []
    for B, H, L, K in cases:
        del B, H  # dispatch 只取决于 L/K
        actual, nfft = select_algorithm(L, K)
        if actual != expected_algo:
            mismatches.append((L, K, nfft, actual))
    assert not mismatches, f"{function_name}() 中存在错误分组: {mismatches}"


def test_run_test_exposes_fftgm_mode():
    source = _source("scripts/run_test.sh")
    assert re.search(r"^\s*fftgm\)\s+run_fft_gm\s*;;", source, flags=re.MULTILINE)
    assert "[direct|fft|fftgm|B H L K]" in source


def test_fft_gm_uses_cann85_mix_1_to_2_message_protocol():
    source = _source("op_kernel/fft_conv1d.cpp")
    code = _without_cpp_comments(source)
    assert "KERNEL_TYPE_MIX_AIC_1_2" in code
    assert "KERNEL_TYPE_MIX_AIC_1_1" not in code
    assert "GM_MM_CFG" not in code
    assert "enableMixDualMaster" not in code

    gm_start = source.index("class FftConv1dFftGm")
    gm_end = source.index("// kernel 入口", gm_start)
    gm_code = _without_cpp_comments(source[gm_start:gm_end])
    assert re.search(r"SyncAll\s*(?:<.*?>)?\s*\(", gm_code) is None, (
        "同步 IterateAll 已完成 KFC 握手，FFT-GM 不应再叠加外层 SyncAll"
    )


def test_mix_1_to_2_uses_both_flattened_aiv_workers():
    source = _without_cpp_comments(_source("op_kernel/fft_conv1d.cpp"))
    worker_idx = _function_slice(source, "int32_t VectorWorkerIdx(")
    assert "GetBlockIdx()" in worker_idx
    assert "GetSubBlockIdx()" not in worker_idx
    assert "GetTaskRation()" not in worker_idx

    direct = source[
        source.index("class FftConv1dDirect"):
        source.index("class FftConv1dFft")
    ]
    direct_init = _function_slice(direct, "void Init(")
    direct_process = _function_slice(direct, "void Process(")
    assert re.search(r"workers_\s*=\s*static_cast<int32_t>\s*"
                     r"\(\s*t\.usedCoreNum\s*\)\s*\*\s*2", direct_init)
    assert "VectorWorkerIdx()" in direct_process
    assert "row += workers_" in direct_process


def test_both_fft_gm_aivs_register_and_execute():
    source = _without_cpp_comments(_source("op_kernel/fft_conv1d.cpp"))
    branch_start = source.index("else if (tilingData.algo == ALGO_FFT_GM)")
    branch_end = source.index("else", branch_start + len("else"))
    branch = source[branch_start:branch_end]
    register = branch.index("REGIST_MATMUL_OBJ")
    init = branch.index("op.Init")
    process = branch.index("op.Process")
    assert register < init < process
    assert "IsPrimaryAiv" not in branch
    assert "return" not in branch


def test_direct_and_fft_ub_do_not_drop_secondary_aiv():
    source = _without_cpp_comments(_source("op_kernel/fft_conv1d.cpp"))
    entry = source[source.index('extern "C" __global__'):]
    assert "IsPrimaryAiv" not in entry

    ub = source[source.index("class FftConv1dFft"):
                source.index("class FftConv1dFftGm")]
    ub_init = _function_slice(ub, "void Init(")
    ub_process = _function_slice(ub, "void Process(")
    assert re.search(r"workers_\s*=\s*static_cast<int32_t>\s*"
                     r"\(\s*t\.usedCoreNum\s*\)\s*\*\s*2", ub_init)
    assert "VectorWorkerIdx()" in ub_process
    assert "h += workers_" in ub_process


def test_fft_gm_offsets_past_the_system_workspace_once():
    source = _source("op_kernel/fft_conv1d.cpp")
    gm_start = source.index("class FftConv1dFftGm")
    gm_end = source.index("// kernel 入口", gm_start)
    gm_code = _without_cpp_comments(source[gm_start:gm_end])

    assert gm_code.count("GetUserWorkspace") == 1, (
        "FFT-GM 只能从原始入口指针计算一次用户 workspace，不能重复偏移"
    )
    assert re.search(
        r"wsGm_\.SetGlobalBuffer\s*\(\s*\(__gm__\s+float\s*\*\)\s*"
        r"GetUserWorkspace\s*\(\s*workspace\s*\)\s*,\s*workspaceElements\s*\)",
        gm_code,
    ), (
        "GM Matmul 的 SetTensorA/B 会读取 GlobalTensor::GetSize；"
        "scratch GlobalTensor 必须显式设置元素数"
    )


def test_fft_gm_gives_each_aiv_worker_independent_scratch():
    source = _without_cpp_comments(_source("op_kernel/fft_conv1d.cpp"))
    gm = _fft_gm_source()
    init = _function_slice(gm, "void Init(")
    process = _function_slice(gm, "void Process(")

    assert re.search(r"workers_\s*=\s*static_cast<int32_t>\s*"
                     r"\(\s*t\.usedCoreNum\s*\)\s*\*\s*2", init)
    assert "workerIdx_ = VectorWorkerIdx()" in init
    assert re.search(
        r"workspaceElements\s*=\s*static_cast<uint64_t>\s*"
        r"\(\s*workers_\s*\)\s*\*\s*FFT_GM_BUFFER_COUNT",
        init,
    )
    assert re.search(
        r"base_\s*=\s*static_cast<uint64_t>\s*\(\s*workerIdx_\s*\)\s*"
        r"\*\s*FFT_GM_BUFFER_COUNT\s*\*\s*N_",
        init,
    )
    assert "h = workerIdx_" in process
    assert "h += workers_" in process
    assert "Half(" not in gm
    assert "subIdx_" not in gm


def test_host_counts_mix_groups_but_allocates_all_aiv_workers():
    host = _without_cpp_comments(_source("op_host/fft_conv1d.cpp"))
    header = _without_cpp_comments(_source("op_host/fft_conv1d_tiling.h"))
    kernel = _without_cpp_comments(_source("op_kernel/fft_conv1d.cpp"))

    assert re.search(
        r"FFT_CONV1D_GM_BUFFER_COUNT\s*=\s*12U?\s*;",
        header,
    ), "FFT-GM buffer count must be defined in the included Host tiling header"
    assert re.search(r"FFT_GM_BUFFER_COUNT\s*=\s*12\s*;", kernel)
    assert "neededGroupNum = CeilDiv(splitUnit, 2)" in host
    assert "vectorWorkerNum = usedCoreNum * 2" in host
    assert re.search(
        r"userWorkspace\s*=\s*static_cast<size_t>\s*"
        r"\(\s*vectorWorkerNum\s*\)\s*\*\s*"
        r"FFT_CONV1D_GM_BUFFER_COUNT",
        host,
    )


@pytest.mark.parametrize(
    "signature",
    [
        "void CopyG(",
        "void BuildTables(",
        "void BinG(",
        "void CMulG(",
        "void LoadRowG(",
        "void StoreOutG(",
    ],
)
def test_fft_gm_vector_helpers_skip_the_aic_participant(signature):
    source = _without_cpp_comments(_fft_gm_source())
    body = _function_slice(source, signature)
    assert re.search(
        r"if\s*(?:\(\s*)?ASCEND_IS_AIC(?:\s*\))?\s*\{\s*return\s*;\s*\}",
        body,
    ), f"{signature} 必须让 AIC 立即返回，只由 AIV 执行 Vector/DMA 工作"


def test_fft_gm_copyg_orders_store_before_next_load():
    source = _source("op_kernel/fft_conv1d.cpp")
    copy_start = source.index("__aicore__ inline void CopyG")
    copy_end = source.index("// ---------------- 常量表", copy_start)
    copy_body = source[copy_start:copy_end]
    assert copy_body.find("LoadG(") < copy_body.find("StoreG(")

    store_start = source.index("__aicore__ inline void StoreG")
    store_end = source.index("__aicore__ inline void CopyG", store_start)
    store_body = source[store_start:store_end]
    copy = store_body.find("DataCopyPad(")
    set_dep = re.search(r"SetFlag\s*<\s*HardEvent::MTE3_MTE2\s*>", store_body)
    wait_dep = re.search(r"WaitFlag\s*<\s*HardEvent::MTE3_MTE2\s*>", store_body)
    assert copy >= 0 and set_dep is not None and wait_dep is not None
    assert copy < set_dep.start() < wait_dep.start(), (
        "StoreG 必须在 MTE3 写出后建立 MTE3_MTE2 依赖，"
        "保证连续 CopyG 复用同一 UB 时下一次 MTE2 不会抢跑"
    )

    load_start = source.index("__aicore__ inline void LoadG")
    load_end = source.index("__aicore__ inline void StoreG", load_start)
    load_body = source[load_start:load_end]
    set_dep = re.search(r"SetFlag\s*<\s*HardEvent::V_MTE2\s*>", load_body)
    wait_dep = re.search(r"WaitFlag\s*<\s*HardEvent::V_MTE2\s*>", load_body)
    copy = load_body.find("DataCopyPad(")
    assert set_dep is not None and wait_dep is not None and copy >= 0
    assert set_dep.start() < wait_dep.start() < copy, (
        "LoadG 必须先建立 V_MTE2 依赖，再覆盖复用中的 UB"
    )
