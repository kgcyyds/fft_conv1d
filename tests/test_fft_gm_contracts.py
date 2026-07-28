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
    end = source.index("#endif // FFT_CONV1D_ENABLE_GM_KERNEL", start)
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
    gm_end = source.index("#endif // FFT_CONV1D_ENABLE_GM_KERNEL", gm_start)
    gm_code = _without_cpp_comments(source[gm_start:gm_end])
    assert re.search(r"SyncAll\s*(?:<.*?>)?\s*\(", gm_code) is None, (
        "同步 IterateAll 已完成 KFC 握手，FFT-GM 不应再叠加外层 SyncAll"
    )


def test_mix_1_to_2_maps_one_primary_aiv_to_each_logical_core():
    source = _without_cpp_comments(_source("op_kernel/fft_conv1d.cpp"))
    core_idx = _function_slice(source, "int32_t CoreIdx(")
    assert "GetBlockIdx()" in core_idx
    assert "GetSubBlockIdx()" in core_idx
    assert "GetTaskRation()" in core_idx

    primary = _function_slice(source, "bool IsPrimaryAiv(")
    assert re.search(r"GetSubBlockIdx\s*\(\s*\)\s*==\s*0", primary)

    direct = source[
        source.index("class FftConv1dDirect"):
        source.index("class FftConv1dFft")
    ]
    direct_process = _function_slice(direct, "void Process(")
    assert "CoreIdx()" in direct_process
    assert "GetBlockIdx()" not in direct_process


def test_secondary_aiv_registers_before_quitting_fft_gm():
    source = _without_cpp_comments(_source("op_kernel/fft_conv1d.cpp"))
    branch_start = source.index("else if (tilingData.algo == ALGO_FFT_GM)")
    branch_end = source.index("#endif", branch_start)
    branch = source[branch_start:branch_end]
    register = branch.index("REGIST_MATMUL_OBJ")
    secondary_guard = branch.index("!IsPrimaryAiv()")
    assert register < secondary_guard, (
        "MIX 1:2 的第二个 AIV 必须先注册 KFC client，再退出并发送 quit"
    )


def test_fft_gm_offsets_past_the_system_workspace_once():
    source = _source("op_kernel/fft_conv1d.cpp")
    gm_start = source.index("class FftConv1dFftGm")
    gm_end = source.index("#endif // FFT_CONV1D_ENABLE_GM_KERNEL", gm_start)
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
