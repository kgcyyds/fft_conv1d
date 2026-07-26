#!/bin/bash
# fft_conv1d 端到端验证：生成数据 -> NPU 执行 -> 与 golden 比对
#
# 前置条件：
#   1. 已 source CANN 环境（set_env.sh）
#   2. 已执行 ./build.sh 并安装 build_out/custom_opp_*.run
#   3. 已在 examples/ 下 cmake && make 得到 fft_conv1d_test
#
# 用法：
#   bash scripts/run_test.sh              # 跑全部用例
#   bash scripts/run_test.sh 2 16 257 17 0  # 跑单个用例 B H L K flip
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN="${ROOT}/examples/build/fft_conv1d_test"
DATA="${ROOT}/data"

if [ ! -x "${BIN}" ]; then
    echo "找不到可执行文件 ${BIN}"
    echo "请先: cd ${ROOT}/examples && mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

run_case() {
    local B=$1 H=$2 L=$3 K=$4 FLIP=$5
    echo "=================================================================="
    echo "用例: B=${B} H=${H} L=${L} K=${K} flip_kernel=${FLIP}"
    echo "=================================================================="
    rm -rf "${DATA}"
    python3 "${ROOT}/scripts/gen_data.py" --b "${B}" --h "${H}" --l "${L}" --k "${K}" \
        --flip "${FLIP}" --out "${DATA}"
    "${BIN}" "${DATA}" "${B}" "${H}" "${L}" "${K}" "${FLIP}"
    python3 "${ROOT}/scripts/compare_data.py" --dir "${DATA}"
    echo ""
}

if [ $# -eq 5 ]; then
    run_case "$1" "$2" "$3" "$4" "$5"
    exit 0
fi

FAILED=0

echo "###### 第一组：题目要求覆盖的 shape（K<64，全部走 DIRECT 路径）######"
run_case 1 1 16 3 0   || FAILED=1
run_case 2 3 31 7 0   || FAILED=1
run_case 4 8 64 1 0   || FAILED=1
run_case 1 5 100 32 0 || FAILED=1
run_case 2 16 257 17 0 || FAILED=1

echo "###### 第二组：K>=64，走 FFT 路径（本算子的目标区间）######"
run_case 1 1 64 64 0   || FAILED=1
run_case 2 4 128 64 0  || FAILED=1
run_case 2 2 256 128 0 || FAILED=1
run_case 1 8 512 256 0 || FAILED=1
run_case 2 2 1024 1024 0 || FAILED=1

echo "###### 第三组：corr 语义（复现题目原始 PyTorch 片段）######"
run_case 2 3 31 7 1    || FAILED=1
run_case 2 2 256 128 1 || FAILED=1

echo "###### 第四组：边界 ######"
run_case 1 1 1 1 0     || FAILED=1
run_case 1 1 8 8 0     || FAILED=1
run_case 3 2 30 3 0    || FAILED=1
run_case 1 3 513 5 0   || FAILED=1
run_case 5 3 100 20 0  || FAILED=1

if [ ${FAILED} -eq 0 ]; then
    echo "全部用例 PASS"
else
    echo "存在 FAIL 用例，见上文输出"
    exit 1
fi
