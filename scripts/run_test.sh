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
#   bash scripts/run_test.sh direct       # 只跑 DIRECT 路径（小 K 或大 N 回退）
#   bash scripts/run_test.sh fft          # 只跑 FFT-UB 路径（N<=1024）
#   bash scripts/run_test.sh fftgm        # 只跑 FFT-GM 路径（N=4096）
#   bash scripts/run_test.sh 2 16 257 17  # 跑单个用例 B H L K
set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BIN="${ROOT}/examples/build/fft_conv1d_test"
DATA="${ROOT}/data"

if [ ! -x "${BIN}" ]; then
    echo "找不到可执行文件 ${BIN}"
    echo "请先: cd ${ROOT}/examples && mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

FAILED=0

run_case() {
    local B=$1 H=$2 L=$3 K=$4
    echo "=================================================================="
    echo "用例: B=${B} H=${H} L=${L} K=${K}"
    echo "=================================================================="
    rm -rf "${DATA}"
    python3 "${ROOT}/scripts/gen_data.py" --b "${B}" --h "${H}" --l "${L}" --k "${K}" --out "${DATA}"
    "${BIN}" "${DATA}" "${B}" "${H}" "${L}" "${K}"
    if python3 "${ROOT}/scripts/compare_data.py" --dir "${DATA}"; then :; else FAILED=1; fi
    echo ""
}

# DIRECT 路径：K < 64，或 K >= 64 但 need=L+K-1 > 4096
run_direct() {
    echo "###### DIRECT 路径（小 K 或大 N 回退）######"
    run_case 1 1 16 3
    run_case 2 3 31 7
    run_case 4 8 64 1
    run_case 1 5 100 32
    run_case 2 16 257 17
    # 移位副本方案的重点用例：K 跨越 8 的边界、K 接近上限、L 跨多个 tile
    run_case 1 1 8 8
    run_case 2 2 100 63
    run_case 1 1 2000 40
    run_case 3 2 30 3
    run_case 1 3 513 5
    run_case 1 1 1 1
    run_case 2 2 1024 63
    # FFT-GM 当前只支持 need<=4096；该 shape 必须回退 DIRECT（N_fft=16384）
    run_case 1 2 4096 1024
}

# K >= 64 且 need <= 1024 -> FFT-UB 路径
run_fft() {
    echo "###### FFT-UB 路径（K>=64 且 need<=1024）######"
    run_case 1 1 64 64
    run_case 2 4 128 64
    run_case 2 2 256 128
    run_case 1 8 512 256
    run_case 1 1 100 100
    run_case 3 2 300 200
    run_case 1 5 255 255
    # R = B*H 小于核数，检验空闲核不参与时的行为
    run_case 1 1 512 512
    # R 不能被核数整除
    run_case 3 7 256 100
}

# 1024 < need <= 4096 -> FFT-GM 路径；四次幂规划下 N_fft 恒为 4096
run_fft_gm() {
    echo "###### FFT-GM 路径（1024 < need<=4096，N_fft=4096）######"
    run_case 1 1 600 600      # N=4096
    run_case 2 2 1024 1024    # N=4096
    run_case 1 4 2048 512     # N=4096
    run_case 2 2 2048 2048    # N=4096（need=4095，上边界附近）
    run_case 3 3 1500 300     # N=4096，L/K 均非 8 对齐
}

case "$1" in
    direct) run_direct ;;
    fft)    run_fft ;;
    fftgm)  run_fft_gm ;;
    "")     run_direct; run_fft; run_fft_gm ;;
    *)
        if [ $# -eq 4 ]; then
            run_case "$1" "$2" "$3" "$4"
        else
            echo "用法: bash scripts/run_test.sh [direct|fft|fftgm|B H L K]"
            exit 1
        fi
        ;;
esac

if [ ${FAILED} -eq 0 ]; then
    echo "全部用例 PASS"
else
    echo "存在 FAIL 用例，见上文输出"
    exit 1
fi
