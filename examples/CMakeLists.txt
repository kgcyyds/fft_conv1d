cmake_minimum_required(VERSION 3.16)
project(fft_conv1d_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN 安装路径与自定义算子包路径
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/ascend-toolkit/latest)
    endif()
endif()

# 自定义算子包安装后的头文件/库位置（vendor_name 默认 customize）
if(NOT DEFINED CUSTOM_OPP_PATH)
    set(CUSTOM_OPP_PATH ${ASCEND_CANN_PACKAGE_PATH}/opp/vendors/customize)
endif()

add_executable(fft_conv1d_test main.cpp)

target_include_directories(fft_conv1d_test PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CUSTOM_OPP_PATH}/op_api/include
)

target_link_directories(fft_conv1d_test PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${CUSTOM_OPP_PATH}/op_api/lib
)

target_link_libraries(fft_conv1d_test PRIVATE
    ascendcl
    nnopbase
    opapi
    cust_opapi
)
