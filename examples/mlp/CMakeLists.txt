#
# three-layer-fnn

set(TEST_TARGET three-layer-fnn)
add_executable(${TEST_TARGET} main.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common)