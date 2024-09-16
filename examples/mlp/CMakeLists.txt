set(TEST_TARGET mlp)
add_executable(${TEST_TARGET} main_ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

# For GPU offloading
if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
