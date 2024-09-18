#
# resnet-ctx
set(TEST_TARGET resnet-ctx)
add_executable(${TEST_TARGET} resnet-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

#
# resnet-backend
# set(TEST_TARGET resnet-backend)
# add_executable(${TEST_TARGET} resnet-backend.cpp)
# target_link_libraries(${TEST_TARGET} PRIVATE ggml)

if (GGML_CUDA)
    add_compile_definitions(GGML_USE_CUDA)
endif()

if (GGML_METAL)
    add_compile_definitions(GGML_USE_METAL)
endif()
