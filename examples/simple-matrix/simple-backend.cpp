/*
1. Initialize ggml_backend
2. Allocate ggml_context to store tensor metadata (we don't need to allocate tensor data right away)
3. Create tensors metadata (only their shapes and data types)
4. Allocate a ggml_backend_buffer to store all tensors
5. Copy tensor data from main memory (RAM) to backend buffer
5. Create a ggml_cgraph for mul_mat operation
7. Create a ggml_gallocr for cgraph allocation
8. Optionally: schedule the cgraph using ggml_backend_sched
9. Run the computation
10. Retrieve results (output tensors)
11. Free memory and exit
*/

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"            // Conditionally include CUDA backend if GGML_USE_CUDA is defined
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"           // Conditionally include Metal backend if GGML_USE_METAL is defined
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(void) {
    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;
    float matrix_A[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };
    const int rows_B = 3, cols_B = 2;
    float matrix_B[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
    };

    // 1. Initialize ggml_backend
    ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0); // Initialize CUDA backend for device 0
    if (!backend)
    {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

#ifdef GGML_USE_METAL
    fprintf(stderr, "%s: using Metal backend\n", __func__);
    backend = ggml_backend_metal_init(); // Initialize Metal backend
    if (!backend)
    {
        fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
    }
#endif
    // if there aren't GPU Backends fallback to CPU backend
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }

    // Calculate the size needed to allocate
    size_t ctx_size = 0;
    ctx_size += 2 * ggml_tensor_overhead(); // tensors
    // no need to allocate anything else!

    // 2. Allocate ggml_context to store tensor metadata (we don't need to allocate tensor data right away)
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    struct ggml_context * ctx = ggml_init(params);

    // 3. Create tensors metadata (only their shapes and data types)
    struct ggml_tensor * tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_A, rows_A);
    struct ggml_tensor * tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols_B, rows_B);

    // 4. Allocate a `ggml_backend_buffer` to store all tensors
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // 5. Copy tensor data from main memory (RAM) to backend buffer
    ggml_backend_tensor_set(tensor_a, matrix_A, 0, ggml_nbytes(tensor_a));
    ggml_backend_tensor_set(tensor_b, matrix_B, 0, ggml_nbytes(tensor_b));

    // 6. Create `ggml_cgraph` using forward computation
    struct ggml_cgraph * gf = NULL;
    struct ggml_context * ctx_cgraph = NULL;
    {
        // create a temporally context to build the graph
        struct ggml_init_params params0 = {
            /*.mem_size   =*/ ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph()
        };
        ctx_cgraph = ggml_init(params0);
        gf = ggml_new_graph(ctx_cgraph);

        // result = a*b^T
        // Pay attention: ggml_mul_mat(A, B) ==> B will be transposed internally
        // the result is transposed
        struct ggml_tensor * result0 = ggml_mul_mat(ctx_cgraph, tensor_a, tensor_b);

        // Add "result" tensor and all of its dependencies to the cgraph
        ggml_build_forward_expand(gf, result0);
    }
    // Print the cgraph
    ggml_graph_print(gf);
    ggml_graph_dump_dot(gf, NULL, "simple-backend.dot");

    // 7. Create a `ggml_gallocr` for cgraph computation
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // 8. Run the computation
    int n_threads = 4; // Optional: number of threads to perform some operations with multi-threading
    if (ggml_backend_is_cpu(backend)) {
        ggml_backend_cpu_set_n_threads(backend, n_threads);
    }
#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(backend))
    {
        ggml_backend_metal_set_n_cb(backend, n_threads); // Set number of command buffers for Metal backend
    }
#endif
    ggml_backend_graph_compute(backend, gf); // Perform the computation

    // 9. Retrieve results (output tensors)
    // in this example, output tensor is always the last tensor in the graph
    struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    float * result_data = (float *) malloc(ggml_nbytes(result)); // In C++, malloc returns a void*, and assigning it to a typed pointer requires an explicit cast.
    // because the tensor data is stored in device buffer, we need to copy it back to RAM
    ggml_backend_tensor_get(result, result_data, 0, ggml_nbytes(result));
    printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", result_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");
    free(result_data);

    // 10. Free memory and exit
    ggml_free(ctx_cgraph);
    ggml_gallocr_free(allocr);
    ggml_free(ctx);
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    return 0;
}
