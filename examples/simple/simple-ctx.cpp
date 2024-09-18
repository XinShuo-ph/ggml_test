/*
1. Allocate ggml_context to store tensor data
2. Create tensors and set data
3. Create a ggml_cgraph for mul_mat operation
4. Run the computation
5. Retrieve results (output tensors)
6. Free memory and exit
*/

#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;

    // the context to define the tensor information (dimensions, size, memory data)
    struct ggml_context * ctx;
};

// initialize the tensors of the model in this case two matrices 2x2
void load_model(simple_model & model, float * a, float * b, int rows_A, int cols_A, int rows_B, int cols_B) {
    // 1. Allocate `ggml_context` to store tensor data
    size_t ctx_size = 0;
    // {
    //     ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32); // tensor a
    //     ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32); // tensor b
    //     ctx_size += 2 * ggml_tensor_overhead(), // tensors
    //     ctx_size += ggml_graph_overhead(); // compute graph
    //     ctx_size += 1024; // some overhead
    // }
    ctx_size = 1 * 1024 * 1024; // 1 MB

    // Allocate `ggml_context` to store tensor data
    struct ggml_init_params params {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false, // NOTE: this should be false when using the legacy API
    };
    // create context
    model.ctx = ggml_init(params);

    // 2. Create tensors and set data
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_B, rows_B);
    memcpy(model.a->data, a, ggml_nbytes(model.a));
    memcpy(model.b->data, b, ggml_nbytes(model.b));
}

// 3. Create a `ggml_cgraph` for mul_mat operation
struct ggml_cgraph * build_graph(const simple_model& model) {
    struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);

    // result = a*b^T
    struct ggml_tensor * result = ggml_mul_mat(model.ctx, model.a, model.b);

    ggml_build_forward_expand(gf, result); // Mark the "result" tensor to be computed
    ggml_graph_dump_dot(gf, NULL, "simple-ctx.dot"); // Print the cgraph for visualization
    return gf;
}

// 4. Run the computation
struct ggml_tensor * compute(const simple_model & model) {
    struct ggml_cgraph * gf = build_graph(model);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);

    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}

int main(void) {
    ggml_time_init();

    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;

    float matrix_A[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };

    const int rows_B = 3, cols_B = 2;
    /* Transpose([
        10, 9, 5,
        5, 9, 4
    ]) 2 rows, 3 cols */
    float matrix_B[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
    };

    simple_model model;
    load_model(model, matrix_A, matrix_B, rows_A, cols_A, rows_B, cols_B);

    // perform computation in cpu
    struct ggml_tensor * result = compute(model);

    // 5. Retrieve results (output tensors)
    std::vector<float> out_data(ggml_nelements(result));
    memcpy(out_data.data(), result->data, ggml_nbytes(result));

    printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");

    // 6. Free memory and exit
    ggml_free(model.ctx);
    return 0;
}