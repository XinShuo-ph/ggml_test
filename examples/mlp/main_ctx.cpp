/*
References:
https://github.com/NexaAI/nexa-ggml/blob/main/examples/magika/main.cpp
https://github.com/NexaAI/nexa-ggml/blob/main/examples/simple/simple-ctx.cpp
*/
#include "ggml.h"
#include "common.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cstdlib>  // Added for malloc and srand
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

// Define the structure for a two layer MLP
struct mlp_model {
    // Weights and biases for each layer
    struct ggml_tensor * w1;
    struct ggml_tensor * b1;
    struct ggml_tensor * w2;
    struct ggml_tensor * b2;
    struct ggml_context * ctx;
};

// Function to load the model from a file
bool load_model(const std::string & fname, mlp_model & model) {
    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &model.ctx,
    };
    
    struct gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }

    // Load weights and biases for each layer
    model.w1 = ggml_get_tensor(model.ctx, "fc1.weight");
    model.b1 = ggml_get_tensor(model.ctx, "fc1.bias");
    model.w2 = ggml_get_tensor(model.ctx, "fc2.weight");
    model.b2 = ggml_get_tensor(model.ctx, "fc2.bias");

    if (!model.w1 || !model.b1 || !model.w2 || !model.b2) {
        fprintf(stderr, "%s: failed to load model tensors\n", __func__);
        gguf_free(ctx);
        return false;
    }

    // Print dimensions of loaded tensors with correct format specifiers
    fprintf(stdout, "w1 dimensions: %ld x %ld\n", model.w1->ne[0], model.w1->ne[1]);
    fprintf(stdout, "b1 dimensions: %ld\n", model.b1->ne[0]);
    fprintf(stdout, "w2 dimensions: %ld x %ld\n", model.w2->ne[0], model.w2->ne[1]);
    fprintf(stdout, "b2 dimensions: %ld\n", model.b2->ne[0]);

    gguf_free(ctx);
    return true;
}

/* Used to debug about intermediate tensor information */
void print_tensor_stats(const char* name, struct ggml_tensor* t) {
    float* data = (float*)t->data;
    size_t size = ggml_nelements(t);
    float sum = 0, min = INFINITY, max = -INFINITY;
    for (size_t i = 0; i < size; i++) {
        sum += data[i];
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
    }
    printf("%s: min=%f, max=%f, mean=%f\n", name, min, max, sum/size);
}

// build the compute graph
std::vector<float> build_graph(
        const mlp_model & model,
        const int n_threads,
        std::vector<float> input_data,
        const char * fname_cgraph
        )
{
    // Allocate memory for computations
    static size_t buf_size = 100000 * sizeof(float) * 4; // this is a rough estimate
    static void * buf = malloc(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf,
        /*.no_alloc   =*/ false,
    };

    // Initialize GGML context
    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // Create input tensor
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, input_data.size());
    memcpy(input->data, input_data.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");
    print_tensor_stats("Input", input);

    ggml_tensor * cur = input;

    // First layer
    cur = ggml_mul_mat(ctx0, model.w1, cur);
    print_tensor_stats("After w1 multiplication", cur);
    cur = ggml_add(ctx0, cur, model.b1);
    print_tensor_stats("After b1 addition", cur);
    cur = ggml_relu(ctx0, cur);
    print_tensor_stats("After FC1 ReLU", cur);

    // Second layer
    cur = ggml_mul_mat(ctx0, model.w2, cur);
    print_tensor_stats("After w2 multiplication", cur);
    cur = ggml_add(ctx0, cur, model.b2);
    print_tensor_stats("After b2 addition", cur);
    cur = ggml_relu(ctx0, cur);
    print_tensor_stats("Final output", cur);

    ggml_tensor * result = cur;
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    if (fname_cgraph) {
        ggml_graph_export(gf, fname_cgraph);
        fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__, fname_cgraph);
    }

    const float * output_data = ggml_get_data_f32(result);
    std::vector<float> output_vector(output_data, output_data + ggml_nelements(result));

    ggml_free(ctx0);
    return output_vector;
}


int main(int argc, char ** argv) {
    srand(time(NULL));
    ggml_time_init();

    if (argc != 2) {
        fprintf(stderr, "Usage: %s path/to/model.gguf\n", argv[0]);
        exit(0);
    }

    mlp_model model;

    {
        const int64_t t_start_us = ggml_time_us();
        if (!load_model(argv[1], model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, argv[1]);
            return 1;
        }

        const int64_t t_load_us = ggml_time_us() - t_start_us;
        fprintf(stdout, "%s: loaded model in %8.2f ms\n", __func__, t_load_us / 1000.0f);
    }

    print_tensor_stats("w1", model.w1);
    print_tensor_stats("b1", model.b1);
    print_tensor_stats("w2", model.w2);
    print_tensor_stats("b2", model.b2);

    // Prepare input data with the correct size
    std::vector<float> input_data = {0.1, 0.2, 0.3, 0.4, 0.5};

    // run and build the model
    std::vector<float> output = build_graph(model, 1, input_data, nullptr);
    
    fprintf(stdout, "%s: output vector: [", __func__);
    for (size_t i = 0; i < output.size(); ++i) {
        fprintf(stdout, "%f", output[i]);
        if (i < output.size() - 1) fprintf(stdout, ", ");
    }
    fprintf(stdout, "]\n");

    ggml_free(model.ctx);
    return 0;
}