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
    gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    // Load weights and biases for each layer
    model.w1 = ggml_get_tensor(model.ctx, "w1");
    model.b1 = ggml_get_tensor(model.ctx, "b1");
    model.w2 = ggml_get_tensor(model.ctx, "w2");
    model.b2 = ggml_get_tensor(model.ctx, "b2");

    return true;
}

/* Used to debug about intermediate tensor information */
void print_tensor(ggml_tensor* tensor, const char* name) {
    float* data = ggml_get_data_f32(tensor);
    int size = ggml_nelements(tensor);
    printf("%s: [", name);
    for (int i = 0; i < size; i++) {
        printf("%f", data[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
}


// build the compute graph
std::vector<float> build_graph(
        const three_layer_nn_model & model,
        const int n_threads,
        std::vector<float> input_data,
        const char * fname_cgraph
        )
{
    // Allocate memory for computations
    static size_t buf_size = 100000 * sizeof(float) * 4;
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
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 5);  // Assuming input size is 5
    memcpy(input->data, input_data.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");

    // Forward pass through the network
    ggml_tensor * cur = input;

    // First layer
    cur = ggml_mul_mat(ctx0, model.nexa_fc1_weight, cur);
    cur = ggml_add(ctx0, cur, model.nexa_fc1_bias);
    print_tensor(cur, "After FC1");
    cur = ggml_relu(ctx0, cur); // ReLU activation
    print_tensor(cur, "After FC1 ReLU");

    // Second layer
    cur = ggml_mul_mat(ctx0, model.nexa_fc2_weight, cur);
    cur = ggml_add(ctx0, cur, model.nexa_fc2_bias);
    cur = ggml_relu(ctx0, cur);

    ggml_tensor * result = cur;
    ggml_set_name(result, "result");
    
    // Build and compute the graph
    ggml_build_forward_expand(gf, result);
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    // Export the compute graph if filename is provided
    if (fname_cgraph) {
        ggml_graph_export(gf, fname_cgraph);
        fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__, fname_cgraph);
    }

    // Get the output probabilities
    const float * output_data = ggml_get_data_f32(result);
    std::vector<float> output_vector(output_data, output_data + 3);  // Assuming output size is 3

    // Free the GGML context
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

    mlp_modelmodel;

    // Load the model
    {
        const int64_t t_start_us = ggml_time_us();
        if (!three_layer_nn_model_load(argv[1], model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, argv[1]);
            return 1;
        }

        const int64_t t_load_us = ggml_time_us() - t_start_us;
        fprintf(stdout, "%s: loaded model in %8.2f ms\n", __func__, t_load_us / 1000.0f);
    }

    // Prepare input data (replace this with your actual input)
    std::vector<float> input_data = {0.1, 0.2, 0.3, 0.4, 0.5};

    // Evaluate the model
    std::vector<float> output = build_graph(model, 1, input_data, nullptr);
    
    // Print the output vector
    fprintf(stdout, "%s: output vector: [", __func__);
    for (size_t i = 0; i < output.size(); ++i) {
        fprintf(stdout, "%f", output[i]);
        if (i < output.size() - 1) fprintf(stdout, ", ");
    }
    fprintf(stdout, "]\n");

    // Free the model context
    ggml_free(model.ctx);
    return 0;
}