/*
References:
https://github.com/NexaAI/nexa-ggml/blob/main/examples/magika/main.cpp
https://github.com/NexaAI/nexa-ggml/blob/main/examples/simple/simple-ctx.cpp

1. Model Loading : decide CPU, CUDA, or Metal backend, and load weights and biases for each layer.
2. Graph Building
3. Computation
4. Main Function
5. Memory Management

The key feature of this implementation is its ability to use different backends (CPU, CUDA, or Metal) for computation, 
depending on what's available. This allows for potential performance improvements on systems with compatible GPUs.
*/

#include "ggml.h"
#include "ggml-alloc.h"           // Include GGML allocation utilities
#include "ggml-backend.h"         // Include GGML backend interface

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"            // Conditionally include CUDA backend if GGML_USE_CUDA is defined
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"           // Conditionally include Metal backend if GGML_USE_METAL is defined
#endif

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
#include <inttypes.h> // For PRId64 macro

// Define a default log callback function
static void ggml_log_callback_default(ggml_log_level level, const char *text, void *user_data)
{
    (void)level;                  // Unused parameter
    (void)user_data;              // Unused parameter
    fputs(text, stderr);          // Write the log text to stderr
    fflush(stderr);               // Flush the stderr stream
}

// Define the structure for a two-layer MLP
struct mlp_model {
    // Weights and biases for each layer
    struct ggml_tensor * w1;
    struct ggml_tensor * b1;
    struct ggml_tensor * w2;
    struct ggml_tensor * b2;
    struct ggml_context * ctx;       // GGML context for tensor management
    ggml_backend_t backend = NULL;   // Backend for computation (CPU, CUDA, METAL)
    ggml_backend_buffer_t buffer;    // Backend buffer to store tensor data
};

// Function to load the model from a file
bool load_model(const std::string & fname, mlp_model & model) {
    // initialize the backend
    #ifdef GGML_USE_CUDA
        fprintf(stderr, "%s: using CUDA backend\n", __func__);
        model.backend = ggml_backend_cuda_init(0); // Initialize CUDA backend for device 0
        if (!model.backend)
        {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    #endif

    #ifdef GGML_USE_METAL
        fprintf(stderr, "%s: using Metal backend\n", __func__);
        ggml_backend_metal_log_set_callback(ggml_log_callback_default, nullptr); // Set Metal logging callback
        model.backend = ggml_backend_metal_init(); // Initialize Metal backend
        if (!model.backend)
        {
            fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
        }
    #endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!model.backend)
    {
        model.backend = ggml_backend_cpu_init();
    }

    // Create GGML context with no allocation
    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ 0, // We will not allocate any memory in this context
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true, // No allocation
    };
    
    // Create the context
    model.ctx = ggml_init(ggml_params);

    // Now, create the GGUF context using the existing GGML context
    struct gguf_init_params params = {
        /*.no_alloc   =*/ true,
        /*.ctx        =*/ &model.ctx, // Use the context we created (pass address of model.ctx)
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

    // Number of tensors in the model
    // int num_tensors = 4;

    // // Compute required memory for the context (overhead)
    // size_t mem_size = ggml_tensor_overhead() * num_tensors;

    // Create the backend buffer and allocate the tensors from the context
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // Load data from CPU memory to backend buffer
    ggml_backend_tensor_set(model.w1, model.w1->data, 0, ggml_nbytes(model.w1));
    ggml_backend_tensor_set(model.b1, model.b1->data, 0, ggml_nbytes(model.b1));
    ggml_backend_tensor_set(model.w2, model.w2->data, 0, ggml_nbytes(model.w2));
    ggml_backend_tensor_set(model.b2, model.b2->data, 0, ggml_nbytes(model.b2));

    // Print dimensions of loaded tensors
    fprintf(stdout, "w1 dimensions: %" PRId64 " x %" PRId64 "\n", model.w1->ne[0], model.w1->ne[1]);
    fprintf(stdout, "b1 dimensions: %" PRId64 "\n", model.b1->ne[0]);
    fprintf(stdout, "w2 dimensions: %" PRId64 " x %" PRId64 "\n", model.w2->ne[0], model.w2->ne[1]);
    fprintf(stdout, "b2 dimensions: %" PRId64 "\n", model.b2->ne[0]);

    gguf_free(ctx);
    return true;
}


/* Used to debug about intermediate tensor information */
void print_tensor_stats(const char* name, struct ggml_tensor* t) {
    size_t size = ggml_nelements(t);
    std::vector<float> data(size);

    // Copy data from backend to CPU
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));

    float sum = 0, min = INFINITY, max = -INFINITY;
    for (size_t i = 0; i < size; i++) {
        sum += data[i];
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
    }
    printf("%s: min=%f, max=%f, mean=%f\n", name, min, max, sum/size);
}


// Function to build the compute graph
struct ggml_cgraph * build_graph(
        struct ggml_context * ctx0,
        const mlp_model & model,
        const std::vector<float> & input_data,
        struct ggml_tensor ** result_ptr) {

    // Create computation graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // Create input tensor
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, input_data.size());
    ggml_set_name(input, "input");

    // Copy data into the backend buffer
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));

    ggml_tensor * cur = input;

    // First layer
    cur = ggml_mul_mat(ctx0, model.w1, cur);
    ggml_set_name(cur, "mul_mat_0");
    cur = ggml_add(ctx0, cur, model.b1);
    ggml_set_name(cur, "add_0");
    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "relu_0"); // ReLU activation function makes all negative values zero

    // Second layer
    cur = ggml_mul_mat(ctx0, model.w2, cur);
    ggml_set_name(cur, "mul_mat_1");
    cur = ggml_add(ctx0, cur, model.b2);
    ggml_set_name(cur, "add_1");

    *result_ptr = cur;

    ggml_build_forward_expand(gf, cur);

    return gf;
}

// Function to compute the graph
struct ggml_tensor * compute_graph(
        struct ggml_cgraph * gf,
        const mlp_model & model,
        ggml_gallocr_t allocr,
        const int n_threads) {

    // Allocate memory for graph tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    if (ggml_backend_is_cpu(model.backend))
    {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads); // Set number of threads for CPU backend
    }

#ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend))
    {
        ggml_backend_metal_set_n_cb(model.backend, n_threads); // Set number of threads for Metal backend
    }
#endif

    ggml_backend_graph_compute(model.backend, gf); // Perform the computation

    // Return the output tensor (last node in the graph)
    return gf->nodes[gf->n_nodes - 1];
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
    std::vector<float> input_data = {0.5, 0.4, 0.3, 0.2, 0.1};

    // Allocate memory for computations
    size_t buf_size = 16*1024*1024; // 16 MB
    std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // Defer tensor allocation
    };

    // Initialize GGML context
    struct ggml_context * ctx0 = ggml_init(params);

    // Build the computation graph
    struct ggml_tensor * result = NULL;
    struct ggml_cgraph * gf = build_graph(ctx0, model, input_data, &result);

    // Calculate the temporary memory required to compute
    ggml_gallocr_t allocr = NULL;

    {
        // Create a new allocator
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // Reserve memory based on the graph
        ggml_gallocr_reserve(allocr, gf);

        // Get required buffer size
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);

        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size / 1024.0);
    }

    // Compute the graph
    struct ggml_tensor * final_result = compute_graph(gf, model, allocr, 1);

    // Now that computation is done, we can print the stats
    print_tensor_stats("Input", ggml_get_tensor(ctx0, "input"));
    print_tensor_stats("After w1 multiplication", ggml_get_tensor(ctx0, "mul_mat_0"));
    print_tensor_stats("After b1 addition", ggml_get_tensor(ctx0, "add_0"));
    print_tensor_stats("After FC1 ReLU", ggml_get_tensor(ctx0, "relu_0"));
    print_tensor_stats("After w2 multiplication", ggml_get_tensor(ctx0, "mul_mat_1"));
    print_tensor_stats("After b2 addition", ggml_get_tensor(ctx0, "add_1"));
    print_tensor_stats("Final output", final_result);

    // Copy result data from backend memory to CPU
    size_t output_size = ggml_nelements(final_result);
    std::vector<float> output_vector(output_size);
    ggml_backend_tensor_get(final_result, output_vector.data(), 0, ggml_nbytes(final_result));

    fprintf(stdout, "%s: output vector: [", __func__);
    for (size_t i = 0; i < output_vector.size(); ++i) {
        fprintf(stdout, "%f", output_vector[i]);
        if (i < output_vector.size() - 1) fprintf(stdout, ", ");
    }
    fprintf(stdout, "]\n");

    // Release backend memory used for computation
    ggml_gallocr_free(allocr);

    ggml_free(ctx0);
    ggml_free(model.ctx);

    // Release backend buffer and free backend
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);

    return 0;
}