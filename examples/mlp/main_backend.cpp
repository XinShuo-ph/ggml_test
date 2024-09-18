/*
References:
https://github.com/NexaAI/nexa-ggml/blob/main/examples/magika/main.cpp
https://github.com/NexaAI/nexa-ggml/blob/main/examples/simple/simple-ctx.cpp
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
    // Initialize the backend
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

    // If there aren't GPU Backends, fallback to CPU backend
    if (!model.backend)
    {
        model.backend = ggml_backend_cpu_init();
    }

    // Initialize the GGML context
    size_t ctx_size = 0;
    ctx_size += 100 * ggml_tensor_overhead(); // tensors

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ ctx_size, // 16 MB for metadata
        /*.mem_buffer =*/ NULL,
         /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
    };
    model.ctx = ggml_init(ggml_params);
    if (!model.ctx) {
        fprintf(stderr, "%s: ggml_init() failed\n", __func__);
        return false;
    }

    // Now, initialize the GGUF context
    struct gguf_init_params params = {
        /*.no_alloc   =*/ true,  // We will handle tensor data allocation
        /*.ctx        =*/ &model.ctx,  // Pass the context we just created
    };

    struct gguf_context * uctx = gguf_init_from_file(fname.c_str(), params);
    if (!uctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        ggml_free(model.ctx);
        return false;
    }

    // Load weights and biases for each layer
    model.w1 = ggml_get_tensor(model.ctx, "fc1.weight");
    model.b1 = ggml_get_tensor(model.ctx, "fc1.bias");
    model.w2 = ggml_get_tensor(model.ctx, "fc2.weight");
    model.b2 = ggml_get_tensor(model.ctx, "fc2.bias");

    if (!model.w1 || !model.b1 || !model.w2 || !model.b2) {
        fprintf(stderr, "%s: failed to load model tensors\n", __func__);
        gguf_free(uctx);
        ggml_free(model.ctx);
        return false;
    }

    // Create the backend buffer and allocate the tensors from the context
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // Copy data from CPU to backend
    ggml_backend_tensor_set(model.w1, model.w1->data, 0, ggml_nbytes(model.w1));
    ggml_backend_tensor_set(model.b1, model.b1->data, 0, ggml_nbytes(model.b1));
    ggml_backend_tensor_set(model.w2, model.w2->data, 0, ggml_nbytes(model.w2));
    ggml_backend_tensor_set(model.b2, model.b2->data, 0, ggml_nbytes(model.b2));

    // Print dimensions of loaded tensors
    fprintf(stdout, "w1 dimensions: %" PRId64 " x %" PRId64 "\n", model.w1->ne[0], model.w1->ne[1]);
    fprintf(stdout, "b1 dimensions: %" PRId64 "\n", model.b1->ne[0]);
    fprintf(stdout, "w2 dimensions: %" PRId64 " x %" PRId64 "\n", model.w2->ne[0], model.w2->ne[1]);
    fprintf(stdout, "b2 dimensions: %" PRId64 "\n", model.b2->ne[0]);

    gguf_free(uctx);
    return true;
}

/* Used to debug intermediate tensor information */
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
    printf("%s: min=%f, max=%f, mean=%f\n", name, min, max, sum / size);
}

// Function to build the compute graph
struct ggml_cgraph * build_graph(
        const mlp_model & model,
        const std::vector<float>::size_type input_data_size,  // input data should not be set when building graph
        struct ggml_tensor ** result_ptr) {

    // the context for compute graph  should be different from the ctx for model
    // for instance, refer to gpt2_graph, a new context is created to build the graph
    //  looks like the ctx for a model is for hyperparameters of the tensors,
    //      while the ctx for a graph is for parameters/operations defining the graph

    // Allocate buffer for temporary context
    static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    // Set up initialization parameters for temporary context
    struct ggml_init_params params0 = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/buf.data(),
        /*.no_alloc   =*/true, // Defer tensor allocation
    };

    // create a temporally context to build the graph
    struct ggml_context *ctx0 = ggml_init(params0);


    // Create computation graph
    std::cout << "Creating computation graph." << std::endl;
    struct ggml_cgraph * gf = ggml_new_graph(ctx0);

    // Create input tensor
    std::cout << "Creating input tensor." << std::endl;
    struct ggml_tensor * input = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, input_data_size);
    ggml_set_name(input, "input");

    ggml_set_input(input); // copying dta to input tensor should happen after the graph is built

    // // Copy data into the backend buffer
    // std::cout << "Copying data into the backend buffer." << std::endl;
    // ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));

    ggml_tensor * cur = input;

    // First layer
    std::cout << "First layer." << std::endl;
    cur = ggml_mul_mat(ctx0, model.w1, cur);
    ggml_set_name(cur, "mul_mat_0");
    cur = ggml_add(ctx0, cur, model.b1);
    ggml_set_name(cur, "add_0");
    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "relu_0"); // ReLU activation function

    // Second layer
    std::cout << "Second layer." << std::endl;
    cur = ggml_mul_mat(ctx0, model.w2, cur);
    ggml_set_name(cur, "mul_mat_1");
    cur = ggml_add(ctx0, cur, model.b2);
    ggml_set_name(cur, "add_1");

    *result_ptr = cur;

    ggml_build_forward_expand(gf, cur);

    return gf;
}

int main(int argc, char ** argv) {
    srand((unsigned int)time(NULL));
    ggml_time_init();

    // Print CUDA availability
    #ifdef GGML_USE_CUDA
        std::cout << "GGML_USE_CUDA is defined. CUDA backend is available." << std::endl;
    #else
        std::cout << "GGML_USE_CUDA is not defined. CUDA backend is not available." << std::endl;
    #endif

    // Print Metal availability
    #ifdef GGML_USE_METAL
        std::cout << "GGML_USE_METAL is defined. Metal backend is available." << std::endl;
    #else
        std::cout << "GGML_USE_METAL is not defined. Metal backend is not available." << std::endl;
    #endif

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

    // Use the same context as the model for computation
    std::cout << "Using the same context as the model for computation." << std::endl;
    struct ggml_context * ctx = model.ctx;

    // Create an allocator
    std::cout << "Creating an allocator." << std::endl;
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

    // Build the computation graph
    std::cout << "Building the computation graph." << std::endl;
    struct ggml_tensor * result = NULL;
    struct ggml_cgraph * gf = build_graph( model, input_data.size(), &result);

    // Reserve memory for the graph
    std::cout << "Reserving memory for the graph." << std::endl;
    ggml_gallocr_reserve(allocr, gf);

    // Get required buffer size
    std::cout << "Getting required buffer size." << std::endl;
    size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);

    fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size / 1024.0);

    // Allocate memory for graph tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    // set the graph inputs after the graph is built and memory is allocated
    struct ggml_tensor * input = ggml_graph_get_tensor(gf, "input");  // Get the input tensor, name is set when build_graph()
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
    

    // Set number of threads if using CPU backend
    std::cout << "Setting number of threads if using CPU backend." << std::endl;
    int n_threads = 4; // Adjust as needed
    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    #ifdef GGML_USE_METAL
    if (ggml_backend_is_metal(model.backend)) {
        ggml_backend_metal_set_n_cb(model.backend, n_threads); // Set number of threads for Metal backend
    }
    #endif

    // Compute the graph
    std::cout << "Computing the graph." << std::endl;
    ggml_backend_graph_compute(model.backend, gf);

    // Copy result data from backend memory to CPU
    std::cout << "Copying result data from backend memory to CPU." << std::endl;
    std::vector<float> output_data(ggml_nelements(result));
    ggml_backend_tensor_get(result, output_data.data(), 0, ggml_nbytes(result));

    fprintf(stdout, "%s: output vector: [", __func__);
    for (size_t i = 0; i < output_data.size(); ++i) {
        fprintf(stdout, "%f", output_data[i]);
        if (i < output_data.size() - 1) fprintf(stdout, ", ");
    }
    fprintf(stdout, "]\n");

    // Release backend memory used for computation
    ggml_gallocr_free(allocr);

    // Free contexts
    ggml_free(model.ctx);        // Free the model context

    // Release backend memory and free backend
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);

    return 0;
}
