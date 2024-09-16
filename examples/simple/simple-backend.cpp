#include "ggml.h"                 // Include the main GGML header
#include "ggml-alloc.h"           // Include GGML allocation utilities
#include "ggml-backend.h"         // Include GGML backend interface

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"            // Conditionally include CUDA backend if GGML_USE_CUDA is defined
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"           // Conditionally include Metal backend if GGML_USE_METAL is defined
#endif

#include <cassert>                // Include for assert macro
#include <cmath>                  // Include for math functions
#include <cstdio>                 // Include for I/O operations
#include <cstring>                // Include for string manipulation
#include <fstream>                // Include for file stream operations
#include <map>                    // Include for std::map container
#include <string>                 // Include for std::string
#include <vector>                 // Include for std::vector container

/*
Model Structure:
   - A `simple_model` struct is defined, containing two tensors (`a` and `b`), a backend, a backend buffer, and a GGML context.

1. Model Loading:
   - The `load_model` function initializes the model:
     - It sets up the appropriate backend (CUDA, Metal, or CPU).
     - Creates a GGML context and two tensors (a and b) representing 2D matrices.
     - Allocates memory for these tensors in the backend buffer.
     - Loads data into the tensors.

2. Graph Building:
   - The `build_graph` function creates a computation graph for matrix multiplication:
     - It sets up a temporary GGML context.
     - Defines the operation: result = a * b^T (matrix multiplication with b transposed).
     - Builds the forward pass of the graph.

3. Computation:
   - The `compute` function performs the actual computation:
     - It builds the graph, allocates memory for intermediate tensors.
     - Executes the computation using the specified backend.

4. Main Function:
   - Initializes input matrices A (4x2) and B (3x2, stored in transposed form).
   - Loads the model with these matrices.
   - Allocates memory for computation.
   - Performs the matrix multiplication.
   - Retrieves and prints the result.

5. Memory Management:
   - The code properly frees allocated resources at the end.

The key feature of this implementation is its ability to use different backends (CPU, CUDA, or Metal) for computation, depending on what's available. This allows for potential performance improvements on systems with compatible GPUs.

The matrix multiplication performed is between a 4x2 matrix (A) and a 2x3 matrix (B^T), resulting in a 4x3 matrix. The code demonstrates how to set up, compute, and retrieve results using the GGML library, showcasing its flexibility in handling different computation backends.
*/

// Define a default log callback function
static void ggml_log_callback_default(ggml_log_level level, const char *text, void *user_data)
{
    (void)level;                  // Unused parameter
    (void)user_data;              // Unused parameter
    fputs(text, stderr);          // Write the log text to stderr
    fflush(stderr);               // Flush the stderr stream
}

// Define a struct to represent a simple model with two tensors
struct simple_model
{
    struct ggml_tensor *a;        // Pointer to tensor 'a'
    struct ggml_tensor *b;        // Pointer to tensor 'b'

    ggml_backend_t backend = NULL; // Backend for computation (CPU, CUDA, METAL)

    ggml_backend_buffer_t buffer; // Backend buffer to store tensor data

    struct ggml_context *ctx;     // GGML context for tensor management
};

// initialize the tensors of the model in this case two matrices 2x2
// 1. Model Loading
void load_model(simple_model &model, float *a, float *b, int rows_A, int cols_A, int rows_B, int cols_B)
{
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

    int num_tensors = 2; // Number of tensors in the model, namely 'a' and 'b'

    // Set up initialization parameters for GGML context
    struct ggml_init_params params
    {
        /*.mem_size   =*/ggml_tensor_overhead() * num_tensors, // Calculate required memory size
        /*.mem_buffer =*/NULL,                                 // No pre-allocated buffer
        /*.no_alloc   =*/true,                                 // Defer memory allocation
    };

    // create context
    model.ctx = ggml_init(params);

    // Create tensors for matrices A and B
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_B, rows_B);

    // create a backend buffer (backend memory) and alloc the tensors from the context
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));
}

// build the compute graph to perform a matrix multiplication
// 2. Graph Building
struct ggml_cgraph *build_graph(const simple_model &model)
{
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

    // Create new computation graph
    struct ggml_cgraph *gf = ggml_new_graph(ctx0);

    // Define matrix multiplication operation: result = a x b
    struct ggml_tensor *result = ggml_mul_mat(ctx0, model.a, model.b);

    // Build the forward pass of the graph
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

// Function to perform the computation using the specified backend
// 3. Computation
struct ggml_tensor *compute(const simple_model &model, ggml_gallocr_t allocr)
{
    // reset the allocator to free all the memory allocated during the previous inference
    struct ggml_cgraph *gf = build_graph(model);

    // Allocate memory for graph tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    int n_threads = 1; // Number of threads for multi-threaded operations

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

// 4. Main Function
int main(void)
{
    ggml_time_init(); // Initialize GGML timing functionality

    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;

    float matrix_A[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6};

    const int rows_B = 3, cols_B = 2;
    /* Transpose([
        10, 9, 5,
        5, 9, 4
    ]) 2 rows, 3 cols */
    float matrix_B[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4};

    // Initialize the model
    simple_model model;
    load_model(model, matrix_A, matrix_B, rows_A, cols_A, rows_B, cols_B); // 1. Model Loading

    // calculate the temporaly memory required to compute
    ggml_gallocr_t allocr = NULL;

    {
        // Create a new allocator
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // create the worst case graph for memory usage estimation
        struct ggml_cgraph *gf = build_graph(model); // 2. Graph Building
        ggml_gallocr_reserve(allocr, gf);

        // Get required buffer size
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);

        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size / 1024.0);
    }

    // perform computation
    struct ggml_tensor *result = compute(model, allocr); // 3. Computation

    // create a array to print result
    std::vector<float> out_data(ggml_nelements(result));

    // Copy result data from backend memory to CPU
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    // expected result:
    // [ 60.00 110.00 54.00 29.00
    //  55.00 90.00 126.00 28.00
    //  50.00 54.00 42.00 64.00 ]

    printf("mul mat (%d x %d) (transposed result):\n[", (int)result->ne[0], (int)result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++)
    {
        if (j > 0)
        {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++)
        {
            printf(" %.2f", out_data[i * result->ne[1] + j]);
        }
    }
    printf(" ]\n");

    // release backend memory used for computation
    ggml_gallocr_free(allocr); // 4. Memory Management

    // free memory
    ggml_free(model.ctx);

    // release backend memory and free backend
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    return 0;
}