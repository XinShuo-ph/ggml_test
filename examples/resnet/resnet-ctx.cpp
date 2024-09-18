/*
References:
https://github.com/NexaAI/nexa-ggml/blob/main/examples/magika/main.cpp
https://github.com/NexaAI/nexa-ggml/blob/main/examples/simple/simple-ctx.cpp
*/

#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <inttypes.h>

// Function to print tensor statistics
void print_tensor_stats(const char *name, struct ggml_tensor *t)
{
    float *data = (float *)t->data;
    size_t size = ggml_nelements(t);
    float sum = 0, min = INFINITY, max = -INFINITY;
    for (size_t i = 0; i < size; i++)
    {
        sum += data[i];
        if (data[i] < min)
            min = data[i];
        if (data[i] > max)
            max = data[i];
    }
    printf("%s: min=%f, max=%f, mean=%f\n", name, min, max, sum / size);
}

// Define the structure for the ResNet-18 model
struct block_params
{
    // conv1
    struct ggml_tensor *conv1_weight;
    // bn1
    struct ggml_tensor *bn1_weight;
    struct ggml_tensor *bn1_bias;
    // conv2
    struct ggml_tensor *conv2_weight;
    // bn2
    struct ggml_tensor *bn2_weight;
    struct ggml_tensor *bn2_bias;
    // Optional downsample
    bool has_downsample;
    struct ggml_tensor *downsample_conv_weight;
    struct ggml_tensor *downsample_bn_weight;
    struct ggml_tensor *downsample_bn_bias;
};

struct resnet_model
{
    // Initial convolutional layer
    struct ggml_tensor *conv1_weight;
    struct ggml_tensor *bn1_weight;
    struct ggml_tensor *bn1_bias;

    // Layers
    std::vector<block_params> layer1;
    std::vector<block_params> layer2;
    std::vector<block_params> layer3;
    std::vector<block_params> layer4;

    // Fully connected layer
    struct ggml_tensor *fc_weight;
    struct ggml_tensor *fc_bias;

    // GGML context
    struct ggml_context *ctx;
};

// Function to initialize the model parameters
bool init_model(resnet_model &model)
{
    // Allocate memory for computations
    size_t buf_size = 1024 * 1024 * 1024; // 1 GB
    struct ggml_init_params params = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/NULL,
        /*.no_alloc   =*/false,
    };

    // Initialize GGML context
    model.ctx = ggml_init(params);
    if (!model.ctx)
    {
        fprintf(stderr, "Failed to initialize GGML context\n");
        return false;
    }

    // Helper function to initialize tensors
    auto init_tensor = [&](struct ggml_tensor *tensor, const char *name)
    {
        float *data = (float *)tensor->data;
        size_t size = ggml_nelements(tensor);
        for (size_t i = 0; i < size; ++i)
        {
            data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f; // Small random values
        }
        ggml_set_name(tensor, name);
        print_tensor_stats(name, tensor);
    };

    // Initial convolutional layer
    model.conv1_weight = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, 64, 3, 7, 7); // [OC, IC, KH, KW]
    init_tensor(model.conv1_weight, "conv1_weight");

    model.bn1_weight = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 64);
    model.bn1_bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 64);
    for (int i = 0; i < 64; ++i)
    {
        ((float *)model.bn1_weight->data)[i] = 1.0f;
        ((float *)model.bn1_bias->data)[i] = 0.0f;
    }
    ggml_set_name(model.bn1_weight, "bn1_weight");
    ggml_set_name(model.bn1_bias, "bn1_bias");
    print_tensor_stats("bn1_weight", model.bn1_weight);
    print_tensor_stats("bn1_bias", model.bn1_bias);

    // Helper lambda to initialize block parameters
    auto init_block = [&](int in_channels, int out_channels, int stride, bool downsample, const std::string &block_name)
    {
        block_params block;

        // conv1
        block.conv1_weight = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, out_channels, in_channels, 3, 3); // [OC, IC, KH, KW]
        init_tensor(block.conv1_weight, (block_name + "_conv1_weight").c_str());

        // bn1
        block.bn1_weight = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, out_channels);
        block.bn1_bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, out_channels);
        for (int i = 0; i < out_channels; ++i)
        {
            ((float *)block.bn1_weight->data)[i] = 1.0f;
            ((float *)block.bn1_bias->data)[i] = 0.0f;
        }
        ggml_set_name(block.bn1_weight, (block_name + "_bn1_weight").c_str());
        ggml_set_name(block.bn1_bias, (block_name + "_bn1_bias").c_str());
        print_tensor_stats((block_name + "_bn1_weight").c_str(), block.bn1_weight);
        print_tensor_stats((block_name + "_bn1_bias").c_str(), block.bn1_bias);

        // conv2
        block.conv2_weight = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, out_channels, out_channels, 3, 3); // [OC, IC, KH, KW]
        init_tensor(block.conv2_weight, (block_name + "_conv2_weight").c_str());

        // bn2
        block.bn2_weight = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, out_channels);
        block.bn2_bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, out_channels);
        for (int i = 0; i < out_channels; ++i)
        {
            ((float *)block.bn2_weight->data)[i] = 1.0f;
            ((float *)block.bn2_bias->data)[i] = 0.0f;
        }
        ggml_set_name(block.bn2_weight, (block_name + "_bn2_weight").c_str());
        ggml_set_name(block.bn2_bias, (block_name + "_bn2_bias").c_str());
        print_tensor_stats((block_name + "_bn2_weight").c_str(), block.bn2_weight);
        print_tensor_stats((block_name + "_bn2_bias").c_str(), block.bn2_bias);

        block.has_downsample = (stride != 1) || (in_channels != out_channels);
        if (downsample)
        {
            block.downsample_conv_weight = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, out_channels, in_channels, 1, 1); // [OC, IC, KH, KW]
            init_tensor(block.downsample_conv_weight, (block_name + "_downsample_conv_weight").c_str());

            block.downsample_bn_weight = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, out_channels);
            block.downsample_bn_bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, out_channels);
            for (int i = 0; i < out_channels; ++i)
            {
                ((float *)block.downsample_bn_weight->data)[i] = 1.0f;
                ((float *)block.downsample_bn_bias->data)[i] = 0.0f;
            }
            ggml_set_name(block.downsample_bn_weight, (block_name + "_downsample_bn_weight").c_str());
            ggml_set_name(block.downsample_bn_bias, (block_name + "_downsample_bn_bias").c_str());
            print_tensor_stats((block_name + "_downsample_bn_weight").c_str(), block.downsample_bn_weight);
            print_tensor_stats((block_name + "_downsample_bn_bias").c_str(), block.downsample_bn_bias);
        }

        return block;
    };

    // Initialize layers
    // Layer 1
    for (int i = 0; i < 2; ++i) {
        model.layer1.push_back(init_block(64, 64, 1, false, "layer1_block" + std::to_string(i)));
    }
    // Layer 2
    for (int i = 0; i < 2; ++i)
    {
        bool downsample = (i == 0);
        model.layer2.push_back(init_block(64, 128, downsample ? 2 : 1, downsample, "layer2_block" + std::to_string(i)));
    }
    // Layer 3
    for (int i = 0; i < 2; ++i)
    {
        bool downsample = (i == 0);
        model.layer3.push_back(init_block(128, 256, downsample ? 2 : 1, downsample, "layer3_block" + std::to_string(i)));
    }
    // Layer 4
    for (int i = 0; i < 2; ++i)
    {
        bool downsample = (i == 0);
        model.layer4.push_back(init_block(256, 512, downsample ? 2 : 1, downsample, "layer4_block" + std::to_string(i)));
    }

    // Fully connected layer
    model.fc_weight = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, 512, 1000); // [N, M]
    init_tensor(model.fc_weight, "fc_weight");

    model.fc_bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 1000);
    for (int i = 0; i < 1000; ++i)
    {
        ((float *)model.fc_bias->data)[i] = 0.0f;
    }
    ggml_set_name(model.fc_bias, "fc_bias");
    print_tensor_stats("fc_bias", model.fc_bias);

    return true;
}

// Function to build the computation graph
struct ggml_cgraph *build_graph(
    resnet_model &model,
    struct ggml_tensor **input_ptr,
    struct ggml_tensor **result_ptr)
{
    // Use model.ctx as the GGML context
    struct ggml_context *ctx0 = model.ctx;

    // Create computation graph
    struct ggml_cgraph *gf = ggml_new_graph(ctx0);

    // Input tensor (batch size 1, channels 3, height 224, width 224)
    struct ggml_tensor *input = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 224, 224, 3, 1); // [N, IC, IH, IW]
    ggml_set_name(input, "input");
    *input_ptr = input;

    struct ggml_tensor *cur = input;

    // Initial convolution
    cur = ggml_conv_2d(ctx0, model.conv1_weight, cur, 2, 2, 3, 3, 1, 1); // [N, OC, OH, OW]
    ggml_set_name(cur, "conv1");

    // BatchNorm1
    float epsilon = 1e-5f;

    // Reshape bn1_weight and bn1_bias to [1, C, 1, 1]
    struct ggml_tensor *bn1_weight_reshaped = ggml_reshape_4d(ctx0, model.bn1_weight, 1, 1, 1, 64);
    struct ggml_tensor *bn1_bias_reshaped = ggml_reshape_4d(ctx0, model.bn1_bias, 1, 1, 1, 64);

    cur = ggml_norm(ctx0, cur, epsilon);
    cur = ggml_add(ctx0,
                   ggml_mul(ctx0, ggml_repeat(ctx0, bn1_weight_reshaped, cur), cur),
                   ggml_repeat(ctx0, bn1_bias_reshaped, cur));
    ggml_set_name(cur, "bn1");

    // ReLU
    cur = ggml_relu(ctx0, cur);
    ggml_set_name(cur, "relu1");

    // MaxPool (kernel size 3, stride 2, padding 1)
    cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, 3, 3, 2, 2, 1, 1);
    ggml_set_name(cur, "maxpool");

    // Helper lambda to process a layer
    auto process_layer = [&](std::vector<block_params> &layer_blocks, const std::string &layer_name, int stride = 1)
    {
        for (size_t i = 0; i < layer_blocks.size(); ++i)
        {
            block_params &block = layer_blocks[i];
            std::string block_name = layer_name + "." + std::to_string(i);

            struct ggml_tensor *identity = cur;

            // Check stride for the first block in layer2, layer3, and layer4
            int block_stride = (i == 0) ? stride : 1;

            // conv1
            cur = ggml_conv_2d(ctx0, block.conv1_weight, cur, block_stride, block_stride, 1, 1, 1, 1);
            ggml_set_name(cur, (block_name + ".conv1").c_str());

            // bn1
            struct ggml_tensor *bn1_weight_reshaped = ggml_reshape_4d(ctx0, block.bn1_weight, 1, 1, 1, block.bn1_weight->ne[0]);
            struct ggml_tensor *bn1_bias_reshaped = ggml_reshape_4d(ctx0, block.bn1_bias, 1, 1, 1, block.bn1_bias->ne[0]);

            cur = ggml_norm(ctx0, cur, epsilon);
            cur = ggml_add(ctx0,
                           ggml_mul(ctx0, ggml_repeat(ctx0, bn1_weight_reshaped, cur), cur),
                           ggml_repeat(ctx0, bn1_bias_reshaped, cur));
            ggml_set_name(cur, (block_name + ".bn1").c_str());

            // ReLU
            cur = ggml_relu(ctx0, cur);
            ggml_set_name(cur, (block_name + ".relu1").c_str());

            // conv2
            cur = ggml_conv_2d(ctx0, block.conv2_weight, cur, 1, 1, 1, 1, 1, 1);
            ggml_set_name(cur, (block_name + ".conv2").c_str());

            // bn2
            struct ggml_tensor *bn2_weight_reshaped = ggml_reshape_4d(ctx0, block.bn2_weight, 1, 1, 1, block.bn2_weight->ne[0]);
            struct ggml_tensor *bn2_bias_reshaped = ggml_reshape_4d(ctx0, block.bn2_bias, 1, 1, 1, block.bn2_bias->ne[0]);

            cur = ggml_norm(ctx0, cur, epsilon);
            cur = ggml_add(ctx0,
                           ggml_mul(ctx0, ggml_repeat(ctx0, bn2_weight_reshaped, cur), cur),
                           ggml_repeat(ctx0, bn2_bias_reshaped, cur));
            ggml_set_name(cur, (block_name + ".bn2").c_str());

            // Downsample if needed
            if (block.has_downsample)
            {
                identity = ggml_conv_2d(ctx0, block.downsample_conv_weight, identity, block_stride, block_stride, 0, 0, 1, 1);
                ggml_set_name(identity, (block_name + ".downsample.0").c_str());

                struct ggml_tensor *downsample_bn_weight_reshaped = ggml_reshape_4d(ctx0, block.downsample_bn_weight, 1, 1, 1, block.downsample_bn_weight->ne[0]);
                struct ggml_tensor *downsample_bn_bias_reshaped = ggml_reshape_4d(ctx0, block.downsample_bn_bias, 1, 1, 1, block.downsample_bn_bias->ne[0]);

                identity = ggml_norm(ctx0, identity, epsilon);
                identity = ggml_add(ctx0,
                                    ggml_mul(ctx0, ggml_repeat(ctx0, downsample_bn_weight_reshaped, identity), identity),
                                    ggml_repeat(ctx0, downsample_bn_bias_reshaped, identity));
                ggml_set_name(identity, (block_name + ".downsample.1").c_str());
            }

            // Add residual
            cur = ggml_add(ctx0, cur, identity);
            ggml_set_name(cur, (block_name + ".add").c_str());

            // ReLU
            cur = ggml_relu(ctx0, cur);
            ggml_set_name(cur, (block_name + ".relu2").c_str());
        }
    };

    // Process each layer
    process_layer(model.layer1, "layer1", 1); // stride 1
    process_layer(model.layer2, "layer2", 2); // stride 2
    process_layer(model.layer3, "layer3", 2); // stride 2
    process_layer(model.layer4, "layer4", 2); // stride 2

    // Average Pooling
    cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_AVG, 7, 7, 1, 1, 0, 0);
    ggml_set_name(cur, "avgpool");

    // Flatten
    cur = ggml_reshape_2d(ctx0, cur, cur->ne[1], -1);
    ggml_set_name(cur, "flatten");

    // Fully connected layer
    cur = ggml_mul_mat(ctx0, model.fc_weight, cur);
    ggml_set_name(cur, "fc");
    cur = ggml_add(ctx0, cur, ggml_repeat(ctx0, model.fc_bias, cur));
    ggml_set_name(cur, "fc_add");

    *result_ptr = cur;

    // Build the computation graph
    ggml_build_forward_expand(gf, cur);

    return gf;
}

// Run the computation graph
void compute_graph(
    struct ggml_cgraph *gf,
    struct ggml_context *ctx0,
    const int n_threads,
    const char *fname_cgraph)
{
    // Execute the computation graph
    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

    if (fname_cgraph)
    {
        ggml_graph_export(gf, fname_cgraph);
        fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__, fname_cgraph);
    }
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    ggml_time_init();

    resnet_model model;

    {
        const int64_t t_start_us = ggml_time_us();
        if (!init_model(model))
        {
            fprintf(stderr, "%s: failed to initialize model\n", __func__);
            return 1;
        }

        const int64_t t_init_us = ggml_time_us() - t_start_us;
        fprintf(stdout, "%s: initialized model in %8.2f ms\n", __func__, t_init_us / 1000.0f);
    }

    // Prepare input data (dummy data for now)
    // Input shape: (batch_size=1, channels=3, height=224, width=224)
    // For simplicity, let's fill it with random values
    size_t input_size = 1 * 3 * 224 * 224;
    std::vector<float> input_data(input_size);
    for (size_t i = 0; i < input_size; ++i)
    {
        input_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Build the computation graph
    struct ggml_tensor *input_tensor = NULL;
    struct ggml_tensor *result = NULL;
    struct ggml_cgraph *gf = build_graph(model, &input_tensor, &result);

    // Set the input data
    memcpy(input_tensor->data, input_data.data(), ggml_nbytes(input_tensor));

    // Run the computation
    compute_graph(gf, model.ctx, 4, nullptr); // Using 4 threads

    // Retrieve the output
    const float *output_data = (float *)result->data;
    size_t output_size = ggml_nelements(result);

    // Print the output
    fprintf(stdout, "Output vector: [");
    for (size_t i = 0; i < output_size; ++i)
    {
        fprintf(stdout, "%f", output_data[i]);
        if (i < output_size - 1)
            fprintf(stdout, ", ");
    }
    fprintf(stdout, "]\n");

    // Free memory and exit
    ggml_free(model.ctx);
    return 0;
}