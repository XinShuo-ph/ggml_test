import sys
import numpy as np
import gguf
import torch
import os

if len(sys.argv) != 2:
    print("Usage: convert-pth-to-ggml.py model_path")
    sys.exit(1)

state_dict_file = sys.argv[1]
fname_out = "model/three_layer_nn-ggml-model-f32.gguf"

# Load the model
model = torch.load(state_dict_file, map_location=torch.device('cpu'))

# Initialize GGUF writer
gguf_writer = gguf.GGUFWriter(fname_out, "three-layer-nn")

# Convert and add tensors for each layer
for layer_name in ['nexa_fc1', 'nexa_fc2', 'nexa_fc3']:
    # Weights
    weights = model[f"{layer_name}.weight"].data.numpy()
    weights = weights.astype(np.float16)
    gguf_writer.add_tensor(f"{layer_name}_weights", weights, raw_shape=weights.shape)

    # Biases
    bias = model[f"{layer_name}.bias"].data.numpy()
    gguf_writer.add_tensor(f"{layer_name}_bias", bias)

# Write the GGUF file
gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()
gguf_writer.close()

print(f"Model converted and saved to '{fname_out}'")