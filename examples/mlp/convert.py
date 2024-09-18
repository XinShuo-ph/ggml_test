# Refer to
# https://github.com/ggerganov/ggml/blob/master/examples/magika/convert.py
import gguf
import torch
import numpy as np
# Import the MLP class from model.py
from model import MLP

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # must specify cpu or cpu-only ggml cannot work


def print_ggml_layer_info(gguf_model_name):
    gguf_data = gguf.GGUFReader(gguf_model_name)
    
    print("\nGGML Layer Information:")
    print(f"{'Layer':<30} {'Shape':<20} {'Type':<15} {'Param #':<10}")
    print("-" * 75)
    
    total_params = 0
    for tensor in gguf_data.tensors:
        num_params = np.prod(tensor.shape)
        total_params += num_params
        print(f"{tensor.name:<30} {str(tensor.shape):<20} {tensor.tensor_type:<15} {num_params:<10}")
    
    print("-" * 75)
    print(f"Total params: {total_params}")


def convert(model_path, gguf_model_name):
    # Load the PyTorch model
    model = MLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize GGUF writer
    gguf_writer = gguf.GGUFWriter(gguf_model_name, "mlp")

    # Convert and add tensors for each layer
    for layer_name in ['fc1', 'fc2']:
        # Weights
        weights = getattr(model, layer_name).weight.data.cpu().numpy() # cast back to CPU, so GGUF does not have difference between CPU and GPU
        weights = weights.astype(np.float32)
        gguf_writer.add_tensor(f"{layer_name}.weight", weights)

        # Biases
        bias = getattr(model, layer_name).bias.data.cpu().numpy()
        bias = bias.astype(np.float32)
        gguf_writer.add_tensor(f"{layer_name}.bias", bias)

    # Write the GGUF file
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"\nModel converted and saved to '{gguf_model_name}'")

    # Print GGML layer information
    print_ggml_layer_info(gguf_model_name)


if __name__ == '__main__':
    model_path = f"model/two_layer_mlp.pth"
    gguf_model_name = f"model/mlp.gguf"
    convert(model_path, gguf_model_name)