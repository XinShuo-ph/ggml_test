# Import required libraries
import gguf
import torch
import numpy as np
from resnet import resnet18  # Assuming the ResNet model code is in resnet.py
import sys
import os

# Check if CUDA is available (though for GGUF conversion, we use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Ensure tensors are on CPU for GGUF

def print_ggml_layer_info(gguf_model_name):
    gguf_data = gguf.GGUFReader(gguf_model_name)
    
    print("\nGGML Layer Information:")
    print(f"{'Layer':<40} {'Shape':<20} {'Type':<15} {'Param #':<10}")
    print("-" * 85)
    
    total_params = 0
    for tensor in gguf_data.tensors:
        num_params = np.prod(tensor.shape)
        total_params += num_params
        print(f"{tensor.name:<40} {str(tensor.shape):<20} {tensor.tensor_type:<15} {num_params:<10}")
    
    print("-" * 85)
    print(f"Total params: {total_params}")

def convert(model_path, gguf_model_name):
    # Load the PyTorch model
    model = resnet18().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Initialize GGUF writer
    gguf_writer = gguf.GGUFWriter(gguf_model_name, "resnet18")

    # Convert and add tensors for each parameter
    for param_name, param in model.state_dict().items():
        data = param.cpu().numpy().astype(np.float32)
        gguf_writer.add_tensor(param_name, data)

    # Write the GGUF file
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"\nModel converted and saved to '{gguf_model_name}'")

    # Optionally, print GGML layer information
    print_ggml_layer_info(gguf_model_name)

if __name__ == '__main__':
    model_path = f'model/resnet.pth'
    gguf_model_name = f'model/resnet.gguf'

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist.")
        sys.exit(1)

    convert(model_path, gguf_model_name)
