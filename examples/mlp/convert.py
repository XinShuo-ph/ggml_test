# Refer to
# https://github.com/ggerganov/ggml/blob/master/examples/magika/convert.py
import gguf
import torch
# Import the MLP class from model.py
from model import MLP
    
# Check if CUDA is available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

def convert(model_path, gguf_model_name):
    # Load the model
    model = MLP()  # Create an instance of your MLP class
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode

    # Initialize GGUF writer
    gguf_writer = gguf.GGUFWriter(gguf_model_name, "mlp")

    # Convert and add tensors for each layer
    for name, param in model.named_parameters():
        print(f"  [{name}] {param.shape} {param.dtype}")
        param_data = param.detach().cpu().numpy()
        if 'weight' in name:
            param_data = param_data.T  # Transpose weights
        gguf_writer.add_tensor(name, param_data)

    # Write the GGUF file
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"Model converted and saved to '{gguf_model_name}'")

if __name__ == '__main__':
    model_path = 'model/two_layer_mlp.pth'
    gguf_model_name = "mlp.gguf"
    convert(model_path, gguf_model_name)