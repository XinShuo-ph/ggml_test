import torch
import torch.nn as nn
import os


# Check if CUDA is available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

# Define MLP Layer
class MLP(nn.Module):
    def __init__(self, input_size=5, hidden_size=10, output_size=3):
        super(MLP, self).__init__()
        # Assign randomly initialized values
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def print_model_architecture_and_weights(model):
    print("Model Architecture:")
    print(model)
    print("\nModel Parameters:")
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"\n{name} shape: {param.shape}")
            print(param.data)
        elif 'bias' in name:
            print(f"\n{name} shape: {param.shape}")
            print(param.data)


def inference(model):
    # Sample input
    sample_input = torch.tensor([0.5, 0.4, 0.3, 0.2, 0.1], dtype=torch.float32).to(device)
    # Reshape the input to match the expected shape (batch_size, input_size)
    sample_input = sample_input.view(1, -1)
    
    # Forward pass
    output = model(sample_input)
    print(f"Model output:\n{output}")


if __name__ == '__main__':
    # Create the model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)

    # Create the model and move it to the device
    model = MLP().to(device)

    # Load or save the model
    model_path = 'model/two_layer_mlp.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

    # Print model architecture and weights
    print_model_architecture_and_weights(model)

    # Run inference
    import time
    start = time.time()
    inference(model)
    print(f"Inference time: {time.time() - start:.4f} seconds")