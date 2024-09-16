import torch
import torch.nn as nn
import os

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the neural network with 3 layers and ReLU activation
class ThreeLayerNN(nn.Module):
    def __init__(self, input_size=5, hidden1_size=10, hidden2_size=8, output_size=3):
        super(ThreeLayerNN, self).__init__()
        self.nexa_fc1 = nn.Linear(input_size, hidden1_size)
        self.nexa_fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.nexa_fc3 = nn.Linear(hidden2_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.nexa_fc1(x)
        x = self.relu(x)
        x = self.nexa_fc2(x)
        x = self.relu(x)
        x = self.nexa_fc3(x)
        return x

# Create the model and move it to the device
model = ThreeLayerNN().to(device)

# Save the model
# if model/three_layer_nn.pth already exists, load it. Need to judge if it exists first.
if os.path.exists('model/three_layer_nn.pth'):
    model.load_state_dict(torch.load('model/three_layer_nn.pth'))
else:
    torch.save(model.state_dict(), 'model/three_layer_nn.pth')

# Sample input
sample_input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32).to(device)

# Reshape the input to match the expected shape (batch_size, input_size)
sample_input = sample_input.view(1, -1)

# Forward pass
output = model(sample_input)

# Now, do the quantization to fp16, and run it again to get the output
model = model.half()
half_sample_input = sample_input.half()
half_output = model(half_sample_input)

print(model.nexa_fc1.bias)

print("Model architecture:")
print(model)


print("\nSample input:", sample_input)
print("Output:", output)

print("\nSample input after quantization:", half_sample_input)
print("Output after quantization:", half_output)