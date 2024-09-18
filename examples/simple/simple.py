import torch

# Create two matrices
matrix1 = torch.tensor([
  [2, 8],
  [5, 1],
  [4, 2],
  [8, 6],
])
matrix2 = torch.tensor([
  [10, 5],
  [9, 9],
  [5, 4],
])

# Perform matrix multiplication
result = torch.matmul(matrix1, matrix2.T)
print(result.T)