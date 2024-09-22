import torch

# Ensure GPU is available and supports Tensor Cores
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set the precision for matrix multiplication to use Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Define matrix dimensions that are multiples of 8 for FP16
M = 256  # Number of rows in matrix A
K = 256  # Number of columns in matrix A and rows in matrix B
N = 256  # Number of columns in matrix B

# Create tensors with FP16 precision
A = torch.randn(M, K, device=device, dtype=torch.half)
B = torch.randn(K, N, device=device, dtype=torch.half)
C = torch.randn(M, N, device=device, dtype=torch.half)

# Perform matrix multiplication using Tensor Cores
D = torch.matmul(A, B) + C

# Print the result
print(D.shape)

# Optionally, benchmark the operation
import time
start_time = time.time()
for _ in range(100):
    D = torch.matmul(A, B) + C
torch.cuda.synchronize()
end_time = time.time()
print(f"Time taken for 100 iterations: {1000*(end_time - start_time)} ms")