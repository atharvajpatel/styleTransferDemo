import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Check which device a tensor is on (optional)
tensor = torch.rand(3, 3).to("cuda" if torch.cuda.is_available() else "cpu")
print(f"Tensor is on device: {tensor.device}")