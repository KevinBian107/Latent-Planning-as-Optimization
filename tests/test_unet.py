import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.getcwd())
from src.models.unet1d import Unet1D
import torch
def test_unet1d():
    # Hyperparameters
    batch_size = 4
    channels = 1
    length = 64
    dim = 32  # hidden dimension inside Unet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    model = Unet1D(dim=dim, channels=channels, dim_mults=(1, 2, 4)).to(device)

    # Generate random input
    x = torch.randn(batch_size, channels, length).to(device)

    # Forward pass
    output = model(x)

    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Check shape consistency
    assert output.shape == x.shape, f"Output shape {output.shape} does not match input shape {x.shape}"

    print("✅ Test passed: Input and output shapes match.")

if __name__ == "__main__":
    test_unet1d()