from src.inference.model import SimpleCNN
import torch

def test_model_forward_pass():
    model = SimpleCNN()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 2)