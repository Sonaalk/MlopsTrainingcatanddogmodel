import os
import torch
import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 56 * 56, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_model():
    model = SimpleCNN()

    model_path = "models/cnn_model.pt"

    if os.path.exists(model_path):
        print("Loading trained model...")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        print("⚠️ Model file not found. Starting with untrained model.")

    model.eval()
    return model

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_image(model, image: Image.Image):
    img_tensor = _transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze().tolist()
    return {
        "cat": probs[0],
        "dog": probs[1]
    }