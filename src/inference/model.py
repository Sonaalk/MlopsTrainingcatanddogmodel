import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

IMG_SIZE = 224
CLASS_NAMES = ["Cat", "Dog"]

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 54 * 54, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load("models/cnn_model.pt", map_location="cpu"))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

def predict_image(model, image: Image.Image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
    return {
        "label": CLASS_NAMES[pred_idx],
        "probability": float(probs[pred_idx])
    }