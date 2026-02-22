import requests
import os

SERVICE_URL = "http://localhost:30007/predict"

# Simulated evaluation data
samples = [
    ("data/sample_eval/cat1.jpg", "cat"),
    ("data/sample_eval/dog1.jpg", "dog"),
]

correct = 0

for img_path, true_label in samples:
    with open(img_path, "rb") as f:
        r = requests.post(SERVICE_URL, files={"file": f})

    pred = max(r.json()["prediction"], key=r.json()["prediction"].get)

    print(f"Image: {img_path}, Predicted: {pred}, True: {true_label}")

    if pred == true_label:
        correct += 1

accuracy = correct / len(samples)
print(f"Post-deployment accuracy: {accuracy:.2f}")