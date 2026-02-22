import os
import cv2
from sklearn.model_selection import train_test_split

IMG_SIZE = 224
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def preprocess_and_split():
    images = []
    labels = []

    for label, cls in enumerate(["cats", "dogs"]):
        cls_path = os.path.join(RAW_DIR, cls)
        if not os.path.exists(cls_path):
            raise FileNotFoundError(f"Missing folder: {cls_path}")

        for img in os.listdir(cls_path):
            images.append(os.path.join(cls_path, img))
            labels.append(label)

    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    for split, (X, y) in splits.items():
        for img_path, label in zip(X, y):
            cls = "cats" if label == 0 else "dogs"
            dest = os.path.join(PROCESSED_DIR, split, cls)
            os.makedirs(dest, exist_ok=True)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(os.path.join(dest, os.path.basename(img_path)), img)

if __name__ == "__main__":
    preprocess_and_split()