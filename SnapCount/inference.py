import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# =========================
# DEVICE
# =========================
DEVICE = torch.device("cpu")

# =========================
# LOAD MODEL ARCHITECTURE
# =========================
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# =========================
# LOAD TRAINED MODEL
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pth")

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# TRANSFORM (same as training)
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# REQUIRED FUNCTION
# =========================
def predict(image_path):
    """
    Returns:
    label: int (0 = Male, 1 = Female)
    confidence: float (0–1)
    """

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)


    raw_label = predicted.item()
    confidence = confidence.item()

    # flip 0<->1
    label = 1 - raw_label

    return label, confidence


# =========================
# MULTI IMAGE TEST
# =========================
if __name__ == "__main__":

    sample_folder = "samples"

    if not os.path.exists(sample_folder):
        print("Folder 'samples' not found.")
    else:
        for file in os.listdir(sample_folder):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(sample_folder, file)
                label, confidence = predict(path)

                gender = "Male" if label == 0 else "Female"

                print(f"File: {file}")
                print(f"Prediction: {gender}")
                print(f"Confidence: {confidence:.4f}")
                print("-" * 40)
