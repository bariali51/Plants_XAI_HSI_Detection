# check_model.py
import torch
import torch.nn as nn
from torchvision import models
import json

with open('analysis/model_files/class_names.json', 'r') as f:
    class_names = json.load(f)

num_classes = len(class_names)
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
state_dict = torch.load('analysis/model_files/efficientnet_b0_plant_disease.pth', map_location='cpu', weights_only=True)
model.load_state_dict(state_dict)
model.eval()

print(f"Model loaded successfully: EfficientNet-B0 with {num_classes} classes")
print(f"Input: 3x224x224")
print(f"Classifier: {model.classifier}")
print(f"\nClass names sample:")
for i in range(min(5, num_classes)):
    print(f"  {i}: {class_names.get(str(i), 'Unknown')}")