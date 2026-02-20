# classifier/ml_utils.py

import torch
import pickle
import json
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from .models import PlantDiseaseModel

# ==================== مسارات الملفات ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model_files', 'best_model.pth')
ENCODER_PATH = os.path.join(BASE_DIR, 'model_files', 'label_encoder.pkl')
TRANSFORM_PATH = os.path.join(BASE_DIR, 'model_files', 'inference_transform.pkl')
CLASS_NAMES_PATH = os.path.join(BASE_DIR, 'model_files', 'class_names.json')

# ==================== توصيات العلاج ====================
TREATMENT_RECOMMENDATIONS = {
    "Tomato_Bacterial_spot": [
        "Remove and destroy infected plants",
        "Rotate crops (avoid planting tomatoes in the same location for 2-3 years)",
        "Use copper-based fungicides",
        "Ensure proper spacing between plants for good air circulation"
    ],
    "Tomato_Early_blight": [
        "Remove infected leaves immediately",
        "Apply fungicides containing chlorothalonil or copper",
        "Mulch around the base of plants",
        "Water at soil level rather than on foliage"
    ],
    "Tomato_Late_blight": [
        "Remove and destroy infected plants",
        "Apply fungicides proactively before symptoms appear",
        "Improve air circulation around plants",
        "Avoid overhead irrigation"
    ],
    "Tomato_Leaf_Mold": [  # ✅ هذا هو المرض الذي تم تشخيصه
        "Increase spacing between plants to improve air circulation",
        "Apply fungicides containing chlorothalonil or copper",
        "Remove infected leaves",
        "Keep foliage dry by watering at the base"
    ],
    "Tomato_Septoria_leaf_spot": [
        "Remove infected leaves",
        "Apply fungicides containing chlorothalonil or copper",
        "Rotate crops",
        "Mulch around plants to prevent spores splashing from soil"
    ],
    "Tomato_Spider_mites_Two_spotted_spider_mite": [
        "Spray plants with strong streams of water to dislodge mites",
        "Apply insecticidal soap or neem oil",
        "Introduce predatory mites",
        "Increase humidity around plants"
    ],
    "Tomato__Target_Spot": [
        "Remove infected plant debris",
        "Apply fungicides",
        "Improve air circulation",
        "Avoid overhead watering"
    ],
    "Tomato__Tomato_YellowLeaf__Curl_Virus": [
        "No cure available - remove and destroy infected plants",
        "Control whitefly populations (vectors)",
        "Use reflective mulches to repel whiteflies",
        "Plant resistant varieties"
    ],
    "Tomato__Tomato_mosaic_virus": [
        "No cure available - remove and destroy infected plants",
        "Wash hands and tools after handling infected plants",
        "Control aphid populations (vectors)",
        "Plant resistant varieties"
    ],
    "Potato___Early_blight": [
        "Remove infected leaves",
        "Apply fungicides containing chlorothalonil",
        "Maintain good soil fertility",
        "Ensure proper hilling to protect tubers"
    ],
    "Potato___Late_blight": [
        "Apply fungicides preventatively",
        "Remove volunteer potato plants",
        "Harvest tubers during dry weather",
        "Ensure proper storage conditions for harvested potatoes"
    ],
    "Pepper__bell___Bacterial_spot": [
        "Remove infected plant debris",
        "Rotate crops",
        "Apply copper-based sprays",
        "Use disease-free seeds"
    ],
}

DEFAULT_HEALTHY_PRACTICES = [
    "Maintain proper watering schedule",
    "Ensure adequate sunlight",
    "Fertilize appropriately for plant type",
    "Monitor regularly for signs of disease"
]


# ==================== كلاس التصنيف ====================
class DiseaseClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label_encoder = None
        self.transform = None
        self.class_names = None
        self.load_resources()

    def load_resources(self):
        print(f"🔧 Loading model on device: {self.device}")

        with open(CLASS_NAMES_PATH, 'r') as f:
            self.class_names = json.load(f)

        num_classes = len(self.class_names)
        self.model = PlantDiseaseModel(num_classes=num_classes)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        with open(ENCODER_PATH, 'rb') as f:
            self.label_encoder = pickle.load(f)
        with open(TRANSFORM_PATH, 'rb') as f:
            self.transform = pickle.load(f)

        print(f"✅ Model loaded successfully with {num_classes} classes")

    def predict(self, image_file):
        img = Image.open(image_file).convert("RGB")
        image_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)

            pred_class = self.label_encoder.inverse_transform([predicted_idx.item()])[0]
            conf_score = float(confidence.item()) * 100

            lookup_key = pred_class.replace(' ', '_')
            recommendations = TREATMENT_RECOMMENDATIONS.get(
                lookup_key,
                TREATMENT_RECOMMENDATIONS.get(pred_class, ["No specific recommendations"])
            )

            if "healthy" in pred_class.lower():
                recommendations = DEFAULT_HEALTHY_PRACTICES

            return {
                "disease": pred_class.replace('_', ' '),
                "confidence": f"{conf_score:.2f}%",
                "recommendations": recommendations,
                "is_healthy": "healthy" in pred_class.lower(),
                "raw_class": pred_class,
                "probabilities": probabilities.cpu().numpy()
            }

    def apply_gradcam(self, image_file, layer_name='conv_block5'):
        """
        توليد خريطة Grad-CAM الحرارية
        Returns: dict with PIL Images (original, heatmap, superimposed)
        """
        self.model.eval()
        activations = None
        gradients = None

        def forward_hook(module, input, output):
            nonlocal activations
            activations = output.detach()

        # ✅ استخدام register_full_backward_hook لإصلاح الـ Warning
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()

        if layer_name == 'conv_block5':
            target_layer = self.model.conv_block5[0]
        elif layer_name == 'conv_block4':
            target_layer = self.model.conv_block4[0]
        else:
            target_layer = self.model.conv_block3[0]

        forward_handle = target_layer.register_forward_hook(forward_hook)
        # ✅ تحديث لـ PyTorch الحديث
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            image_file.seek(0)
            original_img = Image.open(image_file).convert('RGB')
            original_np = np.array(original_img)

            image_file.seek(0)
            img_for_transform = Image.open(image_file).convert('RGB')
            image_tensor = self.transform(img_for_transform).unsqueeze(0).to(self.device)

            output = self.model(image_tensor)
            pred_idx = output.argmax(dim=1).item()

            self.model.zero_grad()
            output[:, pred_idx].backward()

            if activations is not None and gradients is not None:
                pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

                for i in range(activations.size(1)):
                    activations[:, i, :, :] *= pooled_gradients[i]

                heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()
                heatmap = np.maximum(heatmap, 0)

                if np.max(heatmap) > 0:
                    heatmap = heatmap / np.max(heatmap)

                heatmap = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
                heatmap = np.uint8(255 * heatmap)
                heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                original_bgr = cv2.cvtColor(original_np, cv2.COLOR_RGB2BGR)
                superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

                heatmap_pil = Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
                superimposed_pil = Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))

                return {
                    "original": original_img,
                    "heatmap": heatmap_pil,
                    "superimposed": superimposed_pil,
                    "predicted_class": self.label_encoder.inverse_transform([pred_idx])[0]
                }
            else:
                print("⚠️ Could not generate activations or gradients")
                return None

        except Exception as e:
            print(f"❌ Grad-CAM Error: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            forward_handle.remove()
            backward_handle.remove()


classifier_instance = DiseaseClassifier()