import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import json
import os
import random
import cv2
import matplotlib.pyplot as plt

with open('analysis/model_files/class_names.json', 'r') as f:
    class_names = json.load(f)

num_classes = len(class_names)
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
state_dict = torch.load('analysis/model_files/efficientnet_b0_plant_disease.pth', map_location='cpu', weights_only=True)
model.load_state_dict(state_dict)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def apply_gradcam(model, img_path, transform, class_names_map, device, layer_name='features'):
    try:
        import cv2
    except ImportError:
        print("OpenCV (cv2) is required for Grad-CAM visualization. Please install it with: !pip install opencv-python")
        return

    model.eval()

    # Hook for the selected layer
    activations = None
    gradients = None

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output.detach()

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0].detach()

    # Register hooks (Adapted for EfficientNet-B0)
    target_layer = model.features[-1]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    
    if hasattr(target_layer, 'register_full_backward_hook'):
        backward_handle = target_layer.register_full_backward_hook(backward_hook)
    else:
        backward_handle = target_layer.register_backward_hook(backward_hook)

    try:
        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Forward pass
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        
        pred_class = class_names_map.get(str(pred_idx), "Unknown")

        # Backward pass for the predicted class
        model.zero_grad()
        output[:, pred_idx].backward()

        # Generate Grad-CAM
        if activations is not None and gradients is not None:
            # Pool gradients across the channels
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

            # Weight activation maps by gradients
            for i in range(activations.size(1)):
                activations[:, i, :, :] *= pooled_gradients[i]

            # Average over channels
            heatmap = torch.mean(activations, dim=1).squeeze().cpu().numpy()

            # ReLU on heatmap
            heatmap = np.maximum(heatmap, 0)

            # Normalize heatmap
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)

            # Resize heatmap to original image size
            original_img = np.array(img)
            heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

            # Apply colormap to heatmap
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            # Superimpose heatmap on original image
            superimposed = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

            # Create figure with original and heatmap
            plt.figure(figsize=(15, 5))

            # Plot original image
            plt.subplot(1, 3, 1)
            plt.imshow(original_img)
            plt.title("Original Image", fontsize=14)
            plt.axis('off')

            # Plot heatmap
            plt.subplot(1, 3, 2)
            plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
            plt.title("Grad-CAM Heatmap", fontsize=14)
            plt.axis('off')

            # Plot superimposed
            plt.subplot(1, 3, 3)
            plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
            plt.title(f"Prediction: {pred_class}", fontsize=14)
            plt.axis('off')

            plt.tight_layout()
            plt.show()
        else:
            print("Could not generate activations or gradients")
    finally:
        # Always remove hooks to prevent memory leaks
        forward_handle.remove()
        backward_handle.remove()

def generate_gradcam_visualizations(num_samples=2, data_dir=r'D:\Dataset\PlantVillage'):
    print("\n🔍 Generating Grad-CAM visualizations...")
    sample_images = []
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Testing with single dummy image.")
        img_path = 'dummy.jpg'
        if not os.path.exists(img_path):
            Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255)).save(img_path)
        apply_gradcam(model, img_path, transform, class_names, 'cpu')
        return

    for disease_folder in os.listdir(data_dir):
        disease_folder_path = os.path.join(data_dir, disease_folder)
        if not os.path.isdir(disease_folder_path):
            continue
            
        img_files = [f for f in os.listdir(disease_folder_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if img_files:
            selected_img = os.path.join(disease_folder_path, random.choice(img_files))
            sample_images.append((selected_img, disease_folder))
    
    # Apply GradCAM to a couple of sample images
    if sample_images:
        samples_to_visualize = random.sample(sample_images, min(num_samples, len(sample_images)))
        for i, (img_path, true_label) in enumerate(samples_to_visualize):
            print(f"\nVisualizing sample {i+1} - {true_label}...")
            apply_gradcam(model, img_path, transform, class_names, 'cpu')
    else:
        print("No sample images found.")

if __name__ == "__main__":
    # Test on single specific image if it exists
    single_img_path = r'D:\Dataset\PlantVillage\Tomato__Tomato_YellowLeaf__Curl_Virus\00139ae8-d881-4edb-925f-46584b0bd68c___YLCV_NREC 2944.JPG'
    if os.path.exists(single_img_path):
        print(f"Applying Grad-CAM to {single_img_path}")
        apply_gradcam(model, single_img_path, transform, class_names, 'cpu')
    
    # Generate visualizations for 2 random samples
    generate_gradcam_visualizations(num_samples=2)
