# check_model.py
import tensorflow as tf
model = tf.keras.models.load_model('analysis/model_files/plant_disease_prediction_model.h5')
print("✅ Model input shape:", model.input_shape)
print("✅ Model output shape:", model.output_shape)
print("✅ Model summary:")
model.summary()

for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name}")
    print(f"  Type: {type(layer)}")
    print(f"  Output shape: {layer.output_shape}")
    print()