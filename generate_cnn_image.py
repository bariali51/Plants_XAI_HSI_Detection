import tensorflow as tf
from tensorflow.keras.utils import plot_model

model_path = r"c:\Users\bari\Desktop\Project moklid\projer-main\Plants_XAI_HSI_Detection-main\analysis\model_files\plant_disease_prediction_model.h5"

try:
    model = tf.keras.models.load_model(model_path)
    output_path = r"c:\Users\bari\Desktop\Project moklid\projer-main\Plants_XAI_HSI_Detection-main\cnn_architecture.png"
    plot_model(model, to_file=output_path, show_shapes=True, show_layer_names=True)
    print(f"Image successfully saved to {output_path}")
except Exception as e:
    print(f"Error generating image: {e}")
