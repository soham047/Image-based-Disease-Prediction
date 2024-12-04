from django.shortcuts import render
from django.core.files.storage import default_storage
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

CLASS_NAMES = {
    0: "Apple Scab",
    1: "Apple Black Rot",
    2: "Apple Cedar Rust",
    3: "Apple Healthy",
    4: "Blueberry Healthy",
    5: "Cherry Powdery Mildew",
    6: "Cherry Healthy",
    7: "Corn Cercospora Leaf Spot (Gray Leaf Spot)",
    8: "Corn Common Rust",
    9: "Corn Northern Leaf Blight",
    10: "Corn Healthy",
    11: "Grape Black Rot",
    12: "Grape Esca (Black Measles)",
    13: "Grape Leaf Blight",
    14: "Grape Healthy",
    15: "Citrus Huanglongbing (Greening)",
    16: "Peach Bacterial Spot",
    17: "Peach Healthy",
    18: "Pepper Bacterial Spot",
    19: "Pepper Healthy",
    20: "Potato Early Blight",
    21: "Potato Late Blight",
    22: "Potato Healthy",
    23: "Raspberry Healthy",
    24: "Soybean Healthy",
    25: "Squash Powdery Mildew",
    26: "Strawberry Leaf Scorch",
    27: "Strawberry Healthy",
    28: "Tomato Bacterial Spot",
    29: "Tomato Early Blight",
    30: "Tomato Late Blight",
    31: "Tomato Leaf Mold",
    32: "Tomato Septoria Leaf Spot",
    33: "Tomato Spider Mite Damage",
    34: "Tomato Target Spot",
    35: "Tomato Yellow Leaf Curl Virus",
    36: "Tomato Mosaic Virus",
    37: "Tomato Healthy"
}


def load_model():
    model = tf.keras.models.load_model('model/plant_disease_prediction_model.h5')
    return model

model = load_model()

def predict_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    disease_name = CLASS_NAMES[predicted_index]  # Retrieve disease name
    return disease_name, predictions

def index(request):
    context = {}
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']
        file_path = default_storage.save(f'uploaded_images/{uploaded_file.name}', uploaded_file)
        disease_name, pred_prob = predict_image(f'media/{file_path}')
        context = {
            'prediction': f"Detected Disease: {disease_name}",
            'confidence': f"Confidence: {pred_prob.max() * 100:.2f}"
        }
    return render(request, 'index.html', context)
