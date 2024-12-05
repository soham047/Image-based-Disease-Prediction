from django.shortcuts import render
from img_rec.forms import ImageUploadForm  # Change to absolute import
from django.conf import settings
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the pre-trained CNN model globally
model1 = None
model2 = None
model3 = None


model_path = os.path.join(settings.BASE_DIR, 'model.h5')
model_path2 = os.path.join(settings.BASE_DIR, 'model_mri.h5')
model_path3 = os.path.join(settings.BASE_DIR, 'model_xray.h5')

try:
    # Load the model globally so it's accessible in the view
    model1 = load_model(model_path)
    model2 = load_model(model_path2)
    model3 = load_model(model_path3)
        
    print("Model loaded successfully!")
except OSError as e:
    print(f"Error loading model: {e}")

d = {'Actinic keratosis': 0,
 'Atopic Dermatitis': 1,
 'Benign keratosis': 2,
 'Dermatofibroma': 3,
 'Melanocytic nevus': 4,
 'Melanoma': 5,
 'Squamous cell carcinoma': 6,
 'Tinea Ringworm Candidiasis': 7,
 'Vascular lesion': 8}

def recognize(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
            form.save()
            image_url = form.instance.image.url
            
            # Get the file path of the uploaded image
            image_path = form.instance.image.path
            
            # Get the selected disease from the form
            disease = request.POST.get('disease')
            print(f"Selected disease: {disease}")  # Debugging

            # Preprocess the image for the CNN model
            img = image.load_img(image_path, target_size=(180, 180))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0  # Normalize the image
            
            # Initialize prediction as None in case none of the conditions execute
            prediction = None
            
            if disease == "skin":
                prediction = model1.predict(img_array)
                print("Skin model used.")  # Debugging
            elif disease == 'xray':
                prediction = model2.predict(img_array)
                print("X-Ray model used.")  # Debugging
            elif disease == 'tumor':
                prediction = model3.predict(img_array)
                print("Tumor model used.")  # Debugging
            
            if prediction is not None:
                # Make a prediction using the CNN model
                predicted_class = np.argmax(prediction)
                predicted_label = list(d.keys())[list(d.values()).index(predicted_class)]
            else:
                predicted_label = "No prediction made."
            
            # Render the result along with the uploaded image and prediction
            return render(request, 'img_rec/index.html', {
                'form': form,
                'image_url': image_url,
                'predicted_label': predicted_label,
            })
    else:
        form = ImageUploadForm()
    
    return render(request, 'img_rec/index.html', {'form': form})
