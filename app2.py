from flask import Flask, request, jsonify, render_template
import socket
import numpy as np
import cv2
import json
import base64

from custom.essentials import stringToRGB, get_model

'''Get host IP address'''
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

app = Flask(__name__)

# Root URL route
@app.route('/')
def home():
    return render_template('index.html')

# Simple http endpoint
@app.route('/get_name', methods=['GET', 'POST'])
def get_name():
    return 'hello'

# Image upload endpoint
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No image selected'})

    image_bytes = image_file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    prediction = predict_skin_disease(image)

    return jsonify({'prediction': prediction})

def predict_skin_disease(image):
    model_name = 'Model/best_model.h5'
    model = get_model()
    model.load_weights(model_name)

    classes = {
        4: ('nv', 'melanocytic nevi'),
        6: ('mel', 'melanoma'),
        2: ('bkl', 'benign keratosis-like lesions'),
        1: ('bcc', 'basal cell carcinoma'),
        5: ('vasc', 'pyogenic granulomas and hemorrhage'),
        0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),
        3: ('df', 'dermatofibroma')
    }

    img = cv2.resize(image, (28, 28))
    img = img.reshape(1, 28, 28, 3)

    result = model.predict(img)
    max_prob = np.max(result)
    class_ind = np.argmax(result)
    class_name = classes[class_ind][1]
    
    # Create a dictionary to store class probabilities
    class_probs = {classes[i][1]: result[0][i] for i in range(len(classes))}
    
    if max_prob > 0.80:
        class_probs[class_name] = max_prob
    else:
        class_name = 'No Disease'

    return class_name


if __name__ == '__main__':
    app.run(debug=True)

