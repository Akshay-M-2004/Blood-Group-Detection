import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from werkzeug.utils import secure_filename

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model once
model = load_model('model_blood_group_detection_resnet.h5',compile=False)

# Blood group labels
labels = {
    0: 'A+', 1: 'A-', 2: 'AB+', 3: 'AB-', 
    4: 'B+', 5: 'B-', 6: 'O+', 7: 'O-'
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No file selected')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load and preprocess image
            img = image.load_img(filepath, target_size=(256, 256))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # Predict
            result = model.predict(x)
            predicted_class = np.argmax(result)
            predicted_label = labels[predicted_class]
            confidence = float(result[0][predicted_class]) * 100

            return render_template('index.html',
                                   prediction=predicted_label,
                                   confidence=f"{confidence:.2f}%",
                                   image_path=filepath)

        return render_template('index.html', error='Invalid file type. Please upload .png, .jpg, , .bmp, or .jpeg.')

    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
