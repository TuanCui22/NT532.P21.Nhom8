from flask import Flask, request
import os, cv2, time, base64
import numpy as np
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, db

app = Flask(__name__)

# Firebase config
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://iot-hien-dai-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# Load model
model = tf.keras.models.load_model("asl_model")
CONFIDENCE_THRESHOLD = 0.9
LABELS = {i: chr(65 + i) for i in range(26)}

def preprocess(frame):
    frame = cv2.resize(frame, (224, 224))
    return np.expand_dims(frame.astype('float32') / 255.0, axis=0)

def upload(label, frame):
    try:
        timestamp = str(int(time.time() * 1000))
        folder = os.path.join("img", timestamp)
        os.makedirs(folder, exist_ok=True)
        img_path = os.path.join(folder, "image.jpg")
        cv2.imwrite(img_path, frame)

        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        db.reference("images").child(timestamp).set({
            "image": f"data:image/jpeg;base64,{img_b64}",
            "text": label,
            "timestamp": int(timestamp)
        })
        print(f"📤 Uploaded: {label} ({timestamp})")
    except Exception as e:
        print(f"❌ Firebase error: {e}")

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        file = request.files['image']
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        input_tensor = preprocess(frame)
        preds = model.predict(input_tensor, verbose=0)
        pred_class = np.argmax(preds[0])
        confidence = preds[0][pred_class]
        label = LABELS.get(pred_class, "-")

        if confidence > CONFIDENCE_THRESHOLD:
            upload(label, frame)
            return "OK", 200
        else:
            return "Low confidence", 204
    except Exception as e:
        print(f"❌ Server error: {e}")
        return "Error", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
