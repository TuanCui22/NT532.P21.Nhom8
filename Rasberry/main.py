import os
import cv2
import time
import base64
import numpy as np
import tensorflow as tf
import firebase_admin
from firebase_admin import credentials, db

# Cấu hình Firebase
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://iot-hien-dai-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# Tải model TensorFlow
MODEL_PATH = 'asl_model'
CONFIDENCE_THRESHOLD = 0.9

try:
    asl_model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Mô hình đã được tải.")
except Exception as e:
    raise RuntimeError(f"Lỗi khi tải mô hình: {str(e)}")

# Nhãn ký hiệu
LABELS = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
    5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
    15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
    20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
    25: "Z"
}

# Xử lý ảnh đầu vào
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)

# Upload Firebase
def upload_to_firebase(folder, image_path, label):
    try:
        ref = db.reference('images')
        with open(image_path, "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        ref.child(folder).set({
            "image": f"data:image/jpeg;base64,{image_base64}",
            "text": label,
            "timestamp": int(folder)
        })
        print(f"📤 Đã upload Firebase: {folder} - {label}")
    except Exception as e:
        print(f"❌ Lỗi upload Firebase: {str(e)}")

# Tạo thư mục lưu ảnh
BASE_FOLDER = 'img'
os.makedirs(BASE_FOLDER, exist_ok=True)

# Bắt đầu camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được camera.")
    exit()

print("📷 Đang theo dõi... Nhấn Q hoặc Ctrl+C để dừng.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Không lấy được khung hình.")
            continue

        input_data = preprocess_frame(frame)
        preds = asl_model.predict(input_data)
        predicted_class = np.argmax(preds[0])
        confidence = preds[0][predicted_class]
        label = LABELS.get(predicted_class, "KHÔNG_XÁC_ĐỊNH")

        if confidence >= CONFIDENCE_THRESHOLD:
            timestamp = str(int(time.time() * 1000))
            save_folder = os.path.join(BASE_FOLDER, timestamp)
            os.makedirs(save_folder, exist_ok=True)

            image_path = os.path.join(save_folder, "image.jpg")
            text_path = os.path.join(save_folder, "description.txt")

            # Lưu ảnh và mô tả
            cv2.imwrite(image_path, frame)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(label)

            print(f"✅ Dự đoán: {label} (conf={confidence:.2f}) → lưu {image_path}")
            upload_to_firebase(timestamp, image_path, label)

        # Hiển thị camera
        display = frame.copy()
        cv2.putText(display, f"{label} ({confidence:.2f})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("ASL Realtime", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Dừng chương trình.")

finally:
    cap.release()
    cv2.destroyAllWindows()
