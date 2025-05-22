import os
import cv2
import pickle
import base64
import numpy as np
import firebase_admin
from firebase_admin import credentials, db

# Khởi tạo Firebase Admin SDK
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://iot-hien-dai-default-rtdb.asia-southeast1.firebasedatabase.app'
})

# Tải model ASL
MODEL_PATH = 'model.p'
try:
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        asl_model = model_data['model']  # Trích xuất model từ dictionary
except Exception as e:
    raise RuntimeError(f"Không thể tải model: {str(e)}")

def preprocess_image(image_path):
    """Tiền xử lý ảnh đầu vào cho model"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Không đọc được ảnh")
    img = cv2.resize(img, (32, 32))  # Đảm bảo đúng với model đã train
    img = img.astype('float32') / 255.0
    return img.reshape(1, -1)  # (1, 1024)

def predict_asl(image_path):
    """Dự đoán ký hiệu ASL từ ảnh"""
    try:
        processed_img = preprocess_image(image_path)
        preds = asl_model.predict(processed_img)
        # Mapping label index to ASL alphabet
        LABELS = {
            0: "A", 1: "B", 2: "C", 3: "D", 4: "E",
            5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
            10: "K", 11: "L", 12: "M", 13: "N", 14: "O",
            15: "P", 16: "Q", 17: "R", 18: "S", 19: "T",
            20: "U", 21: "V", 22: "W", 23: "X", 24: "Y",
            25: "Z", 26: "SPACE", 27: "DELETE", 28: "NOTHING"
        }
        return LABELS.get(preds[0], "KHÔNG_XÁC_ĐỊNH")
    except Exception as e:
        raise RuntimeError(f"Lỗi dự đoán: {str(e)}")

def process_asl_predictions(folder_path):
    """Xử lý ảnh và tạo file dự đoán ASL"""    
    folders = [f for f in os.listdir(folder_path) 
              if os.path.isdir(os.path.join(folder_path, f)) and f.isdigit()]
    
    print(f"🔍 Đang xử lý {len(folders)} thư mục...")
    
    for folder in folders:
        current_folder = os.path.join(folder_path, folder)
        image_path = os.path.join(current_folder, 'image.jpg')
        text_path = os.path.join(current_folder, 'description.txt')
        
        if os.path.exists(text_path):
            continue
            
        if os.path.exists(image_path):
            try:
                prediction = predict_asl(image_path)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(prediction)
                print(f"✅ Đã xử lý xong: {folder}")
            except Exception as e:
                print(f"❌ Lỗi khi xử lý {folder}: {str(e)}")  # Only print to console

def upload_folder(folder_path):
    """Tải lên Firebase"""
    ref = db.reference('images')
    folders = [f for f in os.listdir(folder_path) 
              if os.path.isdir(os.path.join(folder_path, f)) and f.isdigit()]
    folders.sort(key=lambda x: int(x))

    print(f"📁 Tìm thấy {len(folders)} thư mục")

    for folder in folders:
        current_folder_path = os.path.join(folder_path, folder)
        image_path = os.path.join(current_folder_path, 'image.jpg')
        text_path = os.path.join(current_folder_path, 'description.txt')

        if all([os.path.exists(image_path), os.path.exists(text_path)]):
            try:
                with open(image_path, "rb") as img_file, \
                     open(text_path, "r", encoding='utf-8') as txt_file:
                    
                    image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                    text_content = txt_file.read().strip()

                    ref.child(folder).set({
                        "image": f"data:image/jpeg;base64,{image_base64}",
                        "text": text_content,
                        "timestamp": int(folder)
                    })
                    print(f"✅ Đã tải lên: {folder}")
            except Exception as e:
                print(f"❌ Lỗi khi tải lên {folder}: {str(e)}")  # Only print to console

if __name__ == "__main__":
    process_asl_predictions('./images')
    upload_folder('./images')
    print("🎉 Hoàn tất quá trình xử lý và tải lên")
