import os
import pickle
import cv2
import numpy as np

DATA_DIR = './data'

data = []
labels = []

# Đơn giản hóa: Thay vì sử dụng mediapipe để trích xuất đặc trưng bàn tay,
# chúng ta sẽ sử dụng các đặc trưng đơn giản từ hình ảnh
for dir_ in os.listdir(DATA_DIR):
    if not dir_.isdigit():
        continue

    print(f"Processing class {dir_}")
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        # Đọc hình ảnh
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is None:
            continue

        # Chuyển đổi sang ảnh xám
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Thay đổi kích thước ảnh để giảm số lượng đặc trưng
        img_resized = cv2.resize(img_gray, (32, 32))

        # Làm phẳng ảnh thành vector đặc trưng
        features = img_resized.flatten()

        # Thêm vào tập dữ liệu
        data.append(features)
        labels.append(int(dir_))

# Lưu tập dữ liệu
print(f"Collected {len(data)} samples")

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
