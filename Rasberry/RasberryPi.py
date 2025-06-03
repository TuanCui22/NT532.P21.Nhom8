import cv2
import requests
import time

SERVER_URL = "http://0.0.0.0:5000/upload"  # thay IP server
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không mở được camera.")
    exit()

print("📸 Đang capture, nhấn Ctrl+C để dừng.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        _, img_encoded = cv2.imencode('.jpg', frame)
        response = requests.post(SERVER_URL, files={"image": img_encoded.tobytes()})

        if response.status_code == 200:
            print("✅ Gửi ảnh thành công.")
        else:
            print(f"❌ Gửi ảnh lỗi: {response.status_code}")

except KeyboardInterrupt:
    print("🛑 Dừng lại.")
finally:
    cap.release()
