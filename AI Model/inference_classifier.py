import pickle
import cv2
import numpy as np
import mediapipe as mp

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Tải mô hình
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Thử các camera index khác nhau
for camera_index in range(3):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        ret, test_frame = cap.read()
        if ret and test_frame is not None and test_frame.shape[0] > 0 and test_frame.shape[1] > 0:
            print(f"Using camera index: {camera_index}")
            break
        else:
            cap.release()

    if camera_index == 2:  # Nếu không tìm thấy camera nào hoạt động
        print("Error: Could not find a working camera")
        print("Using dummy video mode for testing")
        # Tạo một frame giả để kiểm tra
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "No camera found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', dummy_frame)
        cv2.waitKey(2000)  # Hiển thị trong 2 giây
        exit(1)

# Tạo dictionary cho tất cả các chữ cái từ A đến Z
labels_dict = {}
for i in range(26):
    labels_dict[i] = chr(65 + i)  # 65 là mã ASCII của 'A'
print("Labels dictionary:", labels_dict)

while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Tạo một bản sao của frame để hiển thị
    display_frame = frame.copy()

    # Chuyển đổi frame sang RGB để xử lý với MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xử lý frame để phát hiện bàn tay
    results = hands.process(frame_rgb)

    # Mặc định vị trí khung hình ở giữa nếu không phát hiện bàn tay
    h, w = frame.shape[:2]
    frame_size = min(h, w) // 2
    x = w//2 - frame_size//2
    y = h//2 - frame_size//2

    # Nếu phát hiện bàn tay, điều chỉnh khung hình để bao quanh bàn tay
    if results.multi_hand_landmarks:
        # Lấy landmark của bàn tay đầu tiên
        hand_landmarks = results.multi_hand_landmarks[0]

        # Vẽ các landmark của bàn tay
        mp_drawing.draw_landmarks(
            display_frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS)

        # Tính toán tọa độ của bounding box
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]

        # Chuyển đổi tọa độ từ tỉ lệ sang pixel
        x_min = int(min(x_coords) * w)
        y_min = int(min(y_coords) * h)
        x_max = int(max(x_coords) * w)
        y_max = int(max(y_coords) * h)

        # Thêm padding để khung hình lớn hơn bàn tay một chút
        padding = 30
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Đảm bảo khung hình là hình vuông
        width = x_max - x_min
        height = y_max - y_min
        size = max(width, height)

        # Điều chỉnh tọa độ để tạo khung hình vuông
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        x_min = max(0, x_center - size // 2)
        y_min = max(0, y_center - size // 2)
        x_max = min(w, x_center + size // 2)
        y_max = min(h, y_center + size // 2)

        # Cập nhật vị trí và kích thước khung hình
        x = x_min
        y = y_min
        frame_size = x_max - x_min

    # Vẽ khung hình vuông
    cv2.rectangle(display_frame, (x, y), (x + frame_size, y + frame_size), (0, 0, 0), 3)

    # Vẽ các điểm đánh dấu trên khung hình (tương tự như trong hình mẫu)
    # Điểm ở góc trên bên trái
    cv2.circle(display_frame, (x, y), 5, (255, 0, 0), -1)
    # Điểm ở góc trên bên phải
    cv2.circle(display_frame, (x + frame_size, y), 5, (0, 255, 0), -1)
    # Điểm ở góc dưới bên trái
    cv2.circle(display_frame, (x, y + frame_size), 5, (0, 0, 255), -1)
    # Điểm ở góc dưới bên phải
    cv2.circle(display_frame, (x + frame_size, y + frame_size), 5, (255, 255, 0), -1)

    # Vẽ thêm một số điểm ở giữa các cạnh (tương tự như trong hình mẫu)
    # Điểm ở giữa cạnh trên
    cv2.circle(display_frame, (x + frame_size//2, y), 5, (255, 0, 255), -1)
    # Điểm ở giữa cạnh dưới
    cv2.circle(display_frame, (x + frame_size//2, y + frame_size), 5, (0, 255, 255), -1)
    # Điểm ở giữa cạnh trái
    cv2.circle(display_frame, (x, y + frame_size//2), 5, (255, 255, 255), -1)
    # Điểm ở giữa cạnh phải
    cv2.circle(display_frame, (x + frame_size, y + frame_size//2), 5, (128, 128, 128), -1)

    # Hiển thị hướng dẫn
    cv2.putText(display_frame, "Place your hand inside the frame", (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Cắt vùng quan tâm (ROI) từ frame gốc
    roi = frame[y:y+frame_size, x:x+frame_size]

    # Kiểm tra kích thước của ROI
    if roi.shape[0] > 0 and roi.shape[1] > 0:
        # Chuyển đổi sang ảnh xám
        img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Thay đổi kích thước ảnh để phù hợp với mô hình
    try:
        # Kiểm tra xem img_gray có tồn tại không
        if 'img_gray' in locals() and img_gray is not None and img_gray.size > 0:
            # Thay đổi kích thước ảnh để phù hợp với mô hình
            img_resized = cv2.resize(img_gray, (32, 32))

            # Làm phẳng ảnh thành vector đặc trưng
            features = img_resized.flatten()

            # Dự đoán
            prediction = model.predict([features])
            predicted_class = int(prediction[0])
            print(f"Predicted class: {predicted_class}")

            # Kiểm tra xem lớp dự đoán có trong labels_dict không
            if predicted_class in labels_dict:
                predicted_character = labels_dict[predicted_class]
                print(f"Predicted character: {predicted_character}")

                # Hiển thị kết quả
                cv2.putText(display_frame, f"Predicted: {predicted_character}",
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                # Hiển thị vùng ROI đã cắt ở góc màn hình
                if roi.shape[0] > 0 and roi.shape[1] > 0:
                    roi_display = cv2.resize(roi, (100, 100))
                    display_frame[20:120, w-120:w-20] = roi_display
                    cv2.rectangle(display_frame, (w-120, 20), (w-20, 120), (0, 255, 0), 2)
            else:
                error_msg = f"Predicted class {predicted_class} not in labels_dict"
                print(error_msg)
                cv2.putText(display_frame, f"Error: {error_msg}",
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(display_frame, "No hand detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        cv2.putText(display_frame, f"Error: {str(e)}",
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # Hiển thị frame
    try:
        # Đặt cửa sổ ở vị trí cố định
        cv2.namedWindow('Sign Language Detector', cv2.WINDOW_NORMAL)
        cv2.moveWindow('Sign Language Detector', 100, 100)
        cv2.imshow('Sign Language Detector', display_frame)
        print("Frame displayed successfully")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit key pressed")
            break
    except Exception as e:
        print(f"Error displaying frame: {str(e)}")


cap.release()
cv2.destroyAllWindows()
