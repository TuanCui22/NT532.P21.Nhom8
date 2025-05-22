import os
import cv2
import mediapipe as mp
import numpy as np
import time
import argparse

# Cấu hình
DATA_DIR = './data'
NUMBER_OF_CLASSES = 26  # A-Z (26 chữ cái)
DATASET_SIZE = 15       # Số lượng hình ảnh cho mỗi chữ cái
PADDING = 30            # Padding xung quanh bàn tay
FRAME_RATE = 25         # FPS cho hiển thị video

# Hàm chuyển đổi chữ cái thành index
def letter_to_index(letter):
    """Chuyển đổi chữ cái thành index (A=0, B=1, ...)"""
    if len(letter) != 1 or not letter.isalpha():
        raise ValueError("Input must be a single letter")

    letter = letter.upper()
    index = ord(letter) - ord('A')

    if index < 0 or index >= 26:
        raise ValueError("Letter must be from A to Z")

    return index

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def initialize_camera():
    """Khởi tạo camera và trả về đối tượng VideoCapture"""
    for camera_index in range(3):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None and test_frame.shape[0] > 0 and test_frame.shape[1] > 0:
                print(f"Using camera index: {camera_index}")
                return cap
            else:
                cap.release()

    print("Error: Could not find a working camera")
    exit(1)

def create_directories():
    """Tạo thư mục dữ liệu và các thư mục con cho từng lớp"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    for j in range(NUMBER_OF_CLASSES):
        class_dir = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

def detect_hand_and_adjust_frame(frame, results):
    """Phát hiện bàn tay và điều chỉnh khung hình để bao quanh bàn tay"""
    h, w = frame.shape[:2]

    # Mặc định vị trí khung hình ở giữa nếu không phát hiện bàn tay
    frame_size = min(h, w) // 2
    x = w//2 - frame_size//2
    y = h//2 - frame_size//2

    # Nếu phát hiện bàn tay, điều chỉnh khung hình để bao quanh bàn tay
    if results.multi_hand_landmarks:
        # Lấy landmark của bàn tay đầu tiên
        hand_landmarks = results.multi_hand_landmarks[0]

        # Tính toán tọa độ của bounding box
        x_coords = [landmark.x for landmark in hand_landmarks.landmark]
        y_coords = [landmark.y for landmark in hand_landmarks.landmark]

        # Chuyển đổi tọa độ từ tỉ lệ sang pixel
        x_min = int(min(x_coords) * w)
        y_min = int(min(y_coords) * h)
        x_max = int(max(x_coords) * w)
        y_max = int(max(y_coords) * h)

        # Thêm padding để khung hình lớn hơn bàn tay một chút
        x_min = max(0, x_min - PADDING)
        y_min = max(0, y_min - PADDING)
        x_max = min(w, x_max + PADDING)
        y_max = min(h, y_max + PADDING)

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

    return x, y, frame_size, results.multi_hand_landmarks

def draw_frame_markers(display_frame, x, y, frame_size):
    """Vẽ khung hình vuông và các điểm đánh dấu"""
    # Vẽ khung hình vuông
    cv2.rectangle(display_frame, (x, y), (x + frame_size, y + frame_size), (0, 0, 0), 3)

    # Vẽ các điểm đánh dấu ở các góc
    corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    corners = [(x, y), (x + frame_size, y), (x, y + frame_size), (x + frame_size, y + frame_size)]

    for corner, color in zip(corners, corner_colors):
        cv2.circle(display_frame, corner, 5, color, -1)

    # Vẽ các điểm ở giữa các cạnh
    midpoint_colors = [(255, 0, 255), (0, 255, 255), (255, 255, 255), (128, 128, 128)]
    midpoints = [
        (x + frame_size//2, y),                # Giữa cạnh trên
        (x + frame_size//2, y + frame_size),   # Giữa cạnh dưới
        (x, y + frame_size//2),                # Giữa cạnh trái
        (x + frame_size, y + frame_size//2)    # Giữa cạnh phải
    ]

    for midpoint, color in zip(midpoints, midpoint_colors):
        cv2.circle(display_frame, midpoint, 5, color, -1)

def process_frame(cap, display_text, instruction_text):
    """Xử lý một frame từ camera và trả về các thông tin cần thiết"""
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        return None, None, None, None, None, None

    # Tạo bản sao của frame để hiển thị
    display_frame = frame.copy()

    # Chuyển đổi frame sang RGB để xử lý với MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Xử lý frame để phát hiện bàn tay
    results = hands.process(frame_rgb)

    # Hiển thị thông tin
    h, w = frame.shape[:2]
    cv2.putText(display_frame, display_text, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Phát hiện bàn tay và điều chỉnh khung hình
    x, y, frame_size, hand_landmarks = detect_hand_and_adjust_frame(frame, results)

    # Vẽ các landmark của bàn tay nếu phát hiện được
    if hand_landmarks:
        mp_drawing.draw_landmarks(
            display_frame,
            hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS)

    # Vẽ khung hình và các điểm đánh dấu
    draw_frame_markers(display_frame, x, y, frame_size)

    # Hiển thị hướng dẫn
    cv2.putText(display_frame, instruction_text, (10, h - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return frame, display_frame, x, y, frame_size, h

def wait_for_start(cap, class_index):
    """Hiển thị màn hình chờ cho đến khi người dùng nhấn 'q'"""
    letter = chr(65 + class_index)  # 65 là mã ASCII của 'A'
    display_text = f'Ready for letter {letter}? Press "Q"!'
    instruction_text = "Place your hand inside the frame"

    while True:
        frame, display_frame, x, y, frame_size, h = process_frame(cap, display_text, instruction_text)
        if frame is None:
            return False

        cv2.imshow('frame', display_frame)
        if cv2.waitKey(1000 // FRAME_RATE) == ord('q'):
            return True

def collect_images(cap, class_index, dataset_size=DATASET_SIZE):
    """Thu thập hình ảnh cho một lớp cụ thể"""
    letter = chr(65 + class_index)  # 65 là mã ASCII của 'A'
    counter = 0

    # Kiểm tra xem thư mục đã có ảnh chưa
    class_dir = os.path.join(DATA_DIR, str(class_index))
    existing_images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
    if existing_images:
        # Tìm số lớn nhất trong tên file hiện có
        max_num = max([int(os.path.splitext(f)[0]) for f in existing_images])
        counter = max_num + 1
        print(f"Found {len(existing_images)} existing images. Starting from index {counter}.")

    while counter < dataset_size:
        # Hiển thị thông tin tiến trình
        display_text = f'Collecting letter {letter}: {counter+1}/{dataset_size}'
        instruction_text = "Keep your hand inside the frame"

        frame, display_frame, x, y, frame_size, h = process_frame(cap, display_text, instruction_text)
        if frame is None:
            break

        cv2.imshow('frame', display_frame)
        key = cv2.waitKey(1000 // FRAME_RATE)

        # Nếu người dùng nhấn 'q', dừng thu thập
        if key == ord('q'):
            print("Collection stopped by user")
            break

        # Nếu người dùng nhấn 's', lưu ảnh hiện tại
        if key == ord('s') or key == 32:  # 's' hoặc phím space
            # Cắt vùng quan tâm (ROI) từ frame gốc
            roi = frame[y:y+frame_size, x:x+frame_size]

            # Kiểm tra kích thước của ROI
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                # Chuyển đổi sang ảnh xám
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Lưu ảnh
                cv2.imwrite(os.path.join(DATA_DIR, str(class_index), f'{counter}.jpg'), gray)

                # Hiển thị thông báo đã lưu ảnh
                cv2.putText(display_frame, f"Saved image {counter+1}", (10, h - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('frame', display_frame)
                cv2.waitKey(200)  # Hiển thị thông báo trong 200ms

                counter += 1
                # Thêm một khoảng thời gian nhỏ giữa các lần chụp để tránh lấy các frame quá giống nhau
                time.sleep(0.1)

def parse_arguments():
    """Phân tích tham số dòng lệnh"""
    parser = argparse.ArgumentParser(description='Collect images for sign language detection')
    parser.add_argument('--letter', type=str, help='Specific letter to collect (A-Z)')
    parser.add_argument('--all', action='store_true', help='Collect all letters from A to Z')
    parser.add_argument('--size', type=int, default=DATASET_SIZE, help='Number of images to collect for each letter')
    return parser.parse_args()

def main():
    """Hàm chính của chương trình"""
    try:
        # Phân tích tham số dòng lệnh
        args = parse_arguments()

        # Cập nhật số lượng ảnh cần thu thập nếu được chỉ định
        dataset_size = args.size if args.size else DATASET_SIZE

        # Tạo thư mục dữ liệu
        create_directories()

        # Khởi tạo camera
        cap = initialize_camera()

        # Xác định các chữ cái cần thu thập
        if args.letter:
            # Thu thập dữ liệu cho một chữ cái cụ thể
            try:
                class_index = letter_to_index(args.letter)
                letters_to_collect = [class_index]
            except ValueError as e:
                print(f"Error: {str(e)}")
                return
        elif args.all:
            # Thu thập dữ liệu cho tất cả các chữ cái
            letters_to_collect = range(NUMBER_OF_CLASSES)
        else:
            # Nếu không có tham số, hiển thị menu để người dùng chọn
            print("Choose a letter to collect data for:")
            print("0-25: Collect data for a specific letter (A-Z)")
            print("26: Collect data for all letters")
            print("q: Quit")

            while True:
                choice = input("Enter your choice: ")
                if choice.lower() == 'q':
                    return

                try:
                    choice = int(choice)
                    if choice == 26:
                        letters_to_collect = range(NUMBER_OF_CLASSES)
                        break
                    elif 0 <= choice < NUMBER_OF_CLASSES:
                        letters_to_collect = [choice]
                        break
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number or 'q'.")

        # Thu thập dữ liệu cho các chữ cái đã chọn
        for class_index in letters_to_collect:
            letter = chr(65 + class_index)
            print(f'Collecting data for class {class_index} (Letter {letter})')

            # Chờ người dùng sẵn sàng
            if not wait_for_start(cap, class_index):
                break

            # Thu thập hình ảnh
            collect_images(cap, class_index, dataset_size)

            # Hiển thị thông báo hoàn thành
            print(f'Completed collecting data for letter {letter}')

    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Giải phóng tài nguyên
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Program finished")

if __name__ == "__main__":
    main()