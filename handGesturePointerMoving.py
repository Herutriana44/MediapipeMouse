import cv2
import mediapipe as mp
import pyautogui

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
sw, sh = pyautogui.size()
cap.set(cv2.CAP_PROP_FRAME_WIDTH,sw+500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,sh+500)

def get_hand_center(hand_landmarks):
    x_coords = [landmark.x for landmark in hand_landmarks.landmark]
    y_coords = [landmark.y for landmark in hand_landmarks.landmark]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return center_x, center_y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Konversi gambar ke RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = image.shape

    # Proses deteksi tangan
    results = hands.process(image)

    # Gambar landmarks tangan jika terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Mendapatkan posisi tengah tangan
            center_x, center_y = get_hand_center(hand_landmarks)
            center_x_px = int(center_x * image_width)
            center_y_px = int(center_y * image_height)
            pyautogui.moveTo(center_x_px, center_y_px)
            cv2.circle(frame, (center_x_px, center_y_px), 5, (255, 0, 0), -1)
            print(f"Posisi tengah tangan: X={center_x_px}, Y={center_y_px}")

            # Deteksi jari telunjuk ke atas
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
            if index_tip < index_pip:
                print("Jari telunjuk ke atas")
                pyautogui.click()
                cv2.putText(frame, "klik kiri", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Deteksi jari tengah ke atas
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
            if middle_tip < middle_pip:
                print("Jari tengah ke atas")
                pyautogui.click(button='right')
                cv2.putText(frame, "klik kanan", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan frame
    cv2.imshow('MediaPipe Hands', frame)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()