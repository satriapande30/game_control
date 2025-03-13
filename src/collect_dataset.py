import cv2
import os
import logging
import time
from datetime import datetime
import mediapipe as mp

# Konfigurasi Dataset
DATASET_PATH = './data/raw'  # Direktori untuk menyimpan dataset
CAMERA_INDEX = 0  # Indeks kamera (biasanya 0 untuk webcam default)
POSES = ['belok_kanan', 'belok_kiri', 'gas', 'nitro', 'rem']  # Daftar pose
IMAGE_SIZE = (1920, 1080)  # Resolusi gambar (1080p)
IMAGE_EXTENSION = '.jpg'  # Format file gambar
LOG_DIR = 'output/log'  # Direktori untuk menyimpan log
MAX_IMAGES = 500  # Jumlah maksimum gambar yang diambil

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def setup_logging():
    """Menyiapkan logging untuk pelacakan proses."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"dataset_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Logging setup completed.")

def create_dataset_structure():
    """Membuat struktur direktori untuk dataset."""
    try:
        for pose in POSES:
            path = os.path.join(DATASET_PATH, pose)
            os.makedirs(path, exist_ok=True)
            logging.info(f"Created directory: {path}")
        logging.info("Dataset structure created successfully.")
    except Exception as e:
        logging.error(f"Failed to create dataset structure: {e}")
        raise

def capture_images():
    """Proses pengambilan gambar otomatis dengan deteksi 2 tangan."""
    logging.info("Starting automatic image acquisition process.")
    
    # Buka kamera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_SIZE[1])

    if not cap.isOpened():
        logging.error("Failed to open camera.")
        return

    pose_index = 0  # Indeks pose saat ini
    count = 0  # Jumlah gambar yang telah diambil
    last_capture_time = 0  # Waktu terakhir gambar diambil

    logging.info("Camera opened successfully.")
    logging.info("Controls: 'p': change pose, 'q': quit")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7) as hands:

        while count < MAX_IMAGES:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture frame.")
                break

            frame = cv2.flip(frame, 1)  # Flip horizontal untuk mirror effect
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            current_pose = POSES[pose_index]

            # Tampilkan informasi pada frame
            cv2.putText(frame, f"Pose: {current_pose}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Count: {count}/500", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Deteksi 2 tangan dan auto-capture setiap 0.2 detik (5x lebih cepat)
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
                current_time = time.time()
                if current_time - last_capture_time >= 0.2:
                    img_name = f"{current_pose}_{count:04d}{IMAGE_EXTENSION}"
                    img_path = os.path.join(DATASET_PATH, current_pose, img_name)
                    cv2.imwrite(img_path, frame)
                    count += 1
                    last_capture_time = current_time
                    logging.info(f"Auto-captured: {img_path}")

                    if count >= MAX_IMAGES:
                        logging.info(f"Reached {MAX_IMAGES} images. Exiting.")
                        break

            # Tampilkan landmarks tangan jika terdeteksi
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('Hand Pose Acquisition', frame)

            key = cv2.waitKey(1)
            if key == ord('p'):  # Ganti pose
                pose_index = (pose_index + 1) % len(POSES)
                logging.info(f"Changed pose to: {POSES[pose_index]}")
            elif key == ord('q'):  # Keluar
                logging.info(f"Exiting. Total images captured: {count}")
                break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Image acquisition process completed.")

def main():
    """Fungsi utama untuk menjalankan script."""
    try:
        setup_logging()
        logging.info("Starting collect_data.py")
        create_dataset_structure()
        capture_images()
        logging.info("collect_data.py completed successfully.")
    except Exception as e:
        logging.error(f"Script failed with error: {e}")

if __name__ == "__main__":
    main()