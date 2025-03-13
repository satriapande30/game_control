import cv2
import mediapipe as mp
import os
import csv
import logging
from datetime import datetime
from tqdm import tqdm
import shutil
import time
from multiprocessing import Pool, cpu_count

class EnhancedHandProcessor:
    def __init__(self, dataset_path, output_file, error_dir='output/error_image',
                 min_detection_confidence=0.5, max_hands=2):
        # Inisialisasi parameter
        self.dataset_path = dataset_path
        self.output_file = output_file
        self.error_dir = error_dir
        self.min_confidence = min_detection_confidence
        self.max_hands = max_hands
        self.hands_params = {
            'static_image_mode': True,
            'max_num_hands': self.max_hands,
            'min_detection_confidence': self.min_confidence
        }
        
        # Setup infrastruktur
        self._setup_infrastructure()
        self._init_stats()

    def _setup_infrastructure(self):
        """Membuat struktur direktori dan logging"""
        os.makedirs('output/log', exist_ok=True)
        os.makedirs(self.error_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Konfigurasi logging
        self.logger = logging.getLogger('HandProcessor')
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(
            f'output/log/processor_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _init_stats(self):
        """Inisialisasi statistik pemrosesan"""
        self.stats = {
            'total': 0,
            'detected': 0,
            'undetected': 0,
            'errors': 0,
            'class_dist': {},
            'processing_time': 0
        }

    def analyze_dataset(self):
        """Analisis awal dataset"""
        self.logger.info("Memulai analisis dataset...")
        for class_name in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg','.jpeg','.png'))])
                self.stats['class_dist'][class_name] = count
                self.stats['total'] += count
        
        # Cek keseimbangan data
        class_counts = list(self.stats['class_dist'].values())
        if len(set(class_counts)) > 1:
            self.logger.warning("Dataset tidak seimbang!")
        
        self.logger.info(f"Total gambar: {self.stats['total']}")
        self.logger.info("Distribusi kelas:")
        for cls, cnt in self.stats['class_dist'].items():
            self.logger.info(f"- {cls}: {cnt} ({cnt/self.stats['total']*100:.1f}%)")

    @staticmethod
    def process_image(args):
        """Fungsi processing untuk multiprocessing"""
        img_path, class_name, hands_params = args
        try:
            # Inisialisasi MediaPipe Hands di setiap worker
            hands = mp.solutions.hands.Hands(**hands_params)
            
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Gambar tidak valid")
            
            # Resize dengan aspect ratio
            h, w = img.shape[:2]
            if h > w:
                new_h = 640
                new_w = int(w * new_h / h)
            else:
                new_w = 640
                new_h = int(h * new_w / w)
            img = cv2.resize(img, (new_w, new_h))
            
            # Deteksi tangan
            results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            landmarks = []
            if results.multi_hand_landmarks:
                for hand in results.multi_hand_landmarks:
                    hand_data = [coord for lm in hand.landmark 
                                for coord in [lm.x, lm.y, lm.z]]
                    landmarks.extend(hand_data)
                
                # Padding jika tangan kurang dari max_hands
                while len(landmarks) < hands_params['max_num_hands']*21*3:
                    landmarks.extend([0.0]*21*3)
                
                return [class_name] + landmarks[:hands_params['max_num_hands']*21*3]
            
            return None
            
        except Exception as e:
            error_dir = 'output/error_image'
            os.makedirs(error_dir, exist_ok=True)
            error_path = os.path.join(error_dir, os.path.basename(img_path))
            shutil.copy(img_path, error_path)
            return f"ERROR|{img_path}|{str(e)}"
        finally:
            hands.close()

    def process_dataset(self):
        """Memproses seluruh dataset"""
        start_time = time.time()
        self.analyze_dataset()
        
        # Siapkan semua argumen pemrosesan
        tasks = []
        for class_name in self.stats['class_dist']:
            class_path = os.path.join(self.dataset_path, class_name)
            for img_file in os.listdir(class_path):
                if img_file.lower().endswith(('.jpg','.jpeg','.png')):
                    tasks.append((
                        os.path.join(class_path, img_file),
                        class_name,
                        self.hands_params
                    ))

        # Proses paralel dengan manajer untuk shared stats
        with Pool(cpu_count()) as pool:
            with open(self.output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Header
                header = ['class']
                for hand in range(self.max_hands):
                    for i in range(21):
                        header.extend([
                            f'hand{hand}_lm{i}_x',
                            f'hand{hand}_lm{i}_y',
                            f'hand{hand}_lm{i}_z'
                        ])
                writer.writerow(header)
                
                # Proses dengan progress bar
                with tqdm(total=len(tasks), desc="Memproses Gambar") as pbar:
                    for result in pool.imap(self.process_image, tasks):
                        if result:
                            if isinstance(result, list):
                                writer.writerow(result)
                                self.stats['detected'] += 1
                            else:
                                # Handle error message
                                _, img_path, error_msg = result.split('|')
                                self.logger.error(f"Error di {img_path}: {error_msg}")
                                self.stats['errors'] += 1
                                self.stats['undetected'] += 1
                        else:
                            self.stats['undetected'] += 1
                        pbar.update(1)

        # Hitung statistik akhir
        self.stats['processing_time'] = time.time() - start_time
        self._generate_report()

    def _generate_report(self):
        """Hasilkan laporan akhir"""
        report = [
            "\n=== LAPORAN PEMROSESAN DATASET ===",
            f"Total gambar: {self.stats['total']}",
            f"Berhasil diproses: {self.stats['detected']} ({self.stats['detected']/self.stats['total']*100:.1f}%)",
            f"Gagal diproses: {self.stats['undetected']}",
            f"Gambar error: {self.stats['errors']}",
            f"Waktu pemrosesan: {self.stats['processing_time']:.2f} detik",
            "==================================="
        ]
        
        for line in report:
            self.logger.info(line)

if __name__ == "__main__":
    processor = EnhancedHandProcessor(
        dataset_path="./data/raw",
        output_file="./output/features/landmarks.csv",
        max_hands=2,
        min_detection_confidence=0.6
    )
    processor.process_dataset()