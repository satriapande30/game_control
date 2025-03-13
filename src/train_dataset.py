import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

class GameControlTrainer:
    def __init__(self, input_file, batch_size=1000):
        self.input_file = input_file
        self.batch_size = batch_size
        self.setup_directories()
        self.logger = self._setup_logger()
        self.scaler = StandardScaler()
        # Define game control poses
        self.pose_classes = ['belok_kanan', 'belok_kiri', 'gas', 'nitro', 'rem']
        
    def setup_directories(self):
        """Create required directories"""
        directories = [
            'output/log',
            'output/model',
            'output/visualization'
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def _setup_logger(self):
        """Setup logging system"""
        log_file = f'output/log/game_control_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger

    def calculate_hand_features(self, data):
        """Calculate geometric features for game control hand poses"""
        features = {}
        
        # Process both hands
        for hand in range(2):
            hand_features = {}
            
            # 1. Calculate palm center
            palm_landmarks = [0, 5, 9, 13, 17]  # wrist and finger bases
            palm_points = np.array([[
                data[f'hand{hand}_lm{i}_x'],
                data[f'hand{hand}_lm{i}_y'],
                data[f'hand{hand}_lm{i}_z']] for i in palm_landmarks
            ])
            palm_center = np.mean(palm_points, axis=0)
            
            # 2. Calculate distances from palm center to fingertips
            fingertip_indices = [4, 8, 12, 16, 20]  # thumb to pinky tips
            for idx, tip_idx in enumerate(fingertip_indices):
                tip_point = np.array([
                    data[f'hand{hand}_lm{tip_idx}_x'],
                    data[f'hand{hand}_lm{tip_idx}_y'],
                    data[f'hand{hand}_lm{tip_idx}_z']
                ])
                hand_features[f'finger{idx}_dist'] = np.linalg.norm(tip_point - palm_center)
            
            # 3. Calculate angles between fingers
            for i in range(len(fingertip_indices)-1):
                p1 = np.array([
                    data[f'hand{hand}_lm{fingertip_indices[i]}_x'],
                    data[f'hand{hand}_lm{fingertip_indices[i]}_y'],
                    data[f'hand{hand}_lm{fingertip_indices[i]}_z']
                ])
                p2 = np.array([
                    data[f'hand{hand}_lm{fingertip_indices[i+1]}_x'],
                    data[f'hand{hand}_lm{fingertip_indices[i+1]}_y'],
                    data[f'hand{hand}_lm{fingertip_indices[i+1]}_z']
                ])
                hand_features[f'angle_{i}_{i+1}'] = np.arctan2(
                    np.linalg.norm(np.cross(p1, p2)),
                    np.dot(p1, p2)
                )
            
            # Add hand-specific features to main features dict
            for key, value in hand_features.items():
                features[f'hand{hand}_{key}'] = value
            
        return features

    def process_batch(self, batch_df):
        """Process one batch of data"""
        features = []
        for _, row in batch_df.iterrows():
            hand_features = self.calculate_hand_features(row)
            features.append(list(hand_features.values()))
        return features

    def prepare_data(self):
        """Prepare data for training"""
        self.logger.info("Reading and preparing dataset...")
        
        df = pd.read_csv(self.input_file)
        total_samples = len(df)
        num_batches = (total_samples + self.batch_size - 1) // self.batch_size
        
        features = []
        labels = df['class'].values
        
        for i in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, total_samples)
            batch_df = df.iloc[start_idx:end_idx]
            
            # Parallel processing for batch
            batch_features = Parallel(n_jobs=-1)(
                delayed(self.calculate_hand_features)(row)
                for _, row in batch_df.iterrows()
            )
            features.extend([list(d.values()) for d in batch_features])
        
        features = np.array(features)
        features = self.scaler.fit_transform(features)
        
        return features, labels

    def train_model(self):
        """Train the SVM model"""
        self.logger.info("Starting training process...")
        
        # Data preparation
        X, y = self.prepare_data()
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train SVM
        self.logger.info("Training SVM model...")
        svm = SVC(kernel='rbf', C=1.0, probability=True)
        svm.fit(X_train, y_train)
        
        # Evaluate model
        train_score = svm.score(X_train, y_train)
        test_score = svm.score(X_test, y_test)
        
        self.logger.info(f"Training accuracy: {train_score*100:.2f}%")
        self.logger.info(f"Testing accuracy: {test_score*100:.2f}%")
        
        # Predictions and evaluation
        y_pred = svm.predict(X_test)
        
        # Save model and scaler
        model_path = 'output/model/game_control_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({'model': svm, 'scaler': self.scaler}, f)
        
        # Visualize results
        self.visualize_results(y_test, y_pred, train_score, test_score)
        
        return svm, (X_train, X_test, y_train, y_test)

    def visualize_results(self, y_true, y_pred, train_score, test_score):
        """Create visualization of training results"""
        self.logger.info("Creating visualizations...")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(cm_percentage, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='Blues',
                   xticklabels=self.pose_classes,
                   yticklabels=self.pose_classes)
        plt.title('Game Control Hand Pose Classification - Confusion Matrix (%)')
        plt.xlabel('Predicted Pose')
        plt.ylabel('True Pose')
        plt.tight_layout()
        plt.savefig('output/visualization/game_control_confusion_matrix.png')
        plt.close()
        
        # 2. Classification Report Visualization
        report = classification_report(y_true, y_pred, 
                                    target_names=self.pose_classes,
                                    output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.iloc[:, :3] *= 100  # Convert to percentages
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(report_df.iloc[:-3, :3].astype(float), 
                   annot=True, 
                   fmt='.1f',
                   cmap='YlOrRd')
        plt.title('Classification Metrics by Game Control Pose (%)')
        plt.tight_layout()
        plt.savefig('output/visualization/game_control_metrics.png')
        plt.close()
        
        # 3. Training vs Testing Performance
        plt.figure(figsize=(8, 6))
        performance = {
            'Training': train_score * 100,
            'Testing': test_score * 100
        }
        plt.bar(performance.keys(), performance.values(), color=['#2ecc71', '#3498db'])
        plt.title('Model Performance Comparison')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        
        for i, v in enumerate(performance.values()):
            plt.text(i, v + 1, f'{v:.1f}%', ha='center')
            
        plt.tight_layout()
        plt.savefig('output/visualization/game_control_performance.png')
        plt.close()

if __name__ == "__main__":
    INPUT_FILE = "./output/features/landmarks.csv"
    
    trainer = GameControlTrainer(INPUT_FILE)
    model, (X_train, X_test, y_train, y_test) = trainer.train_model()