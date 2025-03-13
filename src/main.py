import cv2
import mediapipe as mp
import numpy as np
import pickle
import keyboard
from pynput.keyboard import Controller

class GameControlImplementation:
    def __init__(self, model_path='output/model/game_control_model.pkl'):
        # Load trained model and scaler
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Initialize keyboard controller
        self.keyboard_controller = Controller()
        
        # Control mapping
        self.control_mapping = {
            'belok_kanan': 'd',
            'belok_kiri': 'a',
            'gas': 'w',
            'nitro': 'n',
            'rem': 's'
        }
        
        # Current states
        self.current_control = None
        self.prev_control = None
        self.wheel_angle = 0
        self.target_angle = 0
        self.wheel_center = None
        self.initial_hand_position = None
        self.steering_active = False
        
        # Smoothing and sensitivity settings
        self.angle_smoothing = 0.3
        self.movement_sensitivity = 2.0
        self.n_features = self.scaler.n_features_in_

    def extract_hand_features(self, results):
        """Extract features for ML model prediction"""
        if not results.multi_hand_landmarks:
            return None
            
        features = np.zeros(self.n_features)
        
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:1]):
            for i, landmark in enumerate(hand_landmarks.landmark):
                if i * 3 + 2 < self.n_features:
                    features[i * 3] = landmark.x
                    features[i * 3 + 1] = landmark.y
                    features[i * 3 + 2] = landmark.z
                
        return features

    def calculate_steering_angle(self, hand_landmarks, frame_width):
        """Calculate steering angle based on hand movement"""
        if not hand_landmarks:
            return 0
            
        # Get center point of palm
        palm_center_x = hand_landmarks.landmark[9].x
        
        # Convert to pixel coordinates
        palm_x = int(palm_center_x * frame_width)
        center_x = frame_width // 2
        
        # Calculate angle based on distance from center
        max_angle = 45  # Maximum steering angle
        relative_pos = (palm_x - center_x) / (frame_width / 4)  # Normalize position
        angle = np.clip(relative_pos * max_angle, -max_angle, max_angle)
        
        return angle

    def get_hands_center(self, results):
        """Calculate center position between hands"""
        if not results.multi_hand_landmarks:
            return None
            
        x_sum = 0
        y_sum = 0
        num_landmarks = 0
        
        for hand_landmarks in results.multi_hand_landmarks:
            # Use only specific landmarks for better center calculation
            key_points = [0, 9, 13]  # wrist, palm center, middle finger base
            for idx in key_points:
                x_sum += hand_landmarks.landmark[idx].x
                y_sum += hand_landmarks.landmark[idx].y
                num_landmarks += 1
                
        if num_landmarks == 0:
            return None
            
        return (x_sum / num_landmarks, y_sum / num_landmarks)

    def draw_steering_wheel(self, frame, control, hands_position=None):
        """Draw interactive steering wheel overlay"""
        h, w, _ = frame.shape
        
        # Update wheel center position based on hands
        if hands_position:
            if not self.wheel_center:
                self.wheel_center = (int(hands_position[0] * w), int(hands_position[1] * h))
            else:
                # Smooth transition for wheel position
                target_x = int(hands_position[0] * w)
                target_y = int(hands_position[1] * h)
                self.wheel_center = (
                    int(self.wheel_center[0] + (target_x - self.wheel_center[0]) * 0.3),
                    int(self.wheel_center[1] + (target_y - self.wheel_center[1]) * 0.3)
                )
        
        if not self.wheel_center:
            self.wheel_center = (w // 2, h // 2)
        
        # Calculate wheel size based on frame height
        radius = int(h * 0.2)  # 20% of frame height
        
        # Create overlay
        overlay = frame.copy()
        
        # Draw main circle
        color = (0, 255, 128)  # Cyan color
        cv2.circle(overlay, self.wheel_center, radius, color, 3)
        
        # Update and draw rotation
        if control == 'belok_kanan':
            self.target_angle = 45
        elif control == 'belok_kiri':
            self.target_angle = -45
        else:
            self.target_angle = 0
        
        # Smooth angle transition
        self.wheel_angle += (self.target_angle - self.wheel_angle) * self.angle_smoothing
        
        # Draw spokes with current rotation
        spoke_length = int(radius * 0.8)
        for spoke_angle in [self.wheel_angle, self.wheel_angle + 180]:
            rad = np.radians(spoke_angle)
            end_x = int(self.wheel_center[0] + spoke_length * np.cos(rad))
            end_y = int(self.wheel_center[1] + spoke_length * np.sin(rad))
            cv2.line(overlay, self.wheel_center, (end_x, end_y), color, 3)
        
        # Add control indicators
        if control:
            indicator_y = self.wheel_center[1] - radius - 30
            if control == 'gas':
                cv2.putText(overlay, "GAS", 
                           (self.wheel_center[0] - 30, indicator_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif control == 'nitro':
                cv2.putText(overlay, "NITRO", 
                           (self.wheel_center[0] - 40, indicator_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif control == 'rem':
                cv2.putText(overlay, "BRAKE", 
                           (self.wheel_center[0] - 40, indicator_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Blend overlay with original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame

    def draw_pose_info(self, frame, control):
        """Draw pose information overlay"""
        h, w, _ = frame.shape
        
        # Create semi-transparent overlay for text background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, h-80), (w-10, h-20), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Draw control information
        if control:
            cv2.putText(frame, f"Control: {control.upper()}", 
                       (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame

    def simulate_key_press(self, control):
        """Simulate keyboard press for game control"""
        if control != self.prev_control:
            if self.prev_control:
                keyboard.release(self.control_mapping[self.prev_control])
            if control:
                keyboard.press(self.control_mapping[control])
            self.prev_control = control

    def run(self):
        """Main control loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        cv2.namedWindow('Game Control', cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                # Get ML model prediction
                features = self.extract_hand_features(results)
                if features is not None:
                    feature_vector = features.reshape(1, -1)
                    scaled_features = self.scaler.transform(feature_vector)
                    self.current_control = self.model.predict(scaled_features)[0]
                    
                    # Simulate key press
                    self.simulate_key_press(self.current_control)
                
                # Calculate hands center for wheel position
                hands_center = self.get_hands_center(results)
                
                # Draw hand landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Update and draw steering wheel
                frame = self.draw_steering_wheel(frame, self.current_control, hands_center)
            else:
                # Reset control when no hands detected
                if self.current_control:
                    self.simulate_key_press(None)
                    self.current_control = None
                frame = self.draw_steering_wheel(frame, None)
            
            # Draw pose information
            frame = self.draw_pose_info(frame, self.current_control)
            
            cv2.imshow('Game Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = GameControlImplementation()
    controller.run()