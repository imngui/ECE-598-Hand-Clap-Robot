

import os
import ast
from datetime import datetime
import numpy as np

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

import torch
import torch.nn as nn
import torch.nn.functional as F

from hand_clap.contrastive_autoencoder_model import ContrastiveAutoencoder, HandFeatureExtractor  # Import the model class


class OnlineGestureBuffer:
    def __init__(self, window_size, feature_dim, device='cpu'):
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.device = device
        
        # Initialize circular buffer with zeros
        self.buffer = torch.zeros(window_size, feature_dim, device=device)
        self.write_idx = 0  # Current write position in circular buffer
        self.initialized = False 
        
    def add_observation(self, observation):
        observation = observation.to(self.device)
        
        if not self.initialized:
            self.buffer[:] = observation.unsqueeze(0).repeat(self.window_size, 1)
            self.initialized = True
            self.write_idx = 0
        else:
            self.buffer[self.write_idx] = observation
            self.write_idx = (self.write_idx + 1) % self.window_size
    
    def get_sequence(self):
        if not self.initialized:
            return self.buffer
        
        return torch.cat([
            self.buffer[self.write_idx:],
            self.buffer[:self.write_idx]
        ], dim=0)
    
    def is_ready(self, min_observations=1):
        return self.initialized
    
    def reset(self):
        self.buffer.zero_()
        self.write_idx = 0
        self.initialized = False


class GestureClassifier(Node):

    def __init__(self):
        super().__init__('gesture_classifier')

        self.declare_parameter('model_path', os.path.join(get_package_share_directory('hand_clap'), 'config', 'contrastive_autoencoder.pt'))
        self.declare_parameter('window_size', 10)
        self.declare_parameter('confidence_threshold', 0.7) 

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.window_size = self.get_parameter('window_size').get_parameter_value().integer_value
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value

        # Load the checkpoint
        try:
            checkpoint = torch.load(self.model_path)
            self.get_logger().info(f'Checkpoint loaded successfully from {self.model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load checkpoint from {self.model_path}: {e}')
            raise e

        input_dim = checkpoint.get('input_dim', 322)
        output_dim = len(checkpoint.get('label_map', {}))
        hidden_dim = checkpoint.get('hidden_dim', 128)
        latent_dim = checkpoint.get('latent_dim', 64)
        window_size = checkpoint.get('window_size', 10)

        self.model = ContrastiveAutoencoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            window_size=window_size,
            num_classes=output_dim
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.label_map = checkpoint.get('label_map', {})

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.buffer = OnlineGestureBuffer(
            window_size=self.window_size,
            feature_dim=input_dim,
            device=self.device
        )

        self.feature_extractor = HandFeatureExtractor()

        self.hand_tracking_joints_sub = self.create_subscription(
            JointTrajectory,
            '/hand_tracking_joints',
            self.hand_tracking_joints_callback,
            10
        )

        self.gesture_pub = self.create_publisher(
            String,
            '/recognized_gesture',
            10
        )

        self.class_labels = [k for k, v in sorted(self.label_map.items(), key=lambda x: x[1])]

        self.get_logger().info(f'Gesture classifier initialized with:')
        self.get_logger().info(f'  Window size: {self.window_size} (fixed memory)')
        self.get_logger().info(f'  Confidence threshold: {self.confidence_threshold}')
        self.get_logger().info(f'  Device: {self.device}')
        self.get_logger().info(f'  Note: First observation will be duplicated to fill buffer')

    def hand_tracking_joints_callback(self, msg):
        left_landmarks = np.zeros((26, 3))
        right_landmarks = np.zeros((26, 3))

        if len(msg.points) >= 2:
            left_point = msg.points[0]
            right_point = msg.points[1]

            left_positions = np.array(left_point.positions).reshape(-1, 3)
            right_positions = np.array(right_point.positions).reshape(-1, 3)

            left_landmarks = left_positions[:26] if len(left_positions) >= 26 else left_landmarks
            right_landmarks = right_positions[:26] if len(right_positions) >= 26 else right_landmarks

        features = self.feature_extractor.extract(left_landmarks, right_landmarks)

        observation = torch.tensor(features, dtype=torch.float32)

        self.buffer.add_observation(observation)

        sequence = self.buffer.get_sequence()

        with torch.no_grad():
            input_tensor = sequence.unsqueeze(0).to(self.device)

            _, _, _, _, logits = self.model(input_tensor)

            probabilities = torch.softmax(logits, dim=-1)

            confidence, predicted_idx = torch.max(probabilities, dim=-1)
            pred = predicted_idx.cpu().numpy()[0]
            conf = confidence.cpu().numpy()[0]
            probs = probabilities.cpu().numpy()[0]

            if conf >= self.confidence_threshold:
                self.get_logger().info(
                    f'✓ Predicted: {self.class_labels[pred]} | Confidence: {conf:.3f} | '
                    f'Probs: {probs}'
                )
                
                gesture_msg = String()
                gesture_msg.data = self.class_labels[pred]
                self.gesture_pub.publish(gesture_msg)

                with open('temp.txt', 'a') as f:
                    f.write(f'Predicted gesture: {self.class_labels[pred]} with confidence: {conf:.2f}\n')
                    f.write(f'Probabilities: {probs}\n')
            else:
                self.get_logger().info(
                    f'✗ Low confidence: {conf:.3f} | Probs: {probs}',
                    throttle_duration_sec=1.0
                )


def main():
    rclpy.init()
    gesture_classifier = GestureClassifier()
    rclpy.spin(gesture_classifier)
    gesture_classifier.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
