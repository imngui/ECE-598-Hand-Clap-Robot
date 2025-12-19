import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        h_last = out[:, -1, :]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar


class TemporalDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.fc(z)
        h = h.unsqueeze(1).repeat(1, self.window_size, 1)
        out, _ = self.lstm(h)
        out = self.fc_out(out)
        return out


class ContrastiveAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, window_size, num_classes):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = TemporalDecoder(latent_dim, hidden_dim, input_dim, window_size)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        logits = self.classifier(z)
        return x_recon, z, mu, logvar, logits


class HandFeatureExtractor:
    def __init__(self):
        self.prev_frame = None

    def extract(self, left_hand, right_hand):
        features = []

        # 1. Normalized positions
        left_normalized = self.normalize_hand(left_hand)
        right_normalized = self.normalize_hand(right_hand)
        features.extend([left_normalized, right_normalized])

        # 2. Velocities
        if self.prev_frame is not None:
            left_velocity = left_hand - self.prev_frame['left']
            right_velocity = right_hand - self.prev_frame['right']
            features.extend([left_velocity, right_velocity])
        else:
            features.extend([np.zeros_like(left_hand), np.zeros_like(right_hand)])

        # 3. Inter-hand features
        hand_distance = np.linalg.norm(
            self.get_palm_center(left_hand) - self.get_palm_center(right_hand)
        )
        hand_relative_pos = self.get_palm_center(right_hand) - self.get_palm_center(left_hand)
        features.extend([np.array([hand_distance]), hand_relative_pos])

        # 4. Hand orientation/rotation features
        left_orientation = self.compute_hand_orientation(left_hand)
        right_orientation = self.compute_hand_orientation(right_hand)
        features.extend([left_orientation, right_orientation])

        # Update history
        self.prev_frame = {'left': left_hand.copy(), 'right': right_hand.copy()}

        # Flatten to 1D vector
        return np.concatenate([f.flatten() for f in features])

    def normalize_hand(self, landmarks):
        wrist = landmarks[0]  # Wrist is first landmark
        palm = landmarks[1]   # Palm is second landmark

        # Center at wrist
        centered = landmarks - wrist

        # Normalize by wrist-to-palm distance
        hand_scale = np.linalg.norm(palm - wrist) + 1e-6
        normalized = centered / hand_scale

        return normalized

    def get_palm_center(self, landmarks):
        return landmarks[1]

    def compute_hand_orientation(self, landmarks):
        wrist = landmarks[0]
        palm = landmarks[1]
        index_base = landmarks[6]  # IndexMetacarpal

        # Normal vector to palm plane
        v1 = palm - wrist
        v2 = index_base - wrist
        normal = np.cross(v1, v2)
        normal = normal / (np.linalg.norm(normal) + 1e-6)

        return normal

    def reset(self):
        self.prev_frame = None
