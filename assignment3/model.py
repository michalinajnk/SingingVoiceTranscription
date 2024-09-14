import torch
import torch.nn as nn

class BaseCNN_mini(nn.Module):
    '''
    This is a base CNN model.
    '''

    def __init__(self, feat_dim=256, pitch_class=13, pitch_octave=5):
        '''
        Definition of network structure.
        '''
        super(BaseCNN_mini, self).__init__()
        self.feat_dim = feat_dim
        self.pitch_class = pitch_class
        self.pitch_octave = pitch_octave

        # Convolutional Layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2)),  # Pool along feature dimension only
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2)),  # Pool along feature dimension only
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(64 * feat_dim // 4, 256),  # Adjust input dimension after convolutional and pooling layers
            nn.ReLU()
        )

        # Prediction Heads
        self.onset_head = nn.Linear(256, 1)
        self.offset_head = nn.Linear(256, 1)
        self.octave_head = nn.Linear(256, self.pitch_octave)
        self.pitch_class_head = nn.Linear(256, self.pitch_class)

    def forward(self, x):
        '''
        Compute output from input
        '''
        # Add channel dimension
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, time, feature)

        # Convolutional layers
        x = self.conv1(x)  # Output shape: (batch_size, 16, time, feature)
        x = self.conv2(x)  # Output shape: (batch_size, 32, time, feature / 2)
        x = self.conv3(x)  # Output shape: (batch_size, 64, time, feature / 4)

        x = x.permute(0, 2, 1, 3)
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # Keeping time dimension separate and batch, merge feature and channel

        # Fully connected layer applied to each frame
        x = self.fc(x)  # Shape: (batch_size, time, 256)

        # Prediction heads
        onset_logits = self.onset_head(x).squeeze(-1)  # Shape: (batch_size, time)
        offset_logits = self.offset_head(x).squeeze(-1)  # Shape: (batch_size, time)
        pitch_octave_logits = self.octave_head(x)  # Shape: (batch_size, time, pitch_octave)
        pitch_class_logits = self.pitch_class_head(x)  # Shape: (batch_size, time, pitch_class)

        return onset_logits, offset_logits, pitch_octave_logits, pitch_class_logits
