import torch.nn as nn


class VoiceAuthenticationBinaryClassifier(nn.Module):
    def __init__(self, num_mfcc, num_frames):
        super(VoiceAuthenticationBinaryClassifier, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        # Dense Layers
        reduced_size = num_frames // 16  # Assuming 4 MaxPooling layers each of size 2x2
        self.fc1 = nn.Linear(6784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape should be [batch_size, 1, num_mfcc, num_frames]
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x



# import torch.nn as nn
#
#
# class VoiceAuthenticationBinaryClassifier(nn.Module):
#     def __init__(self, num_mfcc, num_frames):
#         super(VoiceAuthenticationBinaryClassifier, self).__init__()
#         self.fc1 = nn.Linear(num_mfcc * num_frames, 128)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)  # Add dropout with a dropout rate of 0.5
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x
