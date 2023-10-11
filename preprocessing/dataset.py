import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os


class VoiceAuthenticationDataSet(Dataset):
    def __init__(self, csv_file, root_dir):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        feature_filename = self.data_frame.iloc[idx, 1]
        label = int(self.data_frame.iloc[idx, 0])  # Convert label to integer

        feature_path = os.path.join(self.root_dir, feature_filename)
        features = np.load(feature_path)

        # Convert to tensor and add a channel dimension
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add a channel

        # Standardize the features
        mean = features.mean()
        std = features.std()
        features = (features - mean) / (std + 1e-7)  # Add a small value to prevent division by zero

        label = torch.tensor(label, dtype=torch.float32)

        return features, label



# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import pandas as pd
# import os
#
#
# class VoiceAuthenticationDataSet(Dataset):
#     def __init__(self, csv_file, root_dir):
#         self.data_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#
#     def __len__(self):
#         return len(self.data_frame)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         feature_filename = self.data_frame.iloc[idx, 1]
#         label = int(self.data_frame.iloc[idx, 0])  # Convert label to integer
#
#         feature_path = os.path.join(self.root_dir, feature_filename)
#
#         features = np.load(feature_path)
#
#         features = torch.tensor(features).unsqueeze(0)  # Add a channel
#
#         mean = features.mean()
#         std = features.std()
#         features = (features - mean) / std
#
#         label = torch.tensor(label, dtype=torch.float32)
#
#         return features, label
