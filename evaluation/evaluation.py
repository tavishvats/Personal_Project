import torch
from torch.utils.data import DataLoader
from preprocessing.dataset import VoiceAuthenticationDataSet
from model.model import VoiceAuthenticationBinaryClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_model(model, loader, device):
    model.eval()
    true_labels, predicted_labels = [], []

    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            predicted_labels.extend(torch.sigmoid(outputs).squeeze(1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return true_labels, predicted_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = '../preprocessing'
test_dir = '../preprocessing/test_labels.csv'
model_path = '../training/voice_authentication_binary_model.pth'

batch_size = 32
threshold = 0.5

# Dataset and Loader
test_dataset = VoiceAuthenticationDataSet(test_dir, data_dir)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Initialization
model = VoiceAuthenticationBinaryClassifier(num_mfcc=13, num_frames=431).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

true_labels, predicted_labels = evaluate_model(model, test_loader, device)
predicted_bin_labels = [1 if label > threshold else 0 for label in predicted_labels]

# Evaluation Metrics
accuracy = accuracy_score(true_labels, predicted_bin_labels)
precision = precision_score(true_labels, predicted_bin_labels)
recall = recall_score(true_labels, predicted_bin_labels)
f1 = f1_score(true_labels, predicted_bin_labels)
cm = confusion_matrix(true_labels, predicted_bin_labels)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Confusion Matrix:\n{cm}')


# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from preprocessing.dataset import VoiceAuthenticationDataSet
# from model.model import VoiceAuthenticationBinaryClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
# # Paths to your data and saved model
# data_dir = '../preprocessing'  # Update with the correct data path
# test_dir = '../preprocessing/test_labels.csv'  # Path to your test labels file, no need to go up one directory
# model_path = '../training/voice_authentication_binary_model.pth'  # Path to your saved model
#
# # Hyperparameters
# batch_size = 32
#
# # Create a dataset and data loader for testing
# test_dataset = VoiceAuthenticationDataSet(test_dir, data_dir)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# # Initialize the model
# model = VoiceAuthenticationBinaryClassifier(num_mfcc=13, num_frames=431)
# model.load_state_dict(torch.load(model_path))
# model.eval()
#
# # Lists to store predicted and true labels
# predicted_labels = []
# true_labels = []
#
# # Testing loop
# with torch.no_grad():
#     for features, labels in test_loader:
#         outputs = model(features)
#         print(outputs)
#         # Convert outputs to binary predictions based on a threshold if needed
#         predicted_labels.extend(outputs.squeeze(1).cpu().numpy())
#         true_labels.extend(labels.cpu().numpy())
#
# # Calculate evaluation metrics
# predicted_labels = [1 if label > 0.5 else 0 for label in predicted_labels]  # Adjust threshold as needed
# accuracy = accuracy_score(true_labels, predicted_labels)
# precision = precision_score(true_labels, predicted_labels)
# recall = recall_score(true_labels, predicted_labels)
# f1 = f1_score(true_labels, predicted_labels)
#
# # Print evaluation metrics
# print(f'Accuracy: {accuracy:.4f}')
# print(f'Precision: {precision:.4f}')
# print(f'Recall: {recall:.4f}')
# print(f'F1 Score: {f1:.4f}')
