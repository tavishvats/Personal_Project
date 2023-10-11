import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocessing.dataset import VoiceAuthenticationDataSet
from model.model import VoiceAuthenticationBinaryClassifier

root_dir = '../preprocessing'
train_dir = '../preprocessing/train_labels.csv'
val_dir = '../preprocessing/val_labels.csv'

batch_size = 32
learning_rate = 0.001
num_epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = VoiceAuthenticationDataSet(train_dir, root_dir)
val_dataset = VoiceAuthenticationDataSet(val_dir, root_dir)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the binary classification model
model = VoiceAuthenticationBinaryClassifier(num_mfcc=13, num_frames=431)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Adding the learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

model.to(device)

train_losses = []
val_losses = []
val_accuracies = []

best_val_loss = float('inf')
patience = 5
wait = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        features, labels = data
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        labels = labels.unsqueeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            labels = labels.unsqueeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predicted = torch.round(outputs).squeeze()
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)

    val_accuracy = 100 * correct / total
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch + 1}/{num_epochs}] '
          f'Training Loss: {train_loss:.10f} - '
          f'Validation Loss: {val_loss:.10f} - '
          f'Validation Accuracy: {val_accuracy:.2f}%')

    # Update the scheduler
    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        wait = 0
    else:
        wait += 1

    if wait >= patience:
        print(f'Early stopping after {epoch + 1} epochs without improvement.')
        break

# Plot the training and validation loss curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the trained binary classification model
torch.save(model.state_dict(), 'voice_authentication_binary_model.pth')
print('Training finished')





# import torch
# import torch.optim as optim
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from preprocessing.dataset import VoiceAuthenticationDataSet
# from model.model import VoiceAuthenticationBinaryClassifier  # Updated model class
#
# root_dir = '../preprocessing'
# train_dir = '../preprocessing/train_labels.csv'
# val_dir = '../preprocessing/val_labels.csv'
#
# batch_size = 64
# learning_rate = 0.001
# num_epochs = 100
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# train_dataset = VoiceAuthenticationDataSet(train_dir, root_dir)
# val_dataset = VoiceAuthenticationDataSet(val_dir, root_dir)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
# # Initialize the binary classification model
# model = VoiceAuthenticationBinaryClassifier(num_mfcc=13, num_frames=431)  # Updated model class
# criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# model.to(device)
#
# train_losses = []
# val_losses = []
#
# # Initialize variables for early stopping
# best_val_loss = float('inf')
# patience = 5
# wait = 0
#
# # Training loop
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#
#     for data in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
#         features, labels = data
#
#         features = features.to(device)
#         labels = labels.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(features)  # Get model predictions
#
#         # BCELoss expects labels to be in the range [0, 1]
#         labels = labels.unsqueeze(1)  # Convert to column tensor
#         loss = criterion(outputs, labels)  # Compute BCE loss
#
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#     # Validation loop
#     model.eval()
#     val_loss = 0.0
#
#     with torch.no_grad():
#         for features, labels in val_loader:
#             features = features.to(device)
#             labels = labels.to(device)
#
#             outputs = model(features)  # Get model predictions
#
#             labels = labels.unsqueeze(1)  # Convert to column tensor
#             loss = criterion(outputs, labels)  # Compute BCE loss
#             val_loss += loss.item()
#
#     # Calculate and store training loss
#     train_loss = running_loss / len(train_loader)
#     train_losses.append(train_loss)
#
#     # Calculate and store validation loss
#     val_loss = val_loss / len(val_loader)
#     val_losses.append(val_loss)
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}] '
#           f'Training Loss: {train_loss:.4f} - '
#           f'Validation Loss: {val_loss:.4f}')
#
#     # Check if validation loss improved
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         wait = 0
#     else:
#         wait += 1
#
#     # If validation loss hasn't improved for 'patience' epochs, stop training
#     if wait >= patience:
#         print(f'Early stopping after {epoch + 1} epochs without improvement.')
#         break
#
# # Plot the training and validation loss curves
# plt.figure(figsize=(10, 5))
# plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
# plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # Save the trained binary classification model
# torch.save(model.state_dict(), 'voice_authentication_binary_model.pth')
# print('Training finished')
