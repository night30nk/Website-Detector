# train_model.py

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from features import extract_features
import numpy as np
import joblib

torch.manual_seed(42)

# Load dataset
df = pd.read_csv('Dataset.csv')
df.drop(columns=['Unnamed: 0'], inplace=True)

print("Extracting features from URLs...")
X = [extract_features(url) for url in tqdm(df['URL'])]
y = df['status'].values.reshape(-1, 1)

X = np.array(X)
y = np.array(y)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, y_tensor, test_size=0.2, random_state=42, stratify=y_tensor)

# Prepare DataLoader for batching
batch_size = 64
train_ds = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

test_ds = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# Define model with dropout
class PhishingDetector(nn.Module):
    def __init__(self, input_size):
        super(PhishingDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # logits
        return x

model = PhishingDetector(input_size=X_tensor.shape[1])

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50

print("Training model...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss:.4f}")

# Evaluate
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).float()
        all_preds.append(preds)
        all_labels.append(labels)

all_preds = torch.cat(all_preds).cpu().numpy()
all_labels = torch.cat(all_labels).cpu().numpy()

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds)
rec = recall_score(all_labels, all_preds)

print("\n🧪 Evaluation Results:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")

# Save model and scaler
torch.save(model.state_dict(), 'model.pt')
joblib.dump(scaler, 'scaler.pkl')

print("\n✅ Model saved to model.pt and scaler saved to scaler.pkl")
