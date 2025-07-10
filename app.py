from flask import Flask, request, render_template
import torch
import torch.nn as nn
import joblib
from features import extract_features
import numpy as np

app = Flask(__name__)

# Define the same model architecture as in training
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

# Load model and scaler
model = PhishingDetector(input_size=12)  # Adjust if features count changed
model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
model.eval()

scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    url_input = ''
    if request.method == 'POST':
        url_input = request.form.get('url')

        if url_input:
            # Extract features
            features = extract_features(url_input)
            features = np.array(features).reshape(1, -1)
            # Scale features
            features_scaled = scaler.transform(features)
            # Convert to tensor
            input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
            # Get model output
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
                prediction = "Phishing URL ⚠️" if prob >= 0.5 else "Legitimate URL ✅"
                prediction += f" (Confidence: {prob:.2f})"
        else:
            prediction = "Please enter a URL."

    return render_template('index.html', prediction=prediction, url_input=url_input)

if __name__ == '__main__':
    app.run(debug=True)
