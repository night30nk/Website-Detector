

```markdown
# 🛡️ Phishing URL Detector using PyTorch and Flask

This project is a machine learning-based web application that detects phishing URLs in real time. It uses a neural network built with PyTorch and serves predictions via a simple Flask web interface.

---


---

## 🚀 Features

- Extracts intelligent features from URLs (length, special characters, presence of keywords, etc.).
- Trains a binary classifier using PyTorch to detect phishing vs. legitimate URLs.
- Real-time predictions via a user-friendly Flask web app.
- Uses `tqdm` for progress bars and shows evaluation metrics like accuracy, precision, and recall.

---

## 🧪 Dataset

The dataset used in this project is publicly available:

📎 [https://github.com/ZephyrCX/dataset_url_70000](https://github.com/ZephyrCX/dataset_url_70000)

---

## ✅ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/phishing-url-detector.git
cd phishing-url-detector
````

### 2. Set Up Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you are on a CPU-only machine:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 🧠 Training the Model

If you want to retrain the model from scratch:

```bash
python train.py
```

This will:

* Load and preprocess the dataset
* Extract features
* Train a simple PyTorch neural network
* Save the model to `model.pt`

---

## 🌐 Run the Flask App

```bash
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:5000
```

Enter a URL, and the app will predict whether it’s **phishing** or **safe**, along with a confidence score.

---

## ⚙️ Feature Extraction

The model uses the following features from each URL:

* URL length
* Number of dots `.`
* Number of hyphens `-`
* Number of slashes `/`
* Number of subdomains
* Presence of IP address
* Presence of `@` symbol
* Presence of `//` after protocol
* HTTPS presence
* Count of suspicious keywords
* Number of digits
* Number of special characters

---

## 📊 Example Output

```
🔗 URL: http://malicious-login.freehost.com/secure-update
⚠️ Prediction: Phishing
📈 Confidence: 0.97
```

---

## 🛠️ Requirements

* Python 3.7 to 3.12
* PyTorch
* Flask
* pandas
* scikit-learn
* tqdm
* joblib
* numpy

---

## 📌 Notes

* No GPU (CUDA) is required. Works perfectly on CPU.
* Make sure the `templates/` folder contains `index.html` to avoid Flask `TemplateNotFound` errors.

---

## 📃 License

This project is open-source under the MIT License.

---

## 🤝 Contributing

Pull requests are welcome! If you have suggestions for improvements, feel free to open an issue or PR.

---

```

Let me know if you'd like to include screenshots or deployment instructions (e.g., Docker or Heroku).
```
