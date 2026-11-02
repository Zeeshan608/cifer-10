# 🔮 Vision AI: CIFAR-10 Image Classifier

A deep learning web app that recognizes what is in a photo. It uses a **ResNet50** model, pretrained on ImageNet and fine-tuned on the **CIFAR-10** dataset, with a Streamlit frontend.

🌐 **Live Demo:** [Open the app][https://cifer-10.streamlit.app/]

![App Screenshot](screenshot.png)

---

## 📌 Overview

Upload any photo and the model predicts which of the 10 CIFAR-10 classes it belongs to, along with a confidence score and the top 3 predictions.

**Classes:** ✈️ airplane · 🚗 automobile · 🐦 bird · 🐱 cat · 🦌 deer · 🐶 dog · 🐸 frog · 🐴 horse · 🚢 ship · 🚚 truck

---

## 🧠 How It Works

1. **Data:** 50,000 labelled 32x32 images from the Kaggle CIFAR-10 competition.
2. **Preprocessing:** label encoding, data checks, stratified 80/20 train/validation split, and ResNet50 input normalization.
3. **Model:** pretrained ResNet50 (ImageNet weights) with data augmentation (random flip and translation), upscaling from 32x32 to 128x128, global average pooling, dropout, and a 10-class softmax head.
4. **Training (two stages):**
   - Stage 1: train only the new classification head with the base frozen.
   - Stage 2: unfreeze the whole network and fine-tune with a small learning rate.
5. **Evaluation:** accuracy and loss curves, confusion matrix, and a per-class precision/recall report.
6. **Deployment:** Streamlit app hosted on Streamlit Community Cloud.

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Model | TensorFlow, Keras, ResNet50 |
| Data handling | NumPy, Pandas, Pillow, scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Frontend | Streamlit, custom CSS |
| Training environment | Google Colab (GPU) |

---

## 📁 Project Structure

```
cifar10-app/
├── app.py                    # Streamlit web app
├── cifar10_resnet50.keras    # Trained model
├── requirements.txt          # Python dependencies
├── screenshot.png            # App screenshot used in this README
└── README.md
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Zeeshan608/cifer-10.git
cd cifer-10

# 2. (Optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 💡 Tips for Best Results

- Use a clear photo with **one main object**, centered, on a simple background.
- The model only knows the 10 classes above. Anything else (a person, a laptop) will still be matched to the closest of those 10.
- Images are resized to 32x32 before prediction, so very busy or low-quality photos can reduce accuracy.

---

## ⚠️ Limitations

- CIFAR-10 images are tiny and simple, so the model can struggle with real-world photos that look very different from the training data.
- Cats and dogs are the most commonly confused classes.

---

## 🔮 Future Improvements

- Train on a larger input size (e.g. 224x224) for higher accuracy
- Add more classes with a bigger dataset
- Add Grad-CAM to show which part of the image the model focused on

---

## 👤 Author

**Zeeshan**
GitHub: [@Zeeshan608](https://github.com/Zeeshan608)

---

⭐ If you found this project useful, consider giving it a star!
