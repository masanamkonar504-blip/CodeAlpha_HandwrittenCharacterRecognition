# NeuroWrite AI – Handwritten Character Recognition

NeuroWrite AI is a handwritten character recognition application built using a Convolutional Neural Network (CNN). It predicts handwritten digits (0–9) and uppercase English letters (A–Z).

## Features
- Recognizes digits from 0 to 9.
- Recognizes uppercase letters from A to Z.
- Interactive drawing canvas.
- Displays the top 3 predictions with confidence scores.
- Built with a CNN deep learning model.

## Technologies Used
- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

## Dataset
- MNIST – handwritten digits.
- EMNIST Letters – handwritten English letters.

## Model Performance
- Test Accuracy: **91.77%**
- Test Loss: **0.2556**
- Training: 5 epochs

## Project Files
- `app.py` – Streamlit web application.
- `requirements.txt` – Required Python libraries.
- `neurowrite_model.keras` – Trained CNN model.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Created By
**Konar Masanam Ram**  
B.Tech – Artificial Intelligence and Data Science  
Machine Learning Intern – CodeAlpha

---

*This project was developed for educational and internship purposes.*
