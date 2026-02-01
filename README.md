# Document Sentiment Classifier

## 📖 Overview
This project focuses on **sentiment analysis of text documents**, classifying reviews as **positive or negative** using deep learning techniques. The project implements and compares multiple neural network architectures including **Multi-Layer Perceptron (MLP)**, **Convolutional Neural Networks (CNN)**, and **Long Short-Term Memory (LSTM)** models.

This work was completed as part of the **DSCI 552 – Machine Learning for Data Science** final project.

---

## ❓ Problem Statement
Given a collection of text reviews, the goal is to automatically determine whether a review expresses **positive** or **negative** sentiment.

---

## 📂 Dataset
- Text reviews stored as individual `.txt` files
- Two classes:
  - Positive reviews
  - Negative reviews
- Labels:
  - `+1` → Positive
  - `-1` → Negative
- Data split:
  - **Training:** Files `0–699`
  - **Testing:** Files `700–999`

---

## 🧹 Data Preprocessing
- Removed punctuation and numeric characters
- Tokenized text based on word frequency
- Computed:
  - Number of unique words
  - Average review length
  - Standard deviation of review lengths
- Selected a maximum review length **L** such that 70–90% of reviews fall below it
- Applied:
  - Truncation for long reviews
  - Zero-padding for short reviews

---

## 🧠 Models Implemented

### 1️⃣ Multi-Layer Perceptron (MLP)
- Embedding layer (32-dimensional vectors)
- 3 dense hidden layers (50 ReLU units each)
- Dropout:
  - 20% (first layer)
  - 50% (remaining layers)
- Optimizer: Adam
- Loss: Binary Cross-Entropy

### 2️⃣ Convolutional Neural Network (CNN)
- Embedding layer
- Conv1D with:
  - 32 filters
  - Kernel size = 3
- MaxPooling1D
- Dense layers similar to MLP

### 3️⃣ LSTM Network
- Embedding layer (32 dimensions)
- LSTM layer with dropout
- Dense layer with 256 ReLU units
- Sigmoid output for binary classification

---

## 📊 Results
Each model was evaluated using **training and testing accuracy** to compare performance and generalization capability. The results demonstrate the effectiveness of deep learning methods for text-based sentiment classification.

(Exact accuracy values can be found in the Jupyter notebook.)

---

## 🛠️ Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook

---

## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Document-Sentiment-Classifier.git

2. Navigate to project directory:
   ```bash
   cd Document-Sentiment-Classifie
   
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook

4. Run GUMMADI_VIDYA SRI_PROJECT.ipynb

