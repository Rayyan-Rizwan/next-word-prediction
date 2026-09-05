
# ⚡ AI Next Word Predictor / Sentence Completer

An interactive, deep-learning-powered text completion web application built with **LSTM (Long Short-Term Memory)** neural networks and **Streamlit**. The application provides real-time, customizable next-word predictions with a modern glassmorphism dark-themed user interface.

🌐 **Live Demo:** [https://next-word-prediction-8ehwby6pr2sdkvgakmxxdd.streamlit.app/](https://next-word-prediction-8ehwby6pr2sdkvgakmxxdd.streamlit.app/)

---

## 🌟 Key Features

- **Deep Learning Powered**: Employs a trained LSTM model built on TensorFlow/Keras for coherent text completion.
- **Multi-Option Predictions**: Generates 1 to 5 unique text completion choices per prompt using temperature-based probabilistic sampling.
- **Modern Glassmorphism UI**: Stylish dark theme with interactive cards, custom CSS hover animations, and dynamic controls.
- **Configurable Generation**: Interactively adjust parameters such as text sequence length (1–20 words) and creativity/temperature (0.2–1.5).

---

## 📁 Project Structure

```text
├── app.py                      # Main Streamlit web application code
├── quote_prediction_model.h5   # Trained Keras/TensorFlow LSTM model weights & architecture
├── tokenizer.pickle            # Pickled Keras Tokenizer instance for text serialization
├── model_config.json           # Model configuration file specifying sequence lengths
├── qoute_dataset.csv           # Source quote dataset used for model training
├── requirements.txt            # Python dependencies pinned for Streamlit Cloud
└── README.md                   # Project documentation

```

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites

Ensure you have Python 3.8+ installed on your system.

### 2. Clone the Repository

```bash
git clone [https://github.com/Rayyan-Rizwan/next-word-prediction.git](https://github.com/Rayyan-Rizwan/next-word-prediction.git)
cd next-word-prediction

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the Streamlit App

```bash
streamlit run app.py

```

---

## ⚙️ Deployment Info

The app is deployed on **Streamlit Community Cloud** with pre-configured deserialization handling and pinned dependencies to ensure seamless TensorFlow model loading across Keras runtime environments.

---

## 🛠️ Built With

* **[TensorFlow / Keras](https://www.tensorflow.org/)** - Deep learning architecture & model training
* **[Streamlit](https://streamlit.io/)** - Interactive web UI framework
* **[NumPy](https://numpy.org/)** - Array manipulation and temperature-based sampling math
* **[Pickle](https://docs.python.org/3/library/pickle.html)** - Serialization of the text tokenizer

```

```
