import streamlit as st
import numpy as np
import pickle
import json
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="AI Next Word Predictor",
    page_icon="⚡",
    layout="centered"
)

# -------------------------------------------------------------
# 2. Custom CSS (Styling, Animations, Colors)
# -------------------------------------------------------------
custom_css = """
<style>
/* Gradient Background */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    color: #f8fafc;
}

/* Glassmorphism Title Box */
.main-title {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    margin-bottom: 30px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    animation: fadeIn 1.2s ease-in-out;
}

.main-title h1 {
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem;
    font-weight: 800;
    margin: 0;
}

/* Styled Option Cards with Animations */
.option-card {
    background: rgba(30, 41, 59, 0.7);
    border-left: 5px solid #818cf8;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
    color: #e2e8f0;
    font-size: 1.1rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease-in-out;
    animation: slideUp 0.5s ease-out forwards;
}

.option-card:hover {
    transform: translateY(-4px) scale(1.01);
    border-left-color: #38bdf8;
    box-shadow: 0 8px 25px rgba(56, 189, 248, 0.25);
    background: rgba(30, 41, 59, 0.9);
}

.option-number {
    color: #38bdf8;
    font-weight: 700;
    margin-right: 8px;
}

/* Animations Keyframes */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="main-title">
    <h1>⚡ AI Sentence Completer</h1>
    <p style="color: #94a3b8; margin-top: 8px;">Generate multiple creative continuations using deep learning</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. Load Model Artifacts
# -------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model('quote_prediction_model.h5')
    with open('tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('model_config.json', 'r') as f:
        config = json.load(f)
        
    max_len = config.get('max_sequence_len') or config.get('max_len')
    return model, tokenizer, max_len

try:
    model, tokenizer, max_len = load_artifacts()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -------------------------------------------------------------
# 4. Prediction Logic
# -------------------------------------------------------------
def sample_with_temperature(preds, temperature=0.7):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-7) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def generate_multiple_options(seed_text, next_words, num_options=3, temperature=0.7):
    generated_options = []
    
    for _ in range(num_options):
        current_text = seed_text.strip()
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([current_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_len - 1 if max_len else None, padding='pre')
            
            preds = model.predict(token_list, verbose=0)[0]
            predicted_index = sample_with_temperature(preds, temperature)
            
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_index:
                    output_word = word
                    break
                    
            if not output_word:
                break
                
            current_text += " " + output_word
            
        generated_options.append(current_text)
        
    return generated_options

# -------------------------------------------------------------
# 5. UI Controls
# -------------------------------------------------------------
input_prompt = st.text_input("✨ Enter Starting Words:", value="success comes to")

col1, col2 = st.columns(2)
with col1:
    num_words = st.slider("Words to Generate:", min_value=1, max_value=20, value=6)
with col2:
    num_options = st.slider("Number of Options:", min_value=1, max_value=5, value=3)

with st.expander("⚙️ Advanced Settings"):
    temp = st.slider("Creativity (Temperature):", min_value=0.2, max_value=1.5, value=0.7, step=0.1)

# -------------------------------------------------------------
# 6. Output Display with Animation Effects
# -------------------------------------------------------------
if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
    if not input_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        with st.spinner("AI is thinking..."):
            time.sleep(0.3) # Subtle UI feel
            options = generate_multiple_options(
                seed_text=input_prompt, 
                next_words=num_words, 
                num_options=num_options, 
                temperature=temp
            )
            
            st.markdown("### 🎯 Output Options:")
            
            # Displaying each option with stylish cards
            for idx, opt in enumerate(options, 1):
                card_html = f"""
                <div class="option-card">
                    <span class="option-number">Option {idx}:</span> {opt}
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)