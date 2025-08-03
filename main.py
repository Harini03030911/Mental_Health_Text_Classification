import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
import os

# --- Setup ---
st.set_page_config(page_title="Mental Health Detector", layout="centered")

# --- Custom Enhanced CSS ---
# --- Custom Enhanced CSS ---
# --- Custom Enhanced CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        background-image: url("https://img.freepik.com/free-photo/nature-landscape-with-dreamy-aesthetic-color-year-tones_23-2151393929.jpg?t=st=1746637419~exp=1746641019~hmac=6d36629f9a0ce04e13ce74aa903459026f237f925dfe76f7abd82c1dc56ba460&w=1800");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: #000000;  /* Set all text color to black */
    }

    .stApp {
        background: rgba(255, 255, 255, 0.2);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }

    h1, h2, h3, h4, p, label, .stTextInput label, .stTextArea label, .stButton > button {
        font-weight: bold; /* Make all text bold */
        color: #000000;  /* Ensure the text color is black */
    }

    h1 {
        font-size: 36px; /* Increase font size for headings */
    }

    h2 {
        font-size: 32px; /* Increase font size for subheadings */
    }

    h3, h4 {
        font-size: 28px; /* Adjust font size for smaller headings */
    }

    p, label, .stTextInput label, .stTextArea label {
        font-size: 18px; /* Increase font size for text and labels */
    }

    .stTextInput > div > div > input,
.stTextArea textarea {
    background-color: #1A1A1A;
    color: #FFFFFF;  /* Set input text to white */
    border: 1px solid #F54D3D;
    border-radius: 12px;
    padding: 1rem;
    font-size: 1.2rem; /* Increase font size for inputs */
}


    .stButton > button {
        background-color: #F54D3D;
        color: #000000;  /* Set button text color to black */
        border: none;
        padding: 1em 1.6em;
        border-radius: 12px;
        font-weight: bold;
        transition: 0.3s ease;
        font-size: 18px; /* Increase font size for button */
    }

    .stButton > button:hover {
        background-color: #F2AA4C;
        cursor: pointer;
    }

    .stProgress > div > div > div {
        background-color: #F54D3D;
    }

    .result-container {
        margin-top: 1.5rem;
        background-color: rgba(255, 255, 255, 0.7);
        padding: 1.5rem;
        border-radius: 10px;
        font-size: 18px; /* Increase font size in result container */
    }

    .st-bar-chart {
        border-radius: 10px;
        background-color: #2D2D2D;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)



# --- Load Resources ---
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

model = AutoModelForSequenceClassification.from_pretrained('saved_mental_status_bert')
tokenizer = AutoTokenizer.from_pretrained('saved_mental_status_bert')
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# --- Helper Functions ---
def clean_statement(statement):
    statement = statement.lower()
    statement = re.sub(r'[^\w\s]', '', statement)
    statement = re.sub(r'\d+', '', statement)
    words = [word for word in statement.split() if word not in stop_words]
    return ' '.join(words)

def status_emoji(label):
    return {
        "Anxious": "😟",
        "Neutral": "😐",
        "Depressed": "😞",
        "Happy": "😊"
    }.get(label, "❓")

def detect_anxiety(text):
    cleaned = clean_statement(text)
    inputs = tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=200)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    confidence = torch.max(probs).item()
    predicted_class = torch.argmax(probs, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0], confidence, probs.detach().numpy().flatten()

def get_prob_df(probabilities):
    labels = label_encoder.inverse_transform(range(len(probabilities)))
    return pd.DataFrame({"Class": labels, "Probability": probabilities})

# --- File Upload Functionality ---
def upload_file():
    uploaded_file = st.file_uploader("📄 Upload a text file to analyze", type=["txt", "docx"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        return content, uploaded_file.name, uploaded_file.size
    return None, None, None


# --- Save Past Results ---
def save_result(user_input, predicted_class, confidence):
    if not os.path.exists('past_results.csv'):
        df = pd.DataFrame(columns=["timestamp", "input_text", "predicted_class", "confidence"])
        df.to_csv('past_results.csv', index=False)

    df = pd.read_csv('past_results.csv')
    new_data = pd.DataFrame({"timestamp": [pd.to_datetime("now")], "input_text": [user_input], "predicted_class": [predicted_class], "confidence": [confidence]})
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv('past_results.csv', index=False)

# --- Main Interface ---
# --- Main Interface ---
st.markdown("## 🧠 Mental Health Status Detection")
st.markdown("Enter your thoughts or feelings below. This app will analyze your input and predict a mental health category.")

# Input section with option to upload text file
input_text = st.text_area("✍️ Describe your mental state:", height=200, placeholder="E.g., I’ve been feeling overwhelmed and anxious lately...")
uploaded_text, uploaded_filename, uploaded_size = upload_file()

if uploaded_text:
    input_text = uploaded_text
    readable_size = f"{uploaded_size / 1024:.1f} KB" if uploaded_size < 1024 * 1024 else f"{uploaded_size / (1024 * 1024):.1f} MB"
    st.markdown(f"**🗂️ Uploaded File:** <span style='color: black'>{uploaded_filename} — {readable_size}</span>", unsafe_allow_html=True)


if st.button("🔍 Detect Status"):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter some text before detection.")
    else:
        with st.spinner("Analyzing..."):
            predicted_class, confidence, probs = detect_anxiety(input_text)
            
            st.success(f"🩺 **Predicted Status:** {predicted_class} {status_emoji(predicted_class)}")
            st.write(f"🔎 **Model Confidence:** `{confidence:.2%}`")
            st.progress(confidence)

            st.markdown("### 📊 Class Probabilities")
            prob_df = get_prob_df(probs)
            st.bar_chart(prob_df.set_index("Class"))
            
            # Save results
            save_result(input_text, predicted_class, confidence)
            
            st.markdown("### 🗂️ Past Results:")
            past_results = pd.read_csv('past_results.csv')
            st.dataframe(past_results.tail())

# --- Delete History Button ---
if st.button("🗑️ Clear Past Results"):
    if os.path.exists('past_results.csv'):
        os.remove('past_results.csv')
        st.success("✅ Past results cleared successfully.")
    else:
        st.warning("⚠️ No past results to clear.")
