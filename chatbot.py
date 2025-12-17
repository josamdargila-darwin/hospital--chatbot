import os
import re
import pandas as pd
import torch
import joblib
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
import gdown

# ===============================                

# SAFE MODEL DOWNLOAD (GitHub)
# ===============================
MODEL_DIR = "bert_doctor_classification"
MODEL_FILE = f"{MODEL_DIR}/model.safetensors"

FILE_ID = "1o6wEKooRrqUymBuNyhpFImWowl4yrbq5"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_FILE):
    print("⬇️ Downloading BERT model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    gdown.download(URL, MODEL_FILE, quiet=False)
else:
    print("✅ BERT model already exists")

# ===============================
# Load hospital dataset
# ===============================
df = pd.read_csv("Hospital_Information124.csv")

def norm(s):
    return str(s).strip().lower()

for col in df.columns:
    df[col] = df[col].apply(norm)

# ===============================
# Load BERT
# ===============================
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
label_encoder = joblib.load(f"{MODEL_DIR}/label_encoder.pkl")

def detect_intent(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    idx = torch.argmax(outputs.logits).item()
    return label_encoder.inverse_transform([idx])[0]

# ===============================
# SPECIALTY SYNONYMS
# ===============================
SPECIALITY_SYNONYMS = {
    "ent": ["ent", "ear nose throat"],
    "cardiologist": ["cardio", "heart"],
    "dermatologist": ["skin"],
    "neurologist": ["neuro", "brain"],
    "nephrologist": ["kidney"],
    "urologist": ["urinary"],
    "pulmonologist": ["lung", "respiratory"],
    "ophthalmologist": ["eye"],
    "orthopaedician": ["ortho"],
    "oncologist": ["cancer"],
    "psychiatrist": ["mental"],
    "psychologist": ["counselling"],
    "endocrinologist": ["hormone"],
    "paediatrician": ["child", "kids"],
    "gynecologist": ["gyn"],
    "gastroenterologist": ["gastro", "digestive"]
}

# ===============================
# FIXED SPECIALTY MATCHER ✅
# ===============================
def match_specialty(query):
    q = query.lower()

    # 1️⃣ synonym first (most reliable)
    for canon, synonyms in SPECIALITY_SYNONYMS.items():
        for s in synonyms:
            if re.search(rf"\b{s}\b", q):
                return canon

    # 2️⃣ exact dataset match
    for spec in df["speciality"].unique():
        if re.search(rf"\b{re.escape(spec)}\b", q):
            return spec

    return None

def extract_doctor_name(query):
    q = query.lower()
    for name in df["doctor name"].unique():
        if name in q:
            return name
    return None

# ===============================
# RESPONSES
# ===============================
def list_doctors_by_specialty(spec):
    rows = df[df["speciality"].str.contains(spec, na=False)]
    if rows.empty:
        return f"No {spec.title()} doctors found."
    return "\n".join(
        f"{r['doctor name'].title()} - {r['consultation time']}"
        for _, r in rows.iterrows()
    )

# ===============================
# CHATBOT CORE
# ===============================
def chatbot_response(user_query):
    intent = detect_intent(user_query)
    specialty = match_specialty(user_query)
    doctor = extract_doctor_name(user_query)

    if intent == "find_doctor":
        if not specialty:
            return "Please specify a specialty."
        return list_doctors_by_specialty(specialty)

    if intent == "doctor_availability" and specialty:
        return list_doctors_by_specialty(specialty)

    return "I can help you find doctors, check availability, and book appointments."

# ===============================
# Streamlit helper
# ===============================
def run_chatbot_query(query):
    return chatbot_response(query)

# ===============================
# CLI MODE
# ===============================
if __name__ == "__main__":
    print("PRS Hospital Chatbot (BERT powered)")
    while True:
        q = input("You: ")
        if q.lower() in ["exit", "quit"]:
            break
        print("Bot:", chatbot_response(q))

