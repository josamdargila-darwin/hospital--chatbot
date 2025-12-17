import os
import re
import pandas as pd
import torch
import joblib
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
import gdown

# ===============================
# MODEL & TOKENIZER SETUP
# ===============================
MODEL_DIR = "bert_doctor_classification"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- File IDs from Google Drive ---
FILES = {
    "model.safetensors": "1fCpuig4j-9rVw3arc4imK3vOGgvTa6gL",
    "label_encoder.pkl": "14x-tRD58_MmHuNCzGejyTxiQjWeQuWSK",
    "vocab.txt": "1f8lv8DxQMHFjVtwzoVG4Yxkw9pcPKryM",
    "tokenizer_config.json": "18zHNZB2EJRizWHm1Djk0Sbdqxps51b5m",
    "special_tokens_map.json": "1F9n6co4fchvtTTi0p3Q_9LSYHOlaei6x"
}

# --- Download files if missing ---
for fname, file_id in FILES.items():
    fpath = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(fpath):
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        print(f"⬇️ Downloading {fname}...")
        gdown.download(url, fpath, quiet=False)
    else:
        print(f"✅ {fname} already exists")

# --- Load tokenizer, model, label encoder ---
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

# ===============================
# Load hospital dataset
# ===============================
df = pd.read_csv("Hospital_Information124.csv")

def norm(s):
    return str(s).strip()

for col in df.columns:
    df[col] = df[col].apply(norm)

# ===============================
# SPECIALTY SYNONYMS
# ===============================
SPECIALITY_SYNONYMS = {
    "cardiologist": ["cardio", "heart", "cardiology", "interventional cardiologist", "chief cardiologist"],
    "ent": ["ear nose throat", "otolaryngology", "laryngology", "phonosurgery", "vertigo"],
    "gastroenterologist": ["gastro", "digestive", "hepatology", "pediatric gastroenterology"],
    "gynecologist": ["gyn", "obg", "obstetrics", "fertility"],
    "nephrologist": ["kidney"],
    "neurologist": ["neuro", "brain", "neurovascular", "stroke"],
    "urologist": ["urinary", "genito-urinary"],
    "pulmonologist": ["respiratory", "lungs", "tb"],
    "dermatologist": ["skin"],
    "ophthalmologist": ["eye"],
    "orthopaedician": ["ortho", "orthopedic", "arthroscopy"],
    "oncologist": ["cancer", "medical oncology", "surgical oncology", "clinical oncology"],
    "pathologist": ["pathology"],
    "radiologist": ["radiology", "radiodiagnosis", "interventional radiology"],
    "psychiatrist": ["mental health", "psych","Psychiatrist"],
    "psychologist": ["counseling"],
    "endocrinologist": ["endocrine", "hormone"],
    "general surgeon": ["gen surg", "surgery"],
    "paediatrician": ["kids doctor", "paed", "pediatrician", "child doctor"],
}

# ===============================
# UTILITIES
# ===============================
def detect_intent(user_query):
    inputs = tokenizer(user_query, return_tensors="pt")
    outputs = model(**inputs)
    predicted_id = torch.argmax(outputs.logits).item()
    return label_encoder.inverse_transform([predicted_id])[0]

def match_specialty(user_query):
    q = user_query.lower()
    for canonical, synonyms in SPECIALITY_SYNONYMS.items():
        if any(s in q for s in synonyms) or canonical in q:
            return canonical
    for spec in df["Speciality"].unique():
        if spec.lower() in q:
            return spec.lower()
    return None

def extract_doctor_name(user_query):
    q = user_query.lower()
    for name in df["Doctor Name"].unique():
        if name.lower() in q:
            return name
    return None

def extract_day(user_query):
    q = user_query.lower()
    weekdays = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    if "today" in q:
        return datetime.now().strftime("%A").lower()
    for day in weekdays:
        if day in q:
            return day
    return None

# ===============================
# Doctor Listing / Availability
# ===============================
def list_doctors_by_specialty(specialty):
    rows = df[df["Speciality"].str.contains(specialty, case=False, na=False)]
    if rows.empty:
        return f"No {specialty} doctors found."
    return "\n".join([f"{r['Doctor Name']} - {r['Consultation Time']}" for _, r in rows.iterrows()])

# ===============================
# Appointment Booking
# ===============================
appointments_file = "appointments.csv"
if not os.path.exists(appointments_file):
    pd.DataFrame(columns=["Doctor Name", "Patient Name", "Day", "Time"]).to_csv(appointments_file, index=False)

def book_appointment(doctor_name, patient_name, day, time_slot):
    appt_df = pd.read_csv(appointments_file)
    row = df[df["Doctor Name"].str.contains(re.escape(doctor_name), case=False, na=False)]
    if row.empty:
        return "Doctor not found."
    new_appt = pd.DataFrame([[doctor_name, patient_name, day.capitalize(), time_slot]],
                            columns=["Doctor Name", "Patient Name", "Day", "Time"])
    appt_df = pd.concat([appt_df, new_appt], ignore_index=True)
    appt_df.to_csv(appointments_file, index=False)
    return f"Appointment confirmed with {doctor_name} on {day.capitalize()} at {time_slot}."

# ===============================
# CHATBOT RESPONSE
# ===============================
def chatbot_response(user_query):
    intent = detect_intent(user_query)
    specialty = match_specialty(user_query)
    doctor = extract_doctor_name(user_query)
    day = extract_day(user_query)

    if intent == "find_doctor":
        if specialty:
            return list_doctors_by_specialty(specialty)
        else:
            return "Please specify a specialty."

    if intent == "doctor_availability":
        if specialty and day:
            # List doctors available today
            return list_doctors_by_specialty(specialty)
        elif specialty:
            return list_doctors_by_specialty(specialty)
        else:
            return "Please specify a specialty or doctor."

    if "book appointment" in user_query.lower():
        if doctor and day:
            # Dummy booking message; actual booking via Streamlit form
            return f"Booking available for {doctor} on {day.capitalize()}."
        return "Please specify doctor and day for booking."

    return "I can help you find doctors, check availability, and book appointments."

# ===============================
# Streamlit helper
# ===============================
def run_chatbot_query(query):
    return chatbot_response(query)
