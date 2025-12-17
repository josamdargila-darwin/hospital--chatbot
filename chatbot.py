# chatbot.py
import os
import re
import pandas as pd
import torch
import joblib
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
import gdown

# ===============================
# SAFE MODEL DOWNLOAD
# ===============================
MODEL_DIR = "bert_doctor_classification"
MODEL_FILE = f"{MODEL_DIR}/model.safetensors"

MODEL_URL = "https://drive.google.com/uc?id=1fCpuig4j-9rVw3arc4imK3vOGgvTa6gL"  # BERT model
LABEL_URL = "https://drive.google.com/uc?id=14x-tRD58_MmHuNCzGejyTxiQjWeQuWSK"  # label encoder

if not os.path.exists(MODEL_FILE):
    print("⬇️ Downloading BERT model...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    gdown.download(MODEL_URL, MODEL_FILE, quiet=False)

if not os.path.exists(f"{MODEL_DIR}/label_encoder.pkl"):
    print("⬇️ Downloading label encoder...")
    gdown.download(LABEL_URL, f"{MODEL_DIR}/label_encoder.pkl", quiet=False)

# ===============================
# LOAD HOSPITAL DATASET
# ===============================
df = pd.read_csv("Hospital_Information124.csv")
for col in df.columns:
    df[col] = df[col].astype(str).str.strip()

# ===============================
# LOAD BERT
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
    "cardiologist": ["cardio", "heart", "cardiology"],
    "ent": ["ear nose throat", "otolaryngology", "ent"],
    "gastroenterologist": ["gastro", "digestive", "hepatology"],
    "gynecologist": ["gyn", "obg", "fertility"],
    "nephrologist": ["kidney"],
    "neurologist": ["neuro", "brain", "stroke"],
    "urologist": ["urinary", "genito-urinary"],
    "pulmonologist": ["respiratory", "lungs", "tb"],
    "dermatologist": ["skin"],
    "ophthalmologist": ["eye"],
    "orthopaedician": ["ortho", "orthopedic", "arthroscopy"],
    "oncologist": ["cancer", "oncology"],
    "pathologist": ["pathology"],
    "radiologist": ["radiology", "radiodiagnosis"],
    "psychiatrist": ["mental health", "psych"],
    "psychologist": ["counseling"],
    "endocrinologist": ["endocrine", "hormone"],
    "general surgeon": ["surgery", "gen surg"],
    "paediatrician": ["kids doctor", "paed", "pediatrician", "child doctor"]
}

# ===============================
# HELPER FUNCTIONS
# ===============================
def match_specialty(query):
    q = query.lower()
    for canon, synonyms in SPECIALITY_SYNONYMS.items():
        if any(s in q for s in synonyms) or canon in q:
            return canon
    for spec in df["Speciality"].unique():
        if spec.lower() in q:
            return spec
    return None

def extract_doctor_name(query):
    q = query.lower()
    for name in df["Doctor Name"].unique():
        if name.lower() in q:
            return name
    return None

def extract_day(query):
    q = query.lower()
    today = datetime.now().strftime("%A").lower()
    if "today" in q:
        return today
    days = ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]
    for day in days:
        if day in q:
            return day
    return None

def is_available_on(day_str, available_text):
    if "available all days" in available_text.lower():
        return True
    if "not available on" in available_text.lower():
        blocked = [d.strip().lower() for d in available_text.split(":")[-1].split(",")]
        return day_str.lower() not in blocked
    return True

def list_doctors_by_specialty(spec):
    rows = df[df["Speciality"].str.contains(spec, case=False, na=False)]
    if rows.empty:
        return f"No {spec.title()} doctors found."
    lines = []
    for _, r in rows.iterrows():
        lines.append(f"{r['Doctor Name']} - {r['Consultation Time'].lower()}")
    return "\n".join(lines)

appointments_file = "appointments.csv"
if not os.path.exists(appointments_file):
    pd.DataFrame(columns=["Doctor Name","Patient Name","Day","Time"]).to_csv(appointments_file,index=False)

def book_appointment(doctor_name, patient_name, day, time_slot):
    appt_df = pd.read_csv(appointments_file)
    row = df[df["Doctor Name"].str.contains(re.escape(doctor_name), case=False, na=False)]
    if row.empty:
        return "Doctor not found."
    if not is_available_on(day, row.iloc[0]["Available days"]):
        return f"Sorry, {doctor_name} is not available on {day.capitalize()}."
    new_appt = pd.DataFrame([[doctor_name, patient_name, day.capitalize(), time_slot.lower()]],
                            columns=["Doctor Name","Patient Name","Day","Time"])
    appt_df = pd.concat([appt_df, new_appt], ignore_index=True)
    appt_df.to_csv(appointments_file, index=False)
    return f"Appointment confirmed with {doctor_name} on {day.capitalize()} at {time_slot.lower()}."

# ===============================
# CHATBOT RESPONSE
# ===============================
def chatbot_response(user_query):
    intent = detect_intent(user_query)
    specialty = match_specialty(user_query)
    doctor_name = extract_doctor_name(user_query)
    requested_day = extract_day(user_query)

    # LIST DOCTORS BY SPECIALTY
    if intent=="find_doctor" and specialty:
        return list_doctors_by_specialty(specialty)

    # AVAILABILITY TODAY
    if requested_day:
        if specialty:
            rows = df[df["Speciality"].str.contains(specialty, case=False, na=False)]
            available = []
            for _, r in rows.iterrows():
                if is_available_on(requested_day, r["Available days"]):
                    available.append(f"{r['Doctor Name']} - {r['Consultation Time'].lower()}")
            if not available:
                return f"No {specialty} doctors are available on {requested_day.capitalize()}."
            return "\n".join(available)
        elif doctor_name:
            row = df[df["Doctor Name"].str.contains(doctor_name, case=False, na=False)]
            if row.empty:
                return "Doctor not found."
            available = is_available_on(requested_day, row.iloc[0]["Available days"])
            return f"{doctor_name} is {'available' if available else 'not available'} on {requested_day.capitalize()} - {row.iloc[0]['Consultation Time'].lower()}"

    # BOOK APPOINTMENT
    if "book appointment" in user_query.lower() and doctor_name:
        print("Please enter your name for the appointment:")
        patient_name = input("You: ")
        print("Enter time slot (e.g., 10am to 11am):")
        time_slot = input("You: ")
        day_to_book = requested_day or datetime.now().strftime("%A").lower()
        return book_appointment(doctor_name, patient_name, day_to_book, time_slot)

    return "I can help find doctors, check availability, and book appointments."

# ===============================
# STREAMLIT HELPER
# ===============================
def run_chatbot_query(query):
    return chatbot_response(query)

# ===============================
# CLI MODE
# ===============================
if __name__=="__main__":
    print("PRS Hospital Chatbot (BERT Powered)")
    while True:
        q = input("You: ")
        if q.lower() in ["exit","quit"]:
            break
        print("Bot:", chatbot_response(q))
