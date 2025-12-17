import re
import pandas as pd
import torch
import joblib
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
import gdown

#file_id = "1-eUWEBYaDUoAySlAHkoIyIsIllinlu5Z"
#url = f"https://drive.google.com/uc?id={file_id}"
#output = "model.safetensors"

# Download if not already present
# gdown.download(url, output, quiet=False)



# ===============================
# Load hospital dataset
# ===============================
df = pd.read_csv("Hospital_Information124.csv")

# Normalize whitespace and case-insensitive helpers
def norm(s):
    return str(s).strip()

for col in ["Doctor Name", "Speciality", "Professional Degree", "Consultation Time",
            "Available days", "Contact", "Email", "Location"]:
    df[col] = df[col].apply(norm)

# ===============================
# Load trained model + tokenizer + label encoder
# ===============================
model_path = "./bert_doctor_classification"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
label_encoder = joblib.load(model_path + "/label_encoder.pkl")

def detect_intent(user_query):
    inputs = tokenizer(user_query, return_tensors="pt")
    outputs = model(**inputs)
    predicted_id = torch.argmax(outputs.logits).item()
    return label_encoder.inverse_transform([predicted_id])[0]

# ===============================
# Synonyms & Column Aliases
# ===============================
SPECIALITY_SYNONYMS = {
    "cardiologist": ["cardio", "heart", "cardiology", "interventional cardiologist", "chief cardiologist", "caardio"],
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

COLUMN_ALIASES = {
    "Doctor Name": ["doctor name", "name"],
    "Speciality": ["speciality", "specialty", "specialisation", "specialization"],
    "Consultation Time": ["consultation time", "timings", "hours", "availability", "time"],
    "Available days": ["consultation days", "days", "available days", "availability days", "day"],
    "Contact": ["contact", "phone", "mobile", "number", "appointment", "book appointment"],
    "Email": ["email", "mail"],
    "Location": ["location", "address"],
    "Professional Degree": ["degree", "qualification", "qualifications"],
}

DAY_SYNONYMS = {
    "monday": ["mon"],
    "tuesday": ["tue", "tues"],
    "wednesday": ["wed"],
    "thursday": ["thu", "thur", "thurs"],
    "friday": ["fri"],
    "saturday": ["sat"],
    "sunday": ["sun"]
}
WEEKDAYS = list(DAY_SYNONYMS.keys())

# ===============================
# Utilities
# ===============================
def match_specialty(user_query):
    q = user_query.lower()
    for spec in df["Speciality"].unique():
        if spec and spec.lower() in q:
            return spec
    for canonical, synonyms in SPECIALITY_SYNONYMS.items():
        if any(syn in q for syn in synonyms) or canonical in q:
            return canonical
    return None

def map_field_alias(user_query):
    q = user_query.lower()
    for col, aliases in COLUMN_ALIASES.items():
        if any(alias in q for alias in aliases):
            return col
    return None

def extract_doctor_name(user_query):
    q = user_query.lower()
    for name in df["Doctor Name"].unique():
        if name.lower() in q:
            return name
    tokens = [t for t in re.findall(r"[A-Za-z\.]+", user_query)]
    for name in df["Doctor Name"].unique():
        parts = [p.lower() for p in re.findall(r"[A-Za-z\.]+", name)]
        if all(p in [t.lower() for t in tokens] for p in parts if len(p) > 1):
            return name
    return None

def extract_day(user_query):
    q = user_query.lower()
    if "today" in q:
        return datetime.now().strftime("%A").lower()
    for day in WEEKDAYS:
        if day in q or any(s in q for s in DAY_SYNONYMS[day]):
            return day
    return None

def is_available_on(day_str, available_days_text):
    txt = (available_days_text or "").strip().lower()
    if not day_str:
        return None
    if "available all days" in txt:
        return True
    if "not available on" in txt:
        blocked = [d.strip().lower() for d in re.split(r"[:,]", txt)[-1].split(",")]
        return day_str.lower() not in blocked
    return True

# ===============================
# Response builders
# ===============================
def list_all_doctors():
    lines = [f"{row['Doctor Name']} - {row['Speciality']}" for _, row in df.iterrows()]
    return "\n".join(lines)

def list_doctors_by_specialty(specialty):
    pattern = re.escape(specialty)
    results = df[df["Speciality"].str.contains(pattern, case=False, na=False)]
    if results.empty:
        return f"No {specialty} found."
    return "\n".join([f"{row['Doctor Name']} - {row['Consultation Time']}" for _, row in results.iterrows()])

def get_field_for_doctor(name, field):
    row = df[df["Doctor Name"].str.contains(re.escape(name), case=False, na=False)]
    if row.empty:
        return "Doctor not found."
    val = row.iloc[0][field]
    label = field
    if field == "Professional Degree":
        label = "Degree"
    elif field == "Consultation Time":
        label = "Timings"
    elif field == "Available days":
        label = "Available days"
    return f"{name} {label}: {val}"

def get_contact_block(name):
    row = df[df["Doctor Name"].str.contains(re.escape(name), case=False, na=False)]
    if row.empty:
        return "Doctor not found."
    return f"Contact: {row.iloc[0]['Contact']}, Email: {row.iloc[0]['Email']}"

def availability_on_day_for_doctor(name, day):
    row = df[df["Doctor Name"].str.contains(re.escape(name), case=False, na=False)]
    if row.empty:
        return "Doctor not found."
    ok = is_available_on(day, row.iloc[0]["Available days"])
    return f"{name} is {'available' if ok else 'not available'} on {day.capitalize()}."

def availability_on_day_for_specialty(specialty, day):
    pattern = re.escape(specialty)
    rows = df[df["Speciality"].str.contains(pattern, case=False, na=False)]
    if rows.empty:
        return f"No {specialty} found."
    lines = []
    for _, row in rows.iterrows():
        ok = is_available_on(day, row["Available days"])
        if ok:
            lines.append(f"{row['Doctor Name']} - {row['Consultation Time']}")
    if not lines:
        return f"No {specialty} doctors are available on {day.capitalize()}."
    return "\n".join(lines)

# ===============================
# Appointment Booking
# ===============================
appointments_file = "appointments.csv"
try:
    pd.read_csv(appointments_file)
except FileNotFoundError:
    pd.DataFrame(columns=["Doctor Name", "Patient Name", "Day", "Time"]).to_csv(appointments_file, index=False)

def book_appointment(doctor_name, patient_name, day, time_slot):
    appt_df = pd.read_csv(appointments_file)
    row = df[df["Doctor Name"].str.contains(re.escape(doctor_name), case=False, na=False)]
    if row.empty:
        return "Doctor not found."
    if not is_available_on(day, row.iloc[0]["Available days"]):
        return f"Sorry, {doctor_name} is not available on {day.capitalize()}."
    new_appt = pd.DataFrame([[doctor_name, patient_name, day.capitalize(), time_slot]],
                            columns=["Doctor Name", "Patient Name", "Day", "Time"])
    appt_df = pd.concat([appt_df, new_appt], ignore_index=True)
    appt_df.to_csv(appointments_file, index=False)
    return f"Appointment confirmed with {doctor_name} on {day.capitalize()} at {time_slot}."

# ===============================
# Main chatbot logic
# ===============================
def chatbot_response(user_query):
    intent = detect_intent(user_query)
    q = user_query.lower()

    # LIST ALL DOCTORS
    if "all doctors" in q or "doctors in the hospital" in q or "list doctors" in q or "show doctors" in q:
        return list_all_doctors()

    wants_contact = "contact" in q
    requested_field = map_field_alias(user_query)
    requested_day = extract_day(user_query)
    doctor_name = extract_doctor_name(user_query)
    specialty = match_specialty(user_query)

    # BOOK APPOINTMENT
    if "book appointment" in q:
        if doctor_name:
            print("Chatbot: Please enter your name for the appointment:")
            patient_name = input("You: ")

            while True:
                print(f"Chatbot: Please enter desired time slot between 9AM and 6PM for {requested_day or 'any day'} (e.g., 10AM to 11AM):")
                time_slot = input("You: ")
                try:
                    start_str, end_str = time_slot.split("to")
                    start_hour = datetime.strptime(start_str.strip().upper(), "%I%p")
                    end_hour = datetime.strptime(end_str.strip().upper(), "%I%p")
                    earliest = datetime.strptime("9AM", "%I%p")
                    latest = datetime.strptime("6PM", "%I%p")
                    if start_hour < earliest or end_hour > latest:
                        print("Chatbot: Appointments can only be booked between 9AM and 6PM. Try again.")
                        continue
                except:
                    print("Chatbot: Invalid time format. Use format like '10AM to 11AM'. Try again.")
                    continue
                break

            day_to_book = requested_day or datetime.now().strftime("%A").lower()
            return book_appointment(doctor_name, patient_name, day_to_book, time_slot)
        else:
            return "Please specify a doctor to book the appointment."

    # CONTACT
    if wants_contact and doctor_name:
        return get_contact_block(doctor_name)

    # FIELD
    if requested_field and doctor_name:
        field = requested_field
        alias_to_real = {"Degree": "Professional Degree", "Consultation Days": "Available days"}
        field = alias_to_real.get(field, field)
        if field not in df.columns:
            return "Sorry, I couldn't map the requested field."
        return get_field_for_doctor(doctor_name, field)

    # DAY AVAILABILITY
    if requested_day:
        if doctor_name:
            return availability_on_day_for_doctor(doctor_name, requested_day)
        if specialty:
            return availability_on_day_for_specialty(specialty, requested_day)
        return "Please specify a doctor or specialty for day-based availability."

    # INTENT FALLBACK
    if intent == "find_doctor":
        if not specialty:
            return "Please specify a specialty."
        return list_doctors_by_specialty(specialty)
    elif intent == "doctor_availability":
        if doctor_name:
            row = df[df["Doctor Name"].str.contains(re.escape(doctor_name), case=False, na=False)]
            if row.empty:
                return "Doctor not found."
            return f"{doctor_name} is available {row.iloc[0]['Consultation Time']} ({row.iloc[0]['Available days']})."
        if specialty:
            return list_doctors_by_specialty(specialty)
        return "Please specify a doctor or specialty for availability."
    elif intent == "doctor_contact":
        if doctor_name:
            return get_contact_block(doctor_name)
        if specialty:
            rows = df[df["Speciality"].str.contains(re.escape(specialty), case=False, na=False)]
            if rows.empty:
                return f"No {specialty} found."
            lines = [f"{row['Doctor Name']}: {row['Contact']} | {row['Email']}" for _, row in rows.iterrows()]
            return "\n".join(lines)

    if doctor_name and "degree" in q:
        return get_field_for_doctor(doctor_name, "Professional Degree")

    return "I can help with doctor search, availability, degree, contact, timings, email, listing doctors, and booking appointments."

# ===============================
# Run chatbot loop
# ===============================
if __name__ == "__main__":
    print("Welcome to prs Hospital! Type 'quit' to exit.\n")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(user_query)
        print("Chatbot:")
        print(response)

# Streamlit helper
def run_chatbot_query(query):
    return chatbot_response(query)
