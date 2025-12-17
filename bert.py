import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import joblib

# ====================================================
# STEP 1 — CREATE INTENT TRAINING DATA
# ====================================================

df = pd.read_csv("Hospital_Information124.csv")

training_data = []

for _, row in df.iterrows():
    name = row["Doctor Name"]
    specialty = row["Speciality"]

    # find_doctor examples
    training_data.append([f"{specialty.lower()} doctor", "find_doctor"])
    training_data.append([f"show me {specialty.lower()}s", "find_doctor"])

    # availability examples
    training_data.append([f"availability of {name}", "doctor_availability"])
    training_data.append([f"when is {name} available?", "doctor_availability"])

    # contact examples
    training_data.append([f"contact of {name}", "doctor_contact"])
    training_data.append([f"email of {name}", "doctor_contact"])

train_df = pd.DataFrame(training_data, columns=["Question", "Label"])
train_df.to_csv("intents_train.csv", index=False)
print("Saved intents_train.csv")
print(train_df.head())


# ====================================================
# STEP 2 — LOAD INTENT DATA FOR TRAINING
# ====================================================

df = pd.read_csv("intents_train.csv")
print(df.head())

# Encode label IDs
le = LabelEncoder()
df["label_id"] = le.fit_transform(df["Label"])


# ====================================================
# STEP 3 — TOKENIZE
# ====================================================

dataset = Dataset.from_pandas(df[["Question", "label_id"]])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["Question"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.rename_column("label_id", "labels")
tokenized_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "labels"]
)


# ====================================================
# STEP 4 — LOAD MODEL
# ====================================================

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(le.classes_)
)


# ====================================================
# STEP 5 — TRAIN MODEL
# ====================================================

training_args = TrainingArguments(
    output_dir="./bert_doctor_classification",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_dir="./logs",
    logging_steps=50,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

trainer.train()


# ====================================================
# STEP 6 — SAVE MODEL + TOKENIZER + LABEL ENCODER
# ====================================================

model.save_pretrained("./bert_doctor_classification")
tokenizer.save_pretrained("./bert_doctor_classification")
joblib.dump(le, "./bert_doctor_classification/label_encoder.pkl")

print("Model saved successfully!")
