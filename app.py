import pandas as pd
import spacy
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import json

# Initialize SpaCy and Snowball Stemmer
nlp = spacy.load("en_core_web_lg")
stemmer = SnowballStemmer(language='english')

def extract_keywords(text):
    doc = nlp(text)
    keywords = [stemmer.stem(token.text) for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(keywords)

def extract_root_cause(text):
    keywords = [
        "engine", "transmission", "brake", "electrical", "battery", "cooling", "system", "module", "sensor",
        "oil", "seepage", "leakage", "seep", "leak", "loose", "rub", "gap", "flush", "panel", "lid", "door", 
        "seal", "mark", "cover", "screw", "lock", "adjust", "dirty", "dent", "scratches", "noise", 
        "plug", "clip", "harness", "mount", "striker", "dashboard", "mirror", "fender", "trim", "beeding", 
        "handle", "lamp", "foot", "stain", "valve", "hose", "tank", "decal", "plate", "cap", "shaft", "joint", 
        "axle", "reflector", "bracket", "buckle", "hinge", "bushing", "spring", "radiator", "filter", "extinguisher", 
        "sump", "shelf", "visor", "switch", "steering", "shackle", "clamp", "knob", "cable", "connector", "speaker", 
        "filling", "camera", "pin", "shroud", "paint"
    ]
    root_cause = []
    for keyword in keywords:
        if keyword in text.lower():
            root_cause.append(keyword)
    return ', '.join(root_cause)

def extract_failure_mode(text):
    keywords = [
        "leak", "seepage", "failure", "overheat", "malfunction", "noise", "vibration", "stall", "shutdown", "error",
        "scratches", "marks", "scratch", "stuck", "adjusted", "loose", "disconnected", "gap", "flush", "dirty", "dent",
        "lock", "unlock", "missing", "hand loose", "rubbing", "spilage", "not working", "functioning", "code", 
        "fault", "hitting", "not seated", "not flushed", "adjusted properly", "gap noticed", "seated properly",
        "crack", "twisted", "not locked", "not adjusted", "not locking properly", "screches", "damage", "not connected"
    ]
    failure_mode = []
    for keyword in keywords:
        if keyword in text.lower():
            failure_mode.append(keyword)
    return ', '.join(failure_mode)

# Load the training Excel sheet
train_file_path = r'C:\Users\jatin\OneDrive\Documents\[2] Python\2 Volvo\FaultD\Desc2.xlsx'
df = pd.read_excel(train_file_path)

# Verify columns
print("Columns in training file:", df.columns)

# Extract issue descriptions and complaints
issue_descriptions = df['FaultD'].tolist()
complaints = df['Comp'].tolist()

print("Sample issue descriptions:", issue_descriptions[:5])
print("Sample complaints:", complaints[:5])

# Apply keyword extraction to all issue descriptions
keywords = [extract_keywords(description) for description in issue_descriptions]

print("Sample keywords:", keywords[:5])

# Initialize the BERT tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize the input data
encodings = tokenizer(keywords, truncation=True, padding=True, max_length=128, return_tensors='pt')

# Convert labels to tensor
label_dict = {label: i for i, label in enumerate(set(complaints))}
labels = torch.tensor([label_dict[label] for label in complaints])

# Split the data into training and testing sets
train_idx, val_idx = train_test_split(list(range(len(labels))), test_size=0.2, random_state=42)

train_encodings = {key: val[train_idx] for key, val in encodings.items()}
val_encodings = {key: val[val_idx] for key, val in encodings.items()}
train_labels = labels[train_idx]
val_labels = labels[val_idx]

print("Length of train_encodings:", len(train_encodings['input_ids']))
print("Length of val_encodings:", len(val_encodings['input_ids']))
print("Length of train_labels:", len(train_labels))
print("Length of val_labels:", len(val_labels))

class ComplaintDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = ComplaintDataset(train_encodings, train_labels)
val_dataset = ComplaintDataset(val_encodings, val_labels)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_dict))

output_dir = r'C:\Users\jatin\OneDrive\Documents\[2] Python\2 Volvo\FaultD\Results'

# Set up training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.05,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Save the label dictionary
with open(f'{output_dir}/label_dict.json', 'w') as f:
    json.dump(label_dict, f)

print("Model, tokenizer, and label dictionary saved successfully.")

# Load the trained model and tokenizer for predictions
tokenizer = BertTokenizerFast.from_pretrained(output_dir)
model = BertForSequenceClassification.from_pretrained(output_dir)

# Load the label dictionary
with open(f'{output_dir}/label_dict.json', 'r') as f:
    label_dict = json.load(f)

# Reverse the label_dict to map indices to labels
index_to_label = {v: k for k, v in label_dict.items()}

# Define the function to predict complaint for new issue description
def predict_complaint(FaultD):
    keywords = extract_keywords(FaultD)
    inputs = tokenizer(keywords, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    complaint = index_to_label[predicted_class]
    return complaint

# Load the new Excel file with issue descriptions
new_file_path = r'C:\Users\jatin\OneDrive\Documents\[2] Python\2 Volvo\FaultD\Book1.xlsx'
new_df = pd.read_excel(new_file_path)

# Predict complaints for each issue in the new file
new_df['Prediction'] = new_df['FaultD'].apply(predict_complaint)

# Add the Root Cause and Failure Mode columns
new_df['Root Cause'] = new_df['FaultD'].apply(extract_root_cause)
new_df['Failure Mode'] = new_df['FaultD'].apply(extract_failure_mode)

# Save the updated DataFrame to a new Excel file
output_file_path = r'C:\Users\jatin\OneDrive\Documents\[2] Python\2 Volvo\FaultD\PredictedIssues.xlsx'
new_df.to_excel(output_file_path, index=False)

print("Predictions saved to:", output_file_path)

# Online learning: Retrain the model with new predictions
def retrain_model(new_data_df):
    new_issue_descriptions = new_data_df['FaultD'].tolist()
    new_complaints = new_data_df['Prediction'].tolist()

    # Apply keyword extraction to new issue descriptions
    new_keywords = [extract_keywords(description) for description in new_issue_descriptions]

    # Tokenize the new data
    new_encodings = tokenizer(new_keywords, truncation=True, padding=True, max_length=128, return_tensors='pt')

    # Convert new labels to tensor
    new_labels = torch.tensor([label_dict[label] for label in new_complaints])

    # Combine new data with existing training data
    combined_encodings = {key: torch.cat((val, new_encodings[key]), dim=0) for key, val in train_encodings.items()}
    combined_labels = torch.cat((train_labels, new_labels), dim=0)

    # Create new dataset
    combined_dataset = ComplaintDataset(combined_encodings, combined_labels)

    # Update trainer with combined dataset
    trainer.train_dataset = combined_dataset

    # Retrain the model
    print("Retraining the model with new data...")
    trainer.train()

    # Save the updated model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model retrained with new data.")

# Retrain the model with the predictions
retrain_model(new_df)

print("Model retrained with new predictions.")
