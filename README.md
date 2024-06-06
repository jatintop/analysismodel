# Fault Description to Complaint Classification using BERT

This project aims to classify automotive fault descriptions into predefined complaint categories using a BERT-based model. It also extracts root causes and failure modes from the fault descriptions.

## Project Overview

1. **Data Preprocessing**:
    - Extract keywords from fault descriptions.
    - Identify root causes and failure modes.

2. **Model Training**:
    - Tokenize the input data using BERT tokenizer.
    - Train a BERT model for sequence classification on the preprocessed data.
    - Save the trained model and tokenizer.

3. **Prediction**:
    - Load the trained model and tokenizer.
    - Predict complaints for new fault descriptions.
    - Save predictions to an Excel file.

4. **Online Learning**:
    - Retrain the model with new predictions to improve accuracy over time.

## Installation

### Prerequisites

- Python 3.6+
- PyTorch
- Transformers (Hugging Face)
- SpaCy
- NLTK
- pandas
- openpyxl

### Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/jatintop/fault-description-classification.git
    cd fault-description-classification
    ```

2. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    python -m spacy download en_core_web_lg
    ```

3. **Set up NLTK stemmer**:
    ```python
    import nltk
    nltk.download('snowball_data')
    ```

## Usage

### Data Preprocessing

The script preprocesses fault descriptions by extracting keywords, root causes, and failure modes.

```python
import spacy
from nltk.stem.snowball import SnowballStemmer

# Initialize SpaCy and Snowball Stemmer
nlp = spacy.load("en_core_web_lg")
stemmer = SnowballStemmer(language='english')

def extract_keywords(text):
    doc = nlp(text)
    keywords = [stemmer.stem(token.text) for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(keywords)

def extract_root_cause(text):
    # Root cause extraction logic
    pass

def extract_failure_mode(text):
    # Failure mode extraction logic
    pass
```

### Model Training

Train a BERT model using fault descriptions and complaint categories.

```python
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Tokenize and encode data
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
encodings = tokenizer(keywords, truncation=True, padding=True, max_length=128, return_tensors='pt')
labels = torch.tensor([label_dict[label] for label in complaints])

# Split data into training and validation sets
# Create datasets and train the model
```

### Prediction

Load the trained model and tokenizer to predict complaints for new fault descriptions.

```python
def predict_complaint(FaultD):
    keywords = extract_keywords(FaultD)
    inputs = tokenizer(keywords, return_tensors='pt', truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    complaint = index_to_label[predicted_class]
    return complaint

# Load new data and predict complaints
```

### Online Learning

Retrain the model with new predictions to adapt to new data.

```python
def retrain_model(new_data_df):
    # Combine new data with existing training data
    # Retrain the model
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
```

### Saving Results

Save the predictions along with root causes and failure modes to an Excel file.

```python
new_df['Prediction'] = new_df['FaultD'].apply(predict_complaint)
new_df['Root Cause'] = new_df['FaultD'].apply(extract_root_cause)
new_df['Failure Mode'] = new_df['FaultD'].apply(extract_failure_mode)
new_df.to_excel(output_file_path, index=False)
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- [SpaCy](https://spacy.io/)
- [NLTK](https://www.nltk.org/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)

## Contact

For any queries or suggestions, please contact jatintopakar@yahoo.com.
