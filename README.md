# Human vs LLM Text Disagreement Analysis

This project analyzes cases where human annotators and machine learning models disagreed on whether a given text was written by a human or a large language model (LLM). It extracts interpretable features from text and applies heuristics to explain model failure modes such as humor, irony, grammar variation, or logical inconsistency.

## 📌 Project Goals

- Identify and explain why models fail to classify texts correctly.
- Highlight linguistic patterns that humans can catch but models often miss.
- Visualize key features contributing to disagreement cases.
- Build and evaluate a Random Forest classifier on interpretable features.

## 🧪 Dataset

- **Source:** [Human vs. LLM Text Corpus (Kaggle)](https://www.kaggle.com/datasets/...)
- **Subset Used:** Only disagreement cases (where human ≠ model prediction)
- **Key Columns:**
  - `text`: The text sample.
  - `true_label`: Ground truth (1 = human, 0 = LLM).
  - `model_prediction`: Model's predicted label.
  - `human_judgment`: Human annotator's guess.

## 🔍 Features Extracted

- `readability`: Text readability score (Flesch reading ease).
- `sentiment`: Polarity from TextBlob.
- `pronoun_count`: Number of personal pronouns used.
- `lexical_diversity`: Unique word ratio.
- `failure_reason`: Tagged using heuristic rules:
  - Humor/Irony/Culture
  - Nonstandard Grammar
  - Logical Inconsistency

## 🧠 Modeling

- **Model Used:** `RandomForestClassifier`
- **Inputs:** Extracted interpretable features.
- **Goal:** Predict whether a text is human-written or LLM-generated.

## 📊 Visualizations

- **Boxplots:** Feature distributions by `true_label`.
- **Countplot:** Breakdown of `failure_reason` categories.

## 🛠️ Environment

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 💾 Code to Download the Dataset from Kaggle

Make sure you have your Kaggle API credentials (`kaggle.json`) ready before running the following code.

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("starblasters8/human-vs-llm-text-corpus")

print("Path to dataset files:", path)
```

Or use the Kaggle CLI method below:

```python
import os
import shutil

# Create the Kaggle config directory
os.makedirs("/root/.kaggle", exist_ok=True)

# Replace with your actual kaggle.json path
shutil.copy("/content/kaggle (1).json", "/root/.kaggle/kaggle.json")

# Set correct permissions
os.chmod("/root/.kaggle/kaggle.json", 0o600)

# Now download the dataset from Kaggle
!kaggle datasets download -d starblasters8/human-vs-llm-text-corpus --unzip

# Verify the files
print("Downloaded files:", os.listdir())
```
