# Clause Boundary Detection V2

Automatic clause boundary detection using the **Universal Dependencies (UD) English Web Treebank** dataset. This project implements and compares three approaches: Rule-based, CRF, and BiLSTM.

## 🎯 What is Clause Boundary Detection?

A **clause** is a grammatical unit containing a subject and predicate. Clause boundary detection identifies where clauses start and end in a sentence.

**Example:**
```
Input:  "When the rain stopped, we went outside and played."
Output: [When the rain stopped] [we went outside] [played]
```

## 📁 Project Structure

```
├── data/UD_English-EWT/       # UD dataset (.conllu files)
├── src/
│   ├── data_loader.py         # CoNLL-U parser
│   ├── clause_labeler.py      # BIO label generator
│   ├── feature_extractor.py   # CRF feature engineering
│   ├── rule_based.py          # Rule-based detector
│   ├── crf_model.py           # CRF model
│   ├── bilstm_model.py        # BiLSTM model
│   └── evaluation.py          # Metrics & analysis
├── notebooks/                 # Jupyter notebooks
├── survey/                    # Literature review
├── results/                   # Saved models & metrics
├── app.py                     # Streamlit visualization app
├── train.py                   # Training script
└── requirements.txt           # Dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Download Dataset
Download UD English-EWT from [GitHub](https://github.com/UniversalDependencies/UD_English-EWT) and place files in `data/UD_English-EWT/`.

### 3. Train Models
```bash
python train.py
```

### 4. Launch Visualization App
```bash
streamlit run app.py
```

## 🔧 Models

| Model | Approach | Description |
|-------|----------|-------------|
| Rule-based | Linguistic rules | Uses spaCy dependency parsing with clause-indicating relations |
| CRF | Machine Learning | sklearn-crfsuite with POS, dependency, and context features |
| BiLSTM | Deep Learning | PyTorch BiLSTM with word + POS embeddings |

## 📊 Evaluation Metrics

- **Token-level**: Precision, Recall, F1 per BIO tag
- **Clause-level**: Exact match Precision, Recall, F1
- **Error Analysis**: By sentence length, clause type, nesting depth

## 📚 Dataset

**Universal Dependencies English Web Treebank (EWT)**
- ~16,000 annotated sentences
- Full syntactic dependency annotations
- [Website](https://universaldependencies.org/)

## 🛠️ Technologies

- Python 3.12
- spaCy — NLP & dependency parsing
- sklearn-crfsuite — CRF sequence labeling
- PyTorch — Deep learning
- Streamlit — Interactive visualization
- scikit-learn — Evaluation metrics
