# Fake News Detection with BERT

## Overview

This project builds a BERT-based fake news classification model to classify news headlines as true or fake. The goal is to apply transformer-based NLP to misinformation detection and evaluate how well a BERT encoder with a trained classification head performs on a labeled news dataset.

This project is a supervised text classification workflow. It is not a production misinformation-detection system.

## Business Problem

Fake news and misleading online content can distort public understanding, influence decision-making, and reduce trust in digital information. Automated text classification can help support content review workflows by flagging potentially unreliable news items for further investigation.

This project demonstrates how transformer-based NLP can be applied to classify news content using labeled examples of true and fake articles.

## Dataset

* **Source:** Kaggle Fake and Real News Dataset
* **Total Records:** 44,898 news articles
* **Files Used:** `True.csv` and `Fake.csv`
* **Target Label:** `label`
* **Label Mapping:** True = 0, Fake = 1
* **Input Used for Modeling:** News headline/title

The original dataset includes article title, text, subject, and publication date. In this notebook, the model is trained on article titles rather than full article body text.

## Methodology

* Loaded true and fake news CSV files
* Assigned text labels and numeric target labels
* Merged true and fake articles into one shuffled dataset
* Checked label balance across true and fake classes
* Split the dataset into train, validation, and test sets using a 70/15/15 split
* Loaded `bert-base-uncased` tokenizer and BERT base model from Hugging Face
* Tokenized news titles with a maximum sequence length of 15 tokens
* Converted tokenized inputs into PyTorch tensors
* Created PyTorch `DataLoader` objects for training and validation
* Froze the pretrained BERT encoder layers
* Trained a custom neural classification head on top of BERT embeddings
* Evaluated model performance using precision, recall, F1-score, and accuracy
* Tested the model on unseen example headlines

## Model Architecture

The model uses a frozen BERT encoder with a custom classifier head:

* `bert-base-uncased` pretrained BERT encoder
* Dropout layer
* Fully connected layer: 768 → 512
* ReLU activation
* Fully connected output layer: 512 → 2
* LogSoftmax output for binary classification

Because the BERT encoder is frozen, the project trains the classification head rather than fully fine-tuning all BERT layers.

## Training Setup

* **Framework:** PyTorch
* **Transformer Library:** Hugging Face Transformers
* **Optimizer:** AdamW
* **Loss Function:** Negative Log-Likelihood Loss
* **Epochs:** 3
* **Batch Size:** 32
* **Train / Validation / Test Split:** 70% / 15% / 15%
* **Device:** GPU when available

## Model Evaluation

| Class            | Precision |   Recall | F1-score |   Support |
| ---------------- | --------: | -------: | -------: | --------: |
| True News        |      0.83 |     0.89 |     0.86 |     3,212 |
| Fake News        |      0.90 |     0.83 |     0.86 |     3,523 |
| **Accuracy**     |           |          | **0.86** | **6,735** |
| **Macro Avg**    |  **0.86** | **0.86** | **0.86** | **6,735** |
| **Weighted Avg** |  **0.86** | **0.86** | **0.86** | **6,735** |

The model achieved **86% test accuracy** and **86% macro F1-score**, with balanced performance across true and fake news classes.

## Unseen Headline Test

The trained model was tested on four unseen news headlines:

```text
Predictions: [1, 1, 0, 0]
```

Using the project label mapping:

```text
0 = True
1 = Fake
```

The sample predictions correctly identified the first two examples as fake and the last two examples as true.

## Web Application

The repository includes a basic Flask application structure for serving predictions through a web interface.
![alt text](images/web_app.png)

## Business Impact

This project shows how transformer-based NLP can support misinformation review workflows by classifying news headlines into true or fake categories. The model can help prioritize content for human review, reduce manual screening effort, and demonstrate how pretrained language models can be adapted to text classification tasks.

## Tech Stack

Python, Pandas, NumPy, PyTorch, Hugging Face Transformers, BERT, Scikit-learn, Matplotlib, Flask, Jupyter Notebook

## Repository Structure

```text
fake-news-detection-bert/
├── images/
├── templates/
├── Fake.csv
├── True.csv
├── Fake_News_Detection_BERT.ipynb
├── app.py
├── tokenizer.pickle
├── requirements.txt
├── myvenv.yml
└── README.md
```

## How to Run

```bash
git clone https://github.com/amit4009/fake-news-detection-bert.git
cd fake-news-detection-bert
pip install -r requirements.txt
jupyter notebook
```

Then open:

```text
Fake_News_Detection_BERT.ipynb
```

## Run the Flask App

```bash
python app.py
```

## Limitations

* The model is trained on news titles only, not full article text.
* The BERT encoder is frozen, so the project does not fully fine-tune all BERT layers.
* The maximum sequence length is 15 tokens, which may truncate longer headlines.
* The dataset may contain publisher/source patterns that the model can learn instead of true misinformation signals.
* The model should not be treated as a production misinformation-detection system without broader validation.
* Real-world fake news detection requires additional context, source reliability signals, temporal validation, and human review.

## Future Improvements

* Train on full article text in addition to titles
* Increase maximum sequence length for richer context
* Fine-tune all or selected BERT layers instead of freezing the encoder
* Add confusion matrix visualization
* Compare BERT with DistilBERT and RoBERTa
* Add SHAP or attention-based interpretability analysis
* Add model monitoring for topic drift and emerging misinformation patterns
* Improve the Flask app with cleaner UI and API endpoint examples
