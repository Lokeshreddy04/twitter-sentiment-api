# Twitter Sentiment Analysis API using GRU

A deep learning-based REST API that classifies Twitter posts as **positive** or **negative** using a Gated Recurrent Unit (GRU) neural network built with PyTorch and served through FastAPI.

The application provides single-text and batch sentiment prediction endpoints, with Docker support for containerized deployment.

## Project Overview

This project implements an end-to-end Natural Language Processing (NLP) pipeline, covering:

- Text preprocessing and tokenization.
- Vocabulary construction and sequence encoding.
- GRU-based sentiment classification.
- Model inference using PyTorch.
- REST API development using FastAPI.
- Automated testing using Pytest.
- Containerization using Docker.

## Architecture

```text
                Twitter Text
                     |
                     v
             Text Preprocessing
                     |
                     v
             Tokenization
                     |
                     v
             Vocabulary Encoding
                     |
                     v
             Sequence Padding
                     |
                     v
                GRU Model
                     |
                     v
             Sentiment Prediction
                     |
                     v
              FastAPI Response
```

## Dataset

**Dataset:** Sentiment140

- Original dataset size: 1,600,000 tweets.
- Original sentiment labels: 0 (negative) and 4 (positive).
- Labels converted to binary classes: 0 and 1.
- Conflicting-label and exact normalized duplicates were removed.
- Final dataset: 1,574,062 unique normalized tweets.

The dataset was divided into training, validation, and testing sets using stratified splitting.

## Model Architecture

The project uses a GRU-based neural network for binary sentiment classification.

| Component | Configuration |
|---|---|
| Embedding dimension | 128 |
| GRU hidden size | 128 |
| Dropout | 0.3 |
| Output layer | Linear |
| Output activation | Sigmoid during inference |
| Loss function | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Maximum sequence length | 40 tokens |
| Vocabulary size | Up to 50,000 tokens |

Packed sequences are used to process variable-length tweets efficiently.

## Model Performance

The following results were obtained on the held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 83.82% |
| Precision | 85.34% |
| Recall | 81.78% |
| F1-score | 83.52% |

These results represent performance on the processed Sentiment140 test set. Real-world social media performance may differ because of dataset noise and changes in language usage.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| NLP | Custom tokenizer and vocabulary |
| API framework | FastAPI |
| API server | Uvicorn |
| Data validation | Pydantic |
| Testing | Pytest |
| Containerization | Docker
