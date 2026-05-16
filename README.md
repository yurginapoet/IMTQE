# Interpretable Machine Translation Quality Estimation (MTQE)

## Overview

This project implements an interpretable system for automatic quality estimation of machine translation (MT) without reference translations (reference-free QE).  

The system:
- predicts a sentence-level quality score
- identifies **where errors occur in the translation**
- explains **why the score was assigned**

The approach is based on extracting interpretable features and combining them with a regression model, followed by an explanation layer that produces human-readable error analysis.

---

## Problem Statement

Given:
- Source sentence (EN)
- Machine translation (RU)

Predict:
- A continuous quality score

Additionally:
- Localize errors in the translation (word/token level)
- Provide explanations of detected issues
- Map model signals to interpretable error types

---

## Architecture

The system follows a modular pipeline:


1. SOURCE (EN) + TRANSLATION (RU)
2. Feature Extractors (несколько независимых блоков)
3. Feature Vector (~15–20)
4. Interpretable Model (Ridge / XGBoost)
5. Quality Score (QE)
6. Explanation Layer (SHAP)
7. Оценка, типы ошибок + вклад признаков

Semantic extension:
- the sentence-level feature vector can be extended from `22` to `86`
- `64` extra dimensions come from MiniLM semantic embeddings reduced with PCA
- the training sequence is documented in [architecture_semantic_extension.md](architecture_semantic_extension.md)


---

## Feature Extraction

The system uses independent feature blocks, each responsible for a specific aspect of translation quality.

### 1. Semantic Features (Accuracy)
Measure preservation of meaning between source and translation.

- cosine similarity (cross-lingual embeddings)
- embedding distance

Used for:
- detecting mistranslations
- identifying semantically weak segments

---

### 2. Fluency Features (Language Quality)

Measure how natural the translation is in Russian.

- perplexity
- log-probability

Model: pretrained language model (e.g., ruGPT)

Used for:
- detecting unnatural or grammatically incorrect phrases
- highlighting low-probability word sequences

Span localization sensitivity can be tuned without retraining:
- `IMTQE_SPAN_BAD_THRESHOLD` controls when a token is highlighted as BAD based on `p(BAD)`
- `IMTQE_SPAN_MAJOR_THRESHOLD` controls when a BAD token is marked as `BAD-major`
- defaults: `0.45` and `0.60`

---

### 3. Structural Features

Capture missing or extra content.

- length ratio
- absolute length difference
- token count difference

Used for:
- detecting omissions and additions

---

### 4. Formatting and Numbers

- number match ratio
- digit differences
- punctuation differences

Used for:
- detecting formatting inconsistencies
- identifying incorrect numbers

---

### 5. Named Entities (optional)

- entity overlap ratio
- entity count difference

Used for:
- detecting terminology and entity translation errors

---

## Model

The system uses interpretable regression models:

- Ridge Regression (baseline)
- XGBoost (with SHAP explanations)

Input: feature vector  
Output: sentence-level quality score

---

## Explanation Layer

The explanation module combines:

1. **Global explanation (feature importance)**
2. **Local explanation (per-sentence SHAP values)**
3. **Heuristic error detection for localization**

---

## Error Localization (WHERE)

The system identifies problematic parts of the translation using:

### 1. Token-level signals
- low language model probability (fluency issues)
- rare or unusual words

### 2. Alignment-based signals (optional)
- weak semantic correspondence between source and target tokens

### 3. Heuristics
- unmatched numbers
- missing entities
- abnormal length segments

---

## Error Explanation (WHY)

Model predictions are explained using SHAP values and mapped to error types.


Mapped explanation:

- Meaning is mostly preserved
- Translation is not fluent (high perplexity)
- Possible structural mismatch (length difference)

---

## Error Types

| Signal / Feature       | Error Type            |
|----------------------|----------------------|
| Low similarity        | Accuracy error        |
| High perplexity       | Fluency error         |
| Length mismatch       | Omission / Addition   |
| Number mismatch       | Formatting error      |
| Entity mismatch       | Terminology error     |

---

## Interactive Output

The system is designed to support interactive analysis:

For each sentence:
- quality score
- highlighted problematic tokens
- explanation of each issue
- feature contributions


---

## Datasets

The system is compatible with WMT QE datasets:

- [WMT19 QE (HTER, word-level labels)](https://www.statmt.org/wmt19/qe-task.html)
- [WMT20 QE (Direct Assessment)](https://www.statmt.org/wmt20/quality-estimation-task.html)
- [WMT21 QE](https://www.statmt.org/wmt21/quality-estimation-task.html)
- [WMT22 QE (MQM, error annotations)](https://wmt-qe-task.github.io/wmt-qe-2022/subtasks/task2/)

These datasets can be used for:
- training regression models
- validating error detection

---

## Repository Structure

    qe_system/
    ├── features/
    │    ├── semantic.py
    │    ├── fluency.py
    │    ├── structure.py
    │    ├── entities.py
    │
    ├── model/
    │    ├── ridge.py
    │    ├── xgboost.py
    │
    ├── explain/
    │    ├── shap_analysis.py
    │
    └── pipeline.py

---

## Key Design Principles

- Interpretable by design
- Feature-based (no end-to-end black-box models)
- CPU-friendly (suitable for Google Colab)
- Modular and extensible
- Provides both score and explanation

---

## Summary

The system estimates translation quality using interpretable features and provides detailed explanations of errors.  

Unlike black-box approaches, it explicitly shows:
- where errors occur
- what type of errors they are
- how they affect the final score


Run:   
`poetry run uvicorn src.app.server:app --host 0.0.0.0 --port 8000 --reload`
