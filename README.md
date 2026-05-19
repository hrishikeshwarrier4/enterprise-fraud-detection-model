# Enterprise Fraud Detection (EFD)

A comprehensive enterprise fraud detection system built on Palantir Foundry, featuring real-time transaction processing, machine learning-based fraud prediction, and an interactive web dashboard.

## Architecture Overview

```
Kafka Stream → Transaction Processing → Feature Engineering → ML Inference → Ontology → Dashboard
```

### Components

| Component | Directory | Description |
|-----------|-----------|-------------|
| Data Pipeline | `python-transforms/` | Streaming transaction ingestion, Avro-to-Parquet conversion, and ML inference transforms |
| ML Model | `model-training/` | XGBoost + Random Forest ensemble model with threshold optimization |
| Training Data | `training-data-generation/` | Synthetic customer and transaction data generator with fraud labeling |
| Web Dashboard | `efd-website/` | React OSDK application for fraud monitoring and investigation |

---

## Data Pipeline (`python-transforms/`)

The data pipeline handles real-time transaction ingestion and fraud scoring:

- **`transform.py`** — Streaming transaction processor that ingests from Kafka, deduplicates, and enriches transactions
- **`fraud_predictions.py`** — Applies the trained ML model to score transactions in real-time
- **`avro_to_parquet.py`** — Converts raw Avro-encoded Kafka messages to structured Parquet format

### Key Features
- Incremental processing (only new transactions since last build)
- Automatic schema evolution handling
- Configurable fraud threshold scoring

---

## ML Model (`model-training/`)

The fraud detection model uses an ensemble approach combining XGBoost and Random Forest classifiers.

### Training Pipeline

- **`feature_engineering.py`** — Feature extraction and transformation (transaction velocity, amount statistics, time-based features)
- **`model_training.py`** — Model training with hyperparameter tuning and threshold optimization for precision/recall trade-off
- **`generated_adapter.py`** — `FraudDetectionModelAdapter` for Foundry model deployment
- **`run_inference.py`** — Batch inference pipeline
- **`adapters.py`** — Model adapter interfaces

### Model Performance
- Optimized decision threshold balancing false positives vs. missed fraud
- Feature importance tracking for model interpretability
- Version-controlled model artifacts with full lineage

---

## Training Data Generation (`training-data-generation/`)

- **`examples.py`** — Generates synthetic customer profiles and transaction histories with realistic fraud patterns
  - Normal spending behavior simulation
  - Fraud injection with configurable fraud rate
  - Customer demographic generation
  - Temporal patterns (time-of-day, day-of-week effects)

---

## Web Dashboard (`efd-website/`)

A React application built with the Palantir Ontology SDK (OSDK) providing:

- **Real-time fraud alerts** — Live monitoring of flagged transactions
- **Investigation workflows** — Drill-down into suspicious activity patterns
- **Customer profiles** — 360° view of customer behavior and risk scores
- **Analytics dashboards** — Aggregate fraud metrics and trends

### Tech Stack
- React + TypeScript
- Vite build system
- Palantir OSDK for Ontology integration
- Responsive design for desktop and mobile

---

## Getting Started

This project is built and run on [Palantir Foundry](https://www.palantir.com/platforms/foundry/). The source code is exported here for reference and version control.

### Prerequisites
- Palantir Foundry enrollment with:
  - Data Connection (Kafka connector)
  - Transforms (Python)
  - Model Training
  - OSDK application hosting

### Project Structure
```
EFD/
├── README.md
├── python-transforms/       # Data pipeline transforms
│   ├── pipeline.py
│   ├── datasets/
│   │   ├── transform.py
│   │   ├── fraud_predictions.py
│   │   └── avro_to_parquet.py
│   └── ...
├── model-training/          # ML model code
│   ├── model_training.py
│   ├── feature_engineering.py
│   ├── generated_adapter.py
│   ├── run_inference.py
│   └── ...
├── training-data-generation/ # Synthetic data generation
│   └── examples.py
└── efd-website/             # React OSDK dashboard
    ├── package.json
    ├── src/
    │   ├── main.tsx
    │   ├── client.ts
    │   └── ...
    └── ...
```

---

## License

Proprietary. All rights reserved.

---

## Export Details

This repository is automatically synced from Palantir Foundry using a custom export pipeline. Changes should be made in Foundry and re-exported.

**Last export method:** Foundry → GitHub Contents API (automated pipeline)
