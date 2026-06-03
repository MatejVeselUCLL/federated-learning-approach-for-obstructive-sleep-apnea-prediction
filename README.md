# Federated Learning Project

This repository contains a project focused on **Federated Learning**.

## About the Project

The goal of this project is to explore, develop, and test machine learning approaches where data remains distributed across different clients, institutions, or devices. Instead of collecting all data in one central location, models are trained collaboratively while keeping the data local.

Federated Learning can be useful in scenarios where data privacy, security, and data ownership are important.

## Main Objectives

- Set up a basic Federated Learning workflow
- Train machine learning models across distributed data sources
- Compare centralized and federated learning approaches
- Evaluate model performance, privacy, and scalability
- Prepare the project for future experiments and extensions

## Planned Structure

```text
.
├── data/               # Local or simulated datasets
├── notebooks/          # Jupyter notebooks for experiments
├── src/                # Source code
├── models/             # Saved models or model artifacts
├── results/            # Evaluation results and outputs
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Technologies

Planned technologies may include:

- Python
- PyTorch or TensorFlow
- Flower / FedML / TensorFlow Federated
- Pandas
- Scikit-learn
- Jupyter Notebook

## Status

Project setup in progress.

## Author

Add author information here.

cd ~/matej/federated-learning-poc/ && source .venv/bin/activate && git pull && export PYTHONPATH="$(pwd)" && python3 src/centralized/centralized.py &> ~/matej/federated-learning-poc/src/centralized/logs/log.txt &

cd ~/matej/federated-learning-poc/ && git stash && git pull && chmod +x scripts/flwr-run/federation-start.sh && ./scripts/flwr-run/federation-start.sh