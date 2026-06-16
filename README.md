# Federated Learning Project

This repository contains a project focused on **Federated Learning**.

## About the Project

The goal of this project is to explore, develop, and test machine learning approaches where data remains distributed across different clients, institutions, or devices. Instead of collecting all data in one central location, models are trained collaboratively while keeping the data local.

Federated Learning can be useful in scenarios where data privacy, security, and data ownership are important.

This code supplements the thesis in [documentation/thesis.pdf](documentation/thesis.pdf).

## Main Objectives

- Set up a basic Federated Learning workflow
- Train machine learning models across distributed data sources
- Compare centralized and federated learning approaches
- Evaluate model performance, privacy, and scalability
- Prepare the project for future experiments and extensions

## Structure

```text
.
├── input/              # Local or simulated datasets
├── output/             # Outputs the training
├── logs/               # Logs of federated learning process
├── src/                # Source code
├── results/            # Evaluation results and outputs
├── docker/             # Docker code
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Flower configuration
└── README.md           # Project documentation
```

## Technologies

Technologies include:

- Python
- TensorFlow
- Flower
- Pandas
- Scikit-learn
- Docker

## Run Without Docker
In this project's directory:

`$ chmod +x scripts/flwr-run/federation-start.sh && ./scripts/flwr-run/federation-start.sh`

## Run With Docker
See [docker/README.md](docker/README.md).

## Run Centralized Machine Learning
In this project's directory:

`$ source .venv/bin/activate && export PYTHONPATH="$(pwd)" && python3 src/centralized/centralized.py &> src/centralized/logs/log.txt &`

## Author

Matej Vesel, intern at Result d.o.o