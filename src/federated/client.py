"""federated: A Flower / TensorFlow app."""
import os
import sys
from pprint import pprint

import keras
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from src.model.model import train_model, evaluate_model

# Flower ClientApp
app = ClientApp()


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local input."""

    # Reset local Tensorflow state
    keras.backend.clear_session()

    # Train the model
    metrics, model = train_model(context.node_config["dataset-filename"])


    # Pack and send the model weights and metrics as a message
    content = RecordDict({"arrays": ArrayRecord(model.get_weights()), "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Reset local Tensorflow state
    keras.backend.clear_session()

    metrics = evaluate_model(msg, context)
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)