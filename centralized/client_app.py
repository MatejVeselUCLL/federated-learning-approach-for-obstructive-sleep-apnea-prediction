"""tfexample: A Flower / TensorFlow app."""
import os
import sys
from pprint import pprint

import keras
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from config.config_rok_factor_1_brez_C_v4_0 import config
from task_apnea import train_model


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def train():
    # Train the model
    settings = config["settings"]
    general_parameters = config["general_parameters"]
    hyper_parameters = config["hyper_parameters"]
    print("hello")
    metrics, model = train_model(settings, general_parameters, hyper_parameters)


    # Pack and send the model weights and metrics as a message
    print("metricsss")
    pprint(metrics)

if __name__ == "__main__":
    print("helloooo")
    train()
