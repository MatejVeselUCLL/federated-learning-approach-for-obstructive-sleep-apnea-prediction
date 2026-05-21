"""tfexample: A Flower / TensorFlow app."""
from pprint import pprint

import keras
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from tfexample.config.config_rok_factor_1_brez_C_v4_0 import config
from tfexample.task_apnea import train_model

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    # Reset local Tensorflow state
    keras.backend.clear_session()

    # Train the model
    settings = config["settings"]
    general_parameters = config["general_parameters"]
    hyper_parameters = config["hyper_parameters"]
    metrics, model = train_model(settings, general_parameters, hyper_parameters)


    # Pack and send the model weights and metrics as a message
    content = RecordDict({"arrays": ArrayRecord(model.get_weights()), "metrics": MetricRecord(metrics)})
    print("MESSAGEEE")
    pprint(Message(content=content, reply_to=msg))
    return Message(content=content, reply_to=msg)


# @app.evaluate()
# def evaluate(msg: Message, context: Context):
#     """Evaluate the model on local data."""
#
#     # Reset local Tensorflow state
#     keras.backend.clear_session()
#
#     # Train the model
#     settings = config["settings"]
#     general_parameters = config["general_parameters"]
#     hyper_parameters = config["hyper_parameters"]
#     metrics, model = train_model(settings, general_parameters, hyper_parameters)
#
#     # Pack and send the model weights and metrics as a message
#     content = RecordDict({"arrays": ArrayRecord(model.get_weights()), "metrics": MetricRecord(metrics)})
#     return Message(content=content, reply_to=msg)
#
#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     _, _, x_test, y_test = load_data(partition_id, num_partitions)
#
#     # Load the model
#     model = load_model(context.run_config["learning-rate"])
#     model.set_weights(msg.content["arrays"].to_numpy_ndarrays())
#
#     # Evaluate the model
#     eval_loss, eval_acc = model.evaluate(x_test, y_test, verbose=0)
#
#     # Pack and send the model weights and metrics as a message
#     metrics = {
#         "eval_acc": eval_acc,
#         "eval_loss": eval_loss,
#         "num-examples": len(x_test),
#     }
#     content = RecordDict({"metrics": MetricRecord(metrics)})
#     return Message(content=content, reply_to=msg)
