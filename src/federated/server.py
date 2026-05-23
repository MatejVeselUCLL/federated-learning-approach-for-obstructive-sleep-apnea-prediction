"""federated: A Flower / TensorFlow app."""
from pprint import pprint

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from src.model.config.config_rok_factor_1_brez_C_v4_0 import config
from src.model.model import build_inception_lstm_dual_branch

# Create the ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Load config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    settings = config["settings"]
    general_parameters = config["general_parameters"]
    hyper_parameters = config["hyper_parameters"]

    # Load initial model
    model = build_inception_lstm_dual_branch(
        feature1_len=general_parameters["WINDOW"],
        feature2_len=general_parameters["WINDOW"],
        lstm_units=tuple(hyper_parameters["lstm_units"]),
        depth=hyper_parameters["depth"],
        nb_filters=hyper_parameters["nb_filters"],
        dropout_dense=tuple(hyper_parameters["dropout_dense"]),
        dropout_lstm=hyper_parameters["dropout_lstm"],
        l2_reg=hyper_parameters["l2_reg"]
    )

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=hyper_parameters["learning_rate"]),
        loss='mean_squared_error',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    arrays = ArrayRecord(model.get_weights())

    # Define and start FedAvg strategy
    strategy = FedAvg(
        fraction_train=fraction_train,
    )

    pprint(num_rounds)
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save the final model
    ndarrays = result.arrays.to_numpy_ndarrays()
    final_model_name = "final_model.keras"
    print(f"Saving final model to disk as {final_model_name}...")

    # print("000000000000000000000")
    # pprint(result)
    # print("sdfsdf")
    # pprint(result.arrays)
    # print("AAAAAAAAAAAAAAAAAAAAAA")
    # print(len(model.get_weights()))
    # print("BBBBBBBBBBBBBBBBBBBBBB")
    # print(len(ndarrays))

    model.set_weights(ndarrays)
    model.save(final_model_name)
