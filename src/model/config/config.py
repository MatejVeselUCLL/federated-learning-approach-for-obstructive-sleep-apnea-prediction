factor = 1
# import git

# repo = git.Repo('', search_parent_directories=True)


settings = {
    "config_name": "config_strict_good",
    "experiment_name": "Hyperparameter Tuning Omop 1s",
    "study_name": f"Objective: Val F1 Score (Variance to Rok: {factor}) Brez C10 (1) in C28 (20)",
    "output_path": f"output", # Relative to tuning.py.
    "input_path": f"input", # Relative to tuning.py.
    "dataset_filename": "ml_train_dataset_from_omop_1s.csv",
    "log_file": "log.txt", # Do not change this.
    "objective_metric": "true_val_f1_score", # For possible options, see final_metrics variable in train_model.py.
    "objective_direction": "maximize", # Either 'maximize' or 'minimize'
    "epochs": 255,
    "trials": 1
}

general_parameters = {
    "FEATURES": [ "value_as_number_spo2", "value_as_number_hr" ],
    "TARGET": "value_as_number_apnea",
    "ID_COLUMN": "person_id",
    "TEST_PERSON_IDS": [
        # 'C3'
        'C15', 'C7', 'C13', 'D28', 'D33', 'ND5', 'ND1',
        'C29', 'C11', 'D25', 'D5', 'ND7',
        'C21', 'C32', 'D8', 'D13', 'ND2'
    ],
    "TEST_PERSON_IDS_h1": ['C15', 'C7', 'C13', 'D28', 'D33', 'ND5', 'ND1'],
    "TEST_PERSON_IDS_h2": ['C29', 'C11', 'D25', 'D5', 'ND7'],
    "TEST_PERSON_IDS_h3": ['C21', 'C32', 'D8', 'D13', 'ND2'],
    "WINDOW": 8, # Window size: 5 minutes (4s x 75 = 300s = 5 min)
    "STEP": 5 # Step size: size of a step to take on moving Window
}

def make_positive(x, bo):
    if x <= 0:
        return 1 * bo
    else:
        return x

v1 = 5 * factor
v2 = 0.2 * (factor * 0.1)
v3 = 0.005 * (factor * 0.001)
v4 = 0.00005 * (factor * 0.00001)

hyper_parameters = {
    "window_size": general_parameters["WINDOW"],
    "lstm_units": [64, 48, 24],
    "depth": 16,
    "nb_filters": 64,
    "dropout_dense": [0.0, 0.1],
    "dropout_lstm": 0.0,
    "l2_reg": 0.0,
    "learning_rate": 0.001,
    "epochs": settings["epochs"],
    "batch_size": 128,
    "validation_split": 0.2,
    "prediction_threshold": 0.4,
}

# hyper_parameters = {
#     "window_size": 8,
#     "lstm_units": [64, 32, 16],
#     "depth": 30,
#     "nb_filters": 64,
#     "dropout_dense": [0.0, 0.0],
#     "dropout_lstm": 0.0,
#     "l2_reg": 0.0,
#     "learning_rate": 0.001,
#     "epochs": 1,
#     "batch_size": 64,
#     "validation_split": 0.2,
#     "prediction_threshold": 0.5,
# }

# hyper_parameters = {
#     "window_size": 75,
#     "lstm_units": [64, 32, 16],
#     "depth": 6,
#     "nb_filters": 32,
#     "dropout_dense": [0.5, 0.4],
#     "dropout_lstm": 0.4,
#     "l2_reg": 0.01,
#     "learning_rate": 0.0005,
#     "epochs": 255,
#     "batch_size": 64,
#     "validation_split": 0.2,
#     "prediction_threshold": 0.5,
# }


config = {
    "settings": settings,
    "general_parameters": general_parameters,
    "hyper_parameters": hyper_parameters
}