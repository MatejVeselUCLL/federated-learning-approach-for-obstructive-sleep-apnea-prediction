factor = 1
import git

repo = git.Repo('', search_parent_directories=True)


settings = {
    "config_name": "config_strict_good",
    "experiment_name": "Hyperparameter Tuning Omop 1s",
    "study_name": f"Objective: Val F1 Score (Variance to Rok: {factor}) Brez C10 (1) in C28 (20)",
    "output_path": f"{repo.working_tree_dir}/output", # Relative to tuning.py.
    "input_path": f"{repo.working_tree_dir}/input", # Relative to tuning.py.
    "dataset_filename": "ml_train_dataset_from_omop_1s_minimal.csv",
    "log_file": "log.txt", # Do not change this.
    "objective_metric": "true_val_f1_score", # For possible options, see final_metrics variable in train_model.py.
    "objective_direction": "maximize", # Either 'maximize' or 'minimize'
    "epochs": 10,
    "trials": 1
}

general_parameters = {
    "FEATURES": [ "value_as_number_spo2", "value_as_number_hr" ],
    "TARGET": "value_as_number_apnea",
    "ID_COLUMN": "person_id",
    "TEST_PERSON_IDS": [],
    "WINDOW": 75, # Window size: 5 minutes (4s x 75 = 300s = 5 min)
    "STEP": 15 # Step size: size of a step to take on moving Window
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
    "window_size": 75,
    "lstm_units": [64, 32, 16],
    "depth": 6,
    "nb_filters": 32,
    "dropout_dense": [0.5, 0.4],
    "dropout_lstm": 0.4,
    "l2_reg": 0.01,
    "learning_rate": 0.0005,
    "epochs": 1,
    "batch_size": 16,
    "validation_split": 0.2,
    "prediction_threshold": 0.5,
}

# hyper_parameters = {
#     "window_size": [make_positive(75 - v1, 1), make_positive(75 + v1, 1)],
#     "lstm_units": [
#         [make_positive(64-v1, 1), make_positive(64+v1, 1)], # Lower
#         [make_positive(32-v1, 1), make_positive(32+v1, 1)],
#         [make_positive(16-v1, 1), make_positive(16+v1, 1)], # Higher
#     ],
#     "depth": [make_positive(6-v1, 1), make_positive(6+v1, 1)],
#     "nb_filters": [make_positive(32-v1, 1), make_positive(32+v1, 1)], # Higher
#     "dropout_dense": [
#         [make_positive(0.5-v2, 0.1), make_positive(0.5+v2, 0.1)],
#         [make_positive(0.4-v2, 0.1), make_positive(0.4+v2, 0.1)],
#     ],
#     "dropout_lstm": [make_positive(0.4-v2, 0.1), make_positive(0.4+v2, 0.1)],
#     "l2_reg": [make_positive(0.01-v3, 0.001), make_positive(0.01+v3, 0.001)], # It was log!
#     "learning_rate": [make_positive(0.0005-v4, 0.00001), make_positive(0.0005+v4, 0.00001)], # It was log!
#     "epochs": settings["epochs"],
#     "batch_size": [make_positive(128-v1, 1), make_positive(128+v1, 1)], # Lower
#     "prediction_threshold": [make_positive(0.5-v2, 0.1), make_positive(0.5+v2, 0.1)],
#     "validation_split": [make_positive(0.2-v2, 0.1), make_positive(0.2+v2, 0.1)] # Lower
# }

config = {
    "settings": settings,
    "general_parameters": general_parameters,
    "hyper_parameters": hyper_parameters
}