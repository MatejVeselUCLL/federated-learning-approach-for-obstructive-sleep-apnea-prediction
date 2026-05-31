from pprint import pprint

import numpy as np
import pandas as pd
from opentelemetry.semconv.attributes.telemetry_attributes import \
    TelemetrySdkLanguageValues
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv1D, MaxPooling1D, \
    Dropout, LSTM, Dense

from src.model.config.config_rok_factor_1_brez_C_v4_0 import config

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict

from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def get_test_person_ids(DATASET_FILENAME):
    settings = config["settings"]
    general_parameters = config["general_parameters"]
    TEST_PERSON_IDS_h1 = general_parameters["TEST_PERSON_IDS_h1"]
    TEST_PERSON_IDS_h2 = general_parameters["TEST_PERSON_IDS_h2"]
    TEST_PERSON_IDS_h3 = general_parameters["TEST_PERSON_IDS_h3"]

    print("DATASET_FILENAME", DATASET_FILENAME, DATASET_FILENAME.startswith("h1"))
    if DATASET_FILENAME.startswith("h1"):
        return TEST_PERSON_IDS_h1
    elif DATASET_FILENAME.startswith("h2"):
        return TEST_PERSON_IDS_h2
    elif DATASET_FILENAME.startswith("h3"):
        return TEST_PERSON_IDS_h3
    else:
        return general_parameters["TEST_PERSON_IDS"]

# Load dataset
def load_dataset(INPUT_PATH: str, DATASET_FILENAME: str, TEST_PERSON_IDS: str):
    # df = pd.read_csv(f'{DATASET_FILENAME}')

    df = pd.read_csv(f'{INPUT_PATH}/{DATASET_FILENAME}')
    # df = pd.read_csv(f'tfexample/input/ml_train_dataset_from_omop_1s_minimal.csv')

    # print("Dataset loaded successfully")
    # print(f"Shape: {df.shape}")
    # print(f"\nColumns: {df.columns.tolist()}")
    # print(f"\nFirst few rows:")
    df.head()
    print("DFINFO")
    df.info()
    len(df)

    # In[ ]:

    # In[6]:

    # Dataset statistics
    # print("Dataset Statistics:")
    # print(df.describe())
    # print(f"\nMissing values:\n{df.isnull().sum()}")
    # # print(f"\nUnique persons: {df['person_id'].nunique()}")

    # ### Test/Train split dataset

    # In[7]:

    TEST_PERSON_IDS = get_test_person_ids(DATASET_FILENAME)
    print("TEST PERSON IDS", TEST_PERSON_IDS)

    # subjects you want in first dataset
    subject_id = TEST_PERSON_IDS
    df_test = df[df["person_id"].isin(subject_id)].copy()
    df_train = df[~df["person_id"].isin(subject_id)].copy()

    return df, df_train, df_test

def inceptiontime_temporal(input_tensor, depth=6, nb_filters=32):
        """
        InceptionTime-inspired block for temporal feature extraction.
        Uses multiple kernel sizes to capture different temporal patterns.

        Args:
            input_tensor: Input tensor
            depth: Depth parameter (not used in this version)
            nb_filters: Number of filters for each convolution

        Returns:
            Merged tensor from all parallel paths
        """
        # Parallel convolutions with different kernel sizes
        # print("JJJJJJJ")
        # print(nb_filters)
        # print(type(nb_filters))
        conv1 = Conv1D(nb_filters, kernel_size=1, padding='same', activation='relu')(input_tensor)
        conv3 = Conv1D(nb_filters, kernel_size=3, padding='same', activation='relu')(input_tensor)
        conv5 = Conv1D(nb_filters, kernel_size=5, padding='same', activation='relu')(input_tensor)

        # Maxpooling path
        maxpool = MaxPooling1D(pool_size=3, strides=1, padding='same')(input_tensor)
        convpool = Conv1D(nb_filters, kernel_size=1, padding='same', activation='relu')(maxpool)

        # Concatenate all paths
        merged = Concatenate()([conv1, conv3, conv5, convpool])

        return merged


def build_inception_lstm_dual_branch(feature1_len, feature2_len, lstm_units=(64, 32, 16),
                                     depth=6, nb_filters=32, dropout_dense=(0.5, 0.4),
                                     dropout_lstm=0.4, l2_reg=0.01):
    """
    Build dual-branch Inception-LSTM model for apnea detection.

    Args:
        feature1_len: Length of RR interval sequences
        feature2_len: Length of SpO2 sequences
        lstm_units: Tuple of LSTM layer units
        depth: Depth of InceptionTime block
        nb_filters: Number of filters in convolutional layers
        dropout_dense: Dropout rates for dense layers
        dropout_lstm: Dropout rate for LSTM layers
        l2_reg: L2 regularization factor

    Returns:
        Compiled Keras model
    """
    # RR branch
    feature1_input = Input(shape=(feature1_len, 1), name='feature1_input')
    feature1_x = inceptiontime_temporal(feature1_input, depth=depth, nb_filters=nb_filters)
    feature1_x = LSTM(lstm_units[0], return_sequences=True,
                kernel_regularizer=l2(l2_reg),
                recurrent_regularizer=l2(l2_reg))(feature1_x)
    feature1_x = Dropout(dropout_lstm)(feature1_x)
    feature1_x = LSTM(lstm_units[1], return_sequences=True,
                kernel_regularizer=l2(l2_reg),
                recurrent_regularizer=l2(l2_reg))(feature1_x)
    feature1_x = Dropout(dropout_lstm)(feature1_x)
    feature1_x = LSTM(lstm_units[2],
                kernel_regularizer=l2(l2_reg),
                recurrent_regularizer=l2(l2_reg))(feature1_x)
    feature1_x = Dropout(dropout_lstm)(feature1_x)

    # SpO2 branch
    feature2_input = Input(shape=(feature2_len, 1), name='feature2_input')
    feature2_x = inceptiontime_temporal(feature2_input, depth=depth, nb_filters=nb_filters)
    feature2_x = LSTM(lstm_units[0], return_sequences=True,
                  kernel_regularizer=l2(l2_reg),
                  recurrent_regularizer=l2(l2_reg))(feature2_x)
    feature2_x = Dropout(dropout_lstm)(feature2_x)
    feature2_x = LSTM(lstm_units[1], return_sequences=True,
                  kernel_regularizer=l2(l2_reg),
                  recurrent_regularizer=l2(l2_reg))(feature2_x)
    feature2_x = Dropout(dropout_lstm)(feature2_x)
    feature2_x = LSTM(lstm_units[2],
                  kernel_regularizer=l2(l2_reg),
                  recurrent_regularizer=l2(l2_reg))(feature2_x)
    feature2_x = Dropout(dropout_lstm)(feature2_x)

    # Late fusion
    combined = Concatenate()([feature1_x, feature2_x])

    # Dense head with L2 regularization
    z = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(combined)
    z = Dropout(dropout_dense[0])(z)
    z = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))(z)
    z = Dropout(dropout_dense[1])(z)
    z = Dense(16, activation='relu', kernel_regularizer=l2(l2_reg))(z)
    output = Dense(1, activation='sigmoid', name='apnea_output')(z)

    model = Model(inputs=[feature1_input, feature2_input], outputs=output)

    return model

def prepare_dual_branch_train_data(df, window_size, step, features, target, id_column):
    """
    Prepare separate RR, SpO2 and HR sequences for multi-branch model using sliding window with step size.

    Args:
        df: DataFrame with physiological signals
        window_size: Number of timesteps per sequence
        step: Step size (stride) between consecutive windows
        features: List of feature column names [rr, spo2, hr]
        target: Target column name
        id_column: Column identifying each subject/person

    Returns:
        X_hr: Scaled HR sequences (samples, window_size, 1)
        X_spo2: Scaled SpO2 sequences (samples, window_size, 1)
        y: Target labels
        scaler_rr: Fitted StandardScaler for RR
        scaler_spo2: Fitted StandardScaler for SpO2
        scaler_hr: Fitted StandardScaler for HR
    """

    X_feature1 = []
    X_feature2 = []
    y = []

    # Extract sequences per person
    for pid, g in df.groupby(id_column):
        g = g.reset_index(drop=True)

        feature1_values = g[features[0]].values
        feature2_values = g[features[1]].values

        labels = g[target].values
        # print("LLABELS")
        # pprint(labels)

        # print("GGG")
        # pprint(g)
        # print("lenGGG", len(g), window_size)


        # print("THECOND", len(g) - window_size)
        # print("THECOND2", step)

        # Sliding window with step
        for i in range(0, len(g) - window_size, step):
            X_feature1.append(feature1_values[i:i + window_size])
            X_feature2.append(feature2_values[i:i + window_size])
            # print("HERRE")
            y.append(labels[i + window_size])

    # Convert to numpy arrays
    X_feature1 = np.array(X_feature1).reshape(-1, window_size, 1)
    X_feature2 = np.array(X_feature2).reshape(-1, window_size, 1)
    y = np.array(y)
    # print("YYY")
    # pprint(y)

    # print(f"X_feature1 shape: {X_feature1.shape}")
    # print(f"X_spo2 shape: {X_feature2.shape}")
    # print(f"y shape: {y.shape}")

    # Normalize features separately
    scaler_feature1 = StandardScaler()
    scaler_feature2 = StandardScaler()

    # TODO Matej
    # X_feature1_scaled = scaler_feature1.fit_transform(X_feature1.reshape(-1, 1))
    # X_feature2_scaled = scaler_feature2.fit_transform(X_feature2.reshape(-1, 1))
    #
    # X_feature1 = X_feature1_scaled.reshape(-1, window_size, 1)
    # X_feature2 = X_feature2_scaled.reshape(-1, window_size, 1)

    return X_feature1, X_feature2, y, scaler_feature1, scaler_feature2


def evaluate_model(msg: Message, context: Context):
    settings = config["settings"]
    general_parameters = config["general_parameters"]
    params = config["hyper_parameters"]

    FEATURES = general_parameters["FEATURES"]
    TARGET = general_parameters["TARGET"]
    ID_COLUMN = general_parameters["ID_COLUMN"]
    WINDOW = general_parameters["WINDOW"]
    STEP = general_parameters["STEP"]
    TEST_PERSON_IDS = get_test_person_ids(context.node_config["dataset-filename"])

    INPUT_PATH = settings["input_path"]
    DATASET_FILENAME = context.node_config["dataset-filename"]
    OUTPUT_PATH = settings["output_path"]

    # Load the data
    # _, _, x_test, y_test = load_data(partition_id, num_partitions)

    print("Evaluating on", TEST_PERSON_IDS)
    df, df_train, df_test = load_dataset(INPUT_PATH, DATASET_FILENAME, TEST_PERSON_IDS)

    # Prepare dual-branch input
    X_spo2_test, X_hr_test, y_test, scaler_spo2_test, scaler_hr_test = prepare_dual_branch_train_data(
        df=df_test,
        window_size=WINDOW,
        step=STEP,
        features=FEATURES,
        target=TARGET,
        id_column=ID_COLUMN
    )


    # Load the model
    # Build model
    model = build_inception_lstm_dual_branch(
        feature1_len=WINDOW,
        feature2_len=WINDOW,
        lstm_units=tuple(params["lstm_units"]),
        depth=params["depth"],
        nb_filters=params["nb_filters"],
        dropout_dense=tuple(params["dropout_dense"]),
        dropout_lstm=params["dropout_lstm"],
        l2_reg=params["l2_reg"]
    )

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=params["learning_rate"]),
        loss='mean_squared_error',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
    )

    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())


    # Evaluate the model
    eval_loss, eval_acc, eval_auc, eval_recall, eval_precision = model.evaluate([X_hr_test, X_spo2_test], y_test, verbose=0)

    # Pack and send the model weights and metrics as a message
    metrics = {
        "eval_acc": eval_acc,
        "eval_auc": eval_auc,
        "eval_recall": eval_recall,
        "eval_precision": eval_precision,
        "eval_loss": eval_loss,
        "num-examples": len(X_hr_test),
    }
    return metrics

def train_model(dataset_filename=""):

    settings = config["settings"]
    general_parameters = config["general_parameters"]
    params = config["hyper_parameters"]

    FEATURES = general_parameters["FEATURES"]
    TARGET = general_parameters["TARGET"]
    ID_COLUMN = general_parameters["ID_COLUMN"]
    WINDOW = general_parameters["WINDOW"]
    STEP = general_parameters["STEP"]

    INPUT_PATH = settings["input_path"]
    DATASET_FILENAME = settings["dataset_filename"]
    OUTPUT_PATH = settings["output_path"]

    # Take the dataset from the client context
    print("DDD", dataset_filename)
    DATASET_FILENAME = dataset_filename if dataset_filename != "" else DATASET_FILENAME

    TEST_PERSON_IDS = get_test_person_ids(DATASET_FILENAME)

    #!/usr/bin/env python
    # coding: utf-8

    # # Apnea Detection - Dual-Branch Inception-LSTM Model
    #
    # This notebook trains a dual-branch deep learning model for sleep apnea detection using RR interval and SpO2 signals.

    # ## 1. Import Libraries

    # In[1]:


    import pandas as pd
    from sklearn.utils import class_weight
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf

    import os
    import json
    import pickle

    # print("✅ Libraries imported successfully")
    # print(f"TensorFlow version: {tf.__version__}")


    # In[2]:


    # print("TF version:", tf.__version__)
    # print("Built with CUDA:", tf.test.is_built_with_cuda())
    # print("GPUs:", tf.config.list_physical_devices("GPU"))


    # In[3]:

    #
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #         # print("✅ GPU Memory Growth Enabled")
    #     except RuntimeError as e:
    #         print(e)


    # In[4]:

    # 1. Identify Physical Devices
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                # tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=512)])
            print(f"✅ Found {len(gpus)} GPU(s). Memory Growth Enabled.")
        except RuntimeError as e:
            print(e)
    else:
        print("ℹ️ No GPU found. Running on CPU.")

    # 2. Verify which device is active for computations
    # print("--- Device Status ---")
    logical_gpus = tf.config.list_logical_devices('GPU')
    logical_cpus = tf.config.list_logical_devices('CPU')

    # print(f"Available Logical CPUs: {len(logical_cpus)}")
    # print(f"Available Logical GPUs: {len(logical_gpus)}")

    # 3. Final confirmation test
    device_name = tf.test.gpu_device_name()
    if device_name:
        print(f"🚀 Currently using GPU: {device_name}")
    else:
        print("💻 Currently using CPU")


    # ## 2. Load the dataset
    df, df_train, df_test = load_dataset(INPUT_PATH, DATASET_FILENAME, TEST_PERSON_IDS)


    # ## 3. Model Architecture Functions

    # In[8]:
    ### TODO Matej
    # print("✅ InceptionTime temporal block defined")


    # In[9]:


    # TODO Matej
    # print("✅ Model builder function defined")


    # ## 4. Data Preparation Functions

    # In[10]:


    import numpy as np



    # ## 5. Define Features and Parameters

    # In[11]:


    # Define features and target

    # print(f"Features: {FEATURES}")
    # print(f"Target: {TARGET}")
    # print(f"Window size: {WINDOW} timesteps (5 minutes)")


    # ## 6. Prepare Training Data

    # In[12]:


    # Prepare dual-branch input
    X_spo2, X_hr, y, scaler_spo2, scaler_hr = prepare_dual_branch_train_data(
        df=df_train,
        window_size=WINDOW,
        step=STEP,
        features=FEATURES,
        target=TARGET,
        id_column=ID_COLUMN
    )

    # print(f"\nData preparation complete!")
    # print(f"Total samples: {len(y)}")
    # print(f"Apnea ratio: {np.mean(y):.4f}")
    # print(f"Class distribution: {np.bincount(y.astype(int))}")


    # ## 7. Define Hyperparameters

    # In[13]:


    # Hyperparameters
    # print("Model Hyperparameters:")
    # print("=" * 50)
    for key, value in params.items():
        print(f"  {key}: {value}")
    # print("=" * 50)


    # ## 8. Build and Compile Model

    # In[14]:


    # Build model
    model = build_inception_lstm_dual_branch(
        feature1_len=WINDOW,
        feature2_len=WINDOW,
        lstm_units=tuple(params["lstm_units"]),
        depth=params["depth"],
        nb_filters=params["nb_filters"],
        dropout_dense=tuple(params["dropout_dense"]),
        dropout_lstm=params["dropout_lstm"],
        l2_reg=params["l2_reg"]
    )

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=params["learning_rate"]),
        loss='mean_squared_error',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.Precision(name='precision')]
    )

    # print("✅ Model built and compiled successfully")
    # print("\nModel Summary:")
    # print(model.summary())


    # ## 9. Visualize Model Architecture

    # In[15]:


    # Create input directory if it doesn't exist
    os.makedirs(f'{OUTPUT_PATH}/model_artifacts', exist_ok=True)

    # Save model architecture as JSON and text
    model_json = model.to_json()
    with open(f'{OUTPUT_PATH}/model_artifacts/dual_branch_inception_lstm_architecture.json', 'w', encoding='utf-8') as f:
        f.write(model_json)

    with open(f'{OUTPUT_PATH}/model_artifacts/dual_branch_inception_lstm_summary.txt', 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    configg = model.get_config()
    with open(f'{OUTPUT_PATH}/model_artifacts/dual_branch_inception_lstm_config.json', 'w', encoding='utf-8') as f:
        json.dump(configg, f, indent=2)

    # print("✅ Model architecture saved in multiple formats")


    # In[16]:


    # Model statistics
    # print("\nModel Statistics:")
    # print("=" * 50)
    # print(f"Total parameters: {model.count_params():,}")
    # print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    # print(f"Non-trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]):,}")
    # print(f"Number of layers: {len(model.layers)}")
    # print("=" * 50)


    # ## 10. Compute Class Weights

    # In[17]:


    # print("classweightss")
    # pprint(y)
    # print("ylength")
    # print(len(y))
    # print("ydistinct")
    # print(np.unique(y))
    # print(len(np.unique(y))<2)


    # Compute class weights to handle imbalanced input
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )

    # print("weightss")
    # pprint(weights)
    # print("weightsslength")
    # print(len(weights))
    class_weights = {0: weights[0]}
    # class_weights = {0: weights[0], 1: weights[1]} // TODO Matej

    # print("Class Weights (for imbalanced input):")
    # print(f"  Class 0 (No Apnea): {weights[0]:.4f}")
    # # print(f"  Class 1 (Apnea): {weights[1]:.4f}") // TODO Matej


    # ## 11. Define Callbacks

    # In[18]:


    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        min_delta=0.001,
        verbose=1
    )

    # print("✅ Callbacks configured")


    # ## 12. Train Model

    # In[19]:


    # Train model with TWO inputs [X_rr, X_spo2]
    # print("Starting model training...")
    # print("=" * 50)

    # print("verify shapes", len([X_hr, X_spo2]), len(y))
    # print("verify shapess", type([X_hr, X_spo2]), type(y))


    history = model.fit(
        [X_hr, X_spo2], y,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_split=params["validation_split"],
        shuffle=True,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # print("\n" + "=" * 50)
    # print("✅ Training complete!")
    # print("=" * 50)


    # ## 13. Plot Training History

    # In[20]:


    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', marker='o')
    axes[1].plot(history.history['val_loss'], label='Val Loss', marker='s')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC
    axes[2].plot(history.history['auc'], label='Train AUC', marker='o')
    axes[2].plot(history.history['val_auc'], label='Val AUC', marker='s')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('AUC')
    axes[2].set_title('Model AUC')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PATH}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()

    # print("✅ Training curves saved to '", f"{OUTPUT_PATH}/training_curves.png'")


    # ## 14. Make Predictions

    # In[21]:


    # Make predictions with TWO inputs
    # print("Making predictions...")
    print("Predicting on", TEST_PERSON_IDS)
    y_pred_prob = model.predict([X_hr, X_spo2])
    y_pred = (y_pred_prob > params["prediction_threshold"]).astype(int).flatten()

    # print(f"Predictions shape: {y_pred_prob.shape}")
    # print(f"Sample predictions (first 10): {y_pred[:10]}")
    # print(f"Prediction distribution: {np.bincount(y_pred)}")


    # ## 15. Confusion Matrix

    # In[22]:


    # Generate confusion matrix
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Apnea', 'Apnea'])
    # print("cmmm")
    # pprint(cm)
    try:
        tn, fp, fn, tp = cm.ravel()
    except: # TODO Matej
        tn = 0
        fp = 0
        fn = 0
        tp = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix - Apnea Detection", fontsize=14, fontweight='bold')
    plt.savefig(f'{OUTPUT_PATH}/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    # print("\nConfusion Matrix:")
    # print(cm)
    # print("\n✅ Confusion matrix saved to '", f"{OUTPUT_PATH}/confusion_matrix.png'")


    # ## 16. Classification Report

    # In[23]:


    # Generate classification report
    report = classification_report(y, y_pred, target_names=['No Apnea', 'Apnea'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # print("\nClassification Report:")
    # print("=" * 70)
    # print(report_df)
    # print("=" * 70)

    # Save report
    report_df.to_csv(f'{OUTPUT_PATH}/classification_report.csv')
    # print("\n✅ Classification report saved to '", f"{OUTPUT_PATH}/classification_report.csv'")


    # ## 17. Final Metrics Summary

    # In[24]:


    # Compile final metrics
    final_metrics = {
        "final_train_loss": history.history['loss'][-1],
        "final_val_loss": history.history['val_loss'][-1],
        "final_train_accuracy": history.history['accuracy'][-1],
        "final_val_accuracy": history.history['val_accuracy'][-1],
        "final_train_auc": history.history['auc'][-1],
        "final_val_auc": history.history['val_auc'][-1],
        "total_epochs_trained": len(history.history['loss']),
        "precision_no_apnea": report['No Apnea']['precision'],
        "recall_no_apnea": report['No Apnea']['recall'],
        "f1_score_no_apnea": report['No Apnea']['f1-score'],
        "precision_apnea": report['Apnea']['precision'],
        "recall_apnea": report['Apnea']['recall'],
        "f1_score_apnea": report['Apnea']['f1-score'],
        "num-examples": len(df_train),
        # "true_negative": tn,
        # "false_positive": fp,
        # "false_negative": fn,
        # "true_positive": tp
    }

    metrics_df = pd.DataFrame(final_metrics.items(), columns=['Metric', 'Value'])

    # print("\n" + "=" * 70)
    # print("FINAL MODEL PERFORMANCE SUMMARY")
    # print("=" * 70)
    # print(metrics_df.to_string(index=False))
    # print("=" * 70)

    # Save metrics
    metrics_df.to_csv(f'{OUTPUT_PATH}/final_metrics.csv', index=False)
    # print("\n✅ Final metrics saved to '", f"{OUTPUT_PATH}/final_metrics.csv'")


    # ## 18. Save Model and Artifacts

    # In[25]:


    # Save Keras model
    model_path = f'{OUTPUT_PATH}/model_weights/apnea_inception_dual_branch_model.h5'
    model.save(model_path)
    # print(f"✅ Model saved to '{model_path}'")

    # Save scalers
    with open(f'{OUTPUT_PATH}/model_weights/scaler_hr.pkl', 'wb') as f:
        pickle.dump(scaler_hr, f)

    with open(f'{OUTPUT_PATH}/model_weights/scaler_spo2.pkl', 'wb') as f:
        pickle.dump(scaler_spo2, f)

    # print("✅ Scalers saved to '", f"{OUTPUT_PATH}/model_weights/'")

    # Save model parameters
    os.makedirs(f'{OUTPUT_PATH}/model_metdata', exist_ok=True)
    with open(f'{OUTPUT_PATH}/model_metdata/model_params.json', 'w') as f:
        json.dump(params, f, indent=2)

    # print("✅ Model parameters saved to 'model_metdata/model_params.json'")

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'{OUTPUT_PATH}/model_metdata/training_history.csv', index=False)
    # print("✅ Training history saved to '", f"{OUTPUT_PATH}/model_metdata/training_history.csv'")


    # ## 19. Summary

    # In[26]:


    # print("\n" + "=" * 70)
    # print("🎉 TRAINING PIPELINE COMPLETE 🎉")
    # print("=" * 70)
    # print("\nSaved Artifacts:")
    # print("  📊 Model: ", f"{OUTPUT_PATH}/model_weights/apnea_inception_dual_branch_model.h5")
    # print("  📊 Scalers: ", f"{OUTPUT_PATH}/model_weights/scaler_rr.pkl, {OUTPUT_PATH}/scaler_spo2.pkl")
    # print("  📊 Parameters: ", f"{OUTPUT_PATH}/model_params.json")
    # print("  📊 Training History: ", f"{OUTPUT_PATH}/training_history.csv")
    # print("  📊 Final Metrics: ", f"{OUTPUT_PATH}/final_metrics.csv")
    # print("  📊 Classification Report: ", f"{OUTPUT_PATH}/classification_report.csv")
    # print("  📊 Training Curves: ", f"{OUTPUT_PATH}/training_curves.png")
    # print("  📊 Confusion Matrix: ", f"{OUTPUT_PATH}/confusion_matrix.png")
    # print("  📊 Model Architecture: ", f"{OUTPUT_PATH}/model_artifacts/")
    # print("\n" + "=" * 70)


    # ## Make predictions on Test dataset

    # In[29]:


    import numpy as np
    from joblib import load
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import StandardScaler

    # Load the trained model
    model = load_model(f'{OUTPUT_PATH}/model_weights/apnea_inception_dual_branch_model.h5')
    # Load new input for prediction
    # df = pd.read_csv('ml_train_dataset_from_omop.csv')


    # def prepare_dual_branch_prediction_data(
    #     df,
    #     features,
    #     id_column,
    #     scaler_feature1,
    #     scaler_feature2,
    #     window_size,
    #     step=None,
    #     last_window_only=False
    # ):
    #     """
    #     Prepare RR and SpO2 sequences for prediction using already fitted scalers.

    #     Args:
    #         df: DataFrame with physiological signals
    #         window_size: Number of timesteps per sequence
    #         features: List of feature column names [feature1, feature2]
    #         id_column: Column identifying each subject/person
    #         scaler_feature1: Pre-fitted scaler for feature1 (from training)
    #         scaler_feature2: Pre-fitted scaler for feature2 (from training)
    #         step: Step size between windows (used only if last_window_only=False)
    #         last_window_only: If True, return only the most recent window per subject

    #     Returns:
    #         X_feature1: Scaled sequences (samples, window_size, 1)
    #         X_feature2: Scaled sequences (samples, window_size, 1)
    #         ids: List of subject IDs aligned with samples
    #     """
    #     """
    #     Prepare feature1 and feature2 sequences for prediction using already fitted scalers.

    #     Args:
    #         df: DataFrame with physiological signals
    #         features: List of feature column names [feature1, feature2]
    #         id_column: Column identifying each subject/person
    #         scaler_feature1: Pre-fitted scaler for feature1 (from training)
    #         scaler_feature2: Pre-fitted scaler for feature2 (from training)
    #         window_size: Number of timesteps per sequence
    #         step: Step size between windows (used only if last_window_only=False)
    #         last_window_only: If True, return only the most recent window per subject
    #         debug: If True, # prints number of windows per subject

    #     Returns:
    #         X_feature1: Scaled sequences (samples, window_size, 1)
    #         X_feature2: Scaled sequences (samples, window_size, 1)
    #         ids: List of subject IDs aligned with samples
    #     """

    #     X_feature1 = []
    #     X_feature2 = []
    #     ids = []

    #     for pid, g in df.groupby(id_column):
    #         g = g.reset_index(drop=True)

    #         f1 = g[features[0]].values
    #         f2 = g[features[1]].values

    #         if len(g) < window_size:
    #             continue  # skip too-short sequences

    #         # --- Only last window (common for real-time prediction) ---
    #         if last_window_only:
    #             X_feature1.append(f1[-window_size:])
    #             X_feature2.append(f2[-window_size:])
    #             ids.append(pid)

    #         # --- All sliding windows ---
    #         else:
    #             if step is None:
    #                 step = 1

    #             n_windows = (len(g) - window_size) // step + 1

    #             for i in range(0, len(g) - window_size + 1, step):
    #                 X_feature1.append(f1[i:i + window_size])
    #                 X_feature2.append(f2[i:i + window_size])
    #                 ids.append(pid)

    #     if len(X_feature1) == 0:
    #         raise ValueError("No valid windows were created. Check window_size and input length.")

    #     # Convert to numpy
    #     X_feature1 = np.array(X_feature1).reshape(-1, window_size, 1)
    #     X_feature2 = np.array(X_feature2).reshape(-1, window_size, 1)

    #     # Apply EXISTING scalers (DO NOT FIT)
    #     X_feature1_scaled = scaler_feature1.transform(X_feature1.reshape(-1, 1))
    #     X_feature2_scaled = scaler_feature2.transform(X_feature2.reshape(-1, 1))

    #     X_feature1 = X_feature1_scaled.reshape(-1, window_size, 1)
    #     X_feature2 = X_feature2_scaled.reshape(-1, window_size, 1)

    #     # print(f"Prediction X_feature1 shape: {X_feature1.shape}")
    #     # print(f"Prediction X_feature2 shape: {X_feature2.shape}")

    #     return X_feature1, X_feature2, ids

    def prepare_dual_branch_prediction_data(df, window_size, step, features, id_column, scaler_feature1, scaler_feature2):
        """
        Prepare sequences for inference using PRE-FITTED scalers.

        Args:
            df: DataFrame with physiological signals
            window_size: Number of timesteps per sequence
            step: Step size (stride) between windows
            features: List of feature column names [feat1, feat2]
            id_column: Column identifying each subject/person
            scaler_feature1: The fitted StandardScaler from training for feature 1
            scaler_feature2: The fitted StandardScaler from training for feature 2

        Returns:
            X_feature1: Scaled sequences (samples, window_size, 1)
            X_feature2: Scaled sequences (samples, window_size, 1)
        """

        X_f1 = []
        X_f2 = []

        # Extract sequences per person
        for pid, g in df.groupby(id_column):
            g = g.reset_index(drop=True)

            f1_values = g[features[0]].values
            f2_values = g[features[1]].values

            # print("f1_values")
            # pprint(f1_values)

            # Sliding window with step
            for start_pos in range(0, len(g) - window_size, step):
                end_pos = start_pos + window_size
                X_f1.append(f1_values[start_pos:end_pos])
                X_f2.append(f2_values[start_pos:end_pos])

        # Convert to numpy arrays
        X_f1 = np.array(X_f1).reshape(-1, window_size, 1)
        X_f2 = np.array(X_f2).reshape(-1, window_size, 1)

        # print("XXXXbefore")
        # pprint(X_f1)

        # Transform using EXISTING scaler
        # print("XXX")
        # pprint(X_f1)
        # pprint(len(X_f1))
        # print("XXX2")
        # pprint(X_f2)
        # pprint(len(X_f2))
        # TODO Matej
        # X_f1_scaled = scaler_feature1.transform(X_f1.reshape(-1, 1))
        X_f1_scaled = X_f1
        # X_f2_scaled = scaler_feature2.transform(X_f2.reshape(-1, 1))
        X_f2_scaled = X_f2

        # Reshape to 2D for scaler, then back to 3D for model
        # TODO Matej
        X_f1 = X_f1_scaled
        # X_f1 = X_f1_scaled.reshape(-1, window_size, 1)
        X_f2 = X_f2_scaled
        # X_f2 = X_f2_scaled.reshape(-1, window_size, 1)

        return X_f1, X_f2


    FEATURES = [
        "value_as_number_hr",
        "value_as_number_spo2"
    ]

    # load scalers

    scaler_hr = load(f'{OUTPUT_PATH}/model_weights/scaler_spo2.pkl')
    scaler_spo2 = load(f'{OUTPUT_PATH}/model_weights/scaler_hr.pkl')
    scalers_lst = [scaler_spo2, scaler_hr]

    # Prepare input
    x_hr, x_spo2 = prepare_dual_branch_prediction_data(df=df_test,
                                        window_size=WINDOW,
                                        step=STEP,
                                        features=FEATURES,
                                        id_column=ID_COLUMN,
                                        scaler_feature1=scaler_hr,
                                        scaler_feature2=scaler_spo2)

    # Make predictions (model trained with [X_hr, X_spo2])
    y_pred_prob: np.ndarray = model.predict([x_hr, x_spo2])
    y_pred: np.ndarray = (y_pred_prob > params["prediction_threshold"]).astype(int)

    # print("Saved predictions and probabilities.")
    # print(f"Predictions shape: {y_pred.shape}")
    # print(f"Probability shape: {y_pred.shape}")
    # print(f"Apnea detected: {np.sum(y_pred)} / {len(y_pred)} samples")


    # In[36]:


    # len(df_test), len(y_pred_prob.flatten()), y_pred.shape, np.sum(y_pred.flatten()[:]==1),y_pred[:10],x_hr.shape, X_hr.shape


    # In[33]:


    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Assuming you already have y_pred_prob, y_pred, and y from previous code

    # 1. Summary Statistics
    # print("=" * 50)
    # print("PREDICTION SUMMARY")
    # print("=" * 50)
    # print(f"Total samples: {len(y_pred)}")
    # print(f"Apnea detected (class 1): {np.sum(y_pred)} ({100 * np.sum(y_pred) / len(y_pred):.2f}%)")
    # print(f"No apnea (class 0): {len(y_pred) - np.sum(y_pred)} ({100 * (len(y_pred) - np.sum(y_pred)) / len(y_pred):.2f}%)")
    # print(f"\nProbability Statistics:")
    # print(f"  Min: {y_pred_prob.min():.4f}")
    # print(f"  Max: {y_pred_prob.max():.4f}")
    # print(f"  Mean: {y_pred_prob.mean():.4f}")
    # print(f"  Std: {y_pred_prob.std():.4f}")

    # 2. Probability Distribution Plot
    try:
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 3, 1)
        plt.hist(y_pred_prob, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
        plt.xlabel('Apnea Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()

        # 3. Predictions Over Time
        plt.subplot(1, 3, 2)
        plt.plot(y_pred_prob[:500], alpha=0.7, linewidth=0.8, label='Probability')
        plt.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
        plt.xlabel('Sample Index')
        plt.ylabel('Apnea Probability')
        plt.title('Predictions Over Time (First 500 samples)')
        plt.legend()

        # 4. Predicted Class Distribution (Pie Chart)
        plt.subplot(1, 3, 3)
        labels = ['No Apnea (0)', 'Apnea (1)']
        sizes = [len(y_pred) - np.sum(y_pred), np.sum(y_pred)]
        colors = ['lightgreen', 'salmon']
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Predicted Class Distribution')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("There was an issue creating plots.", e)


    # In[42]:


    def add_predictions_to_df(df, y_pred_prob, y_pred_class, group_by_col, window_size, step) -> pd.DataFrame:
        """
        Add predictions to the original dataframe based on the sliding window approach.
        """
        # Initialize prediction columns with NaN
        result_df = df.copy()
        result_df['prediction_probability'] = np.nan
        result_df['prediction_class'] = np.nan

        pred_idx = 0
        for pid, g in result_df.groupby(group_by_col):
            g_indices = g.index.tolist()
            n_windows = max(0, (len(g) - window_size) // step + 1)

            for w in range(n_windows):
                start_pos = w * step
                # ensure that it doesn't overflow the indices
                end_pos = min(start_pos + step, len(g_indices))

                # Fill all rows in this step interval with the same prediction
                for pos in range(start_pos, end_pos):
                    row_idx = g_indices[pos]
                    # if y_pred_prob.size == pred_idx:
                    #     # print("\nresult_df info\n",result_df.info)
                    #     # # print("\ny_pred_prob info\n",y_pred_prob.info)
                    #     # # print("\ny_pred_prob describe\n",y_pred_prob.describe)
                    #     # print("\ny_pred_prob head\n",y_pred_prob)
                    #     # print("\ny_pred_prob size\n",y_pred_prob.size)
                    #     # print("\npred_idx info\n",pred_idx)


                    if pred_idx < y_pred_prob.size:
                        result_df.loc[row_idx, 'prediction_probability'] = y_pred_prob[pred_idx, 0]
                        result_df.loc[row_idx, 'prediction_class'] = y_pred_class[pred_idx, 0]

                pred_idx += 1

        return result_df.dropna()


    # In[44]:


    result_df = add_predictions_to_df(df=df_test,
                          y_pred_prob=y_pred_prob,
                          y_pred_class=y_pred,
                          group_by_col=ID_COLUMN,
                          window_size=WINDOW,
                          step=STEP)


    # In[45]:


    from sklearn.metrics import confusion_matrix

    # CONFUSION MATRIX FUNC

    import numpy as np
    import matplotlib.pyplot as plt

    def make_confusion_matrix(cf,
                              group_names=None,
                              categories='auto',
                              count=True,
                              percent=True,
                              cbar=True,
                              xyticks=True,
                              xyplotlabels=True,
                              sum_stats=True,
                              figsize=None,
                              cmap='Blues',
                              title=None,
                              save_path='./confusion-matrix.png'):

        blanks = ['' for _ in range(cf.size)]

        # Labels inside squares
        if group_names and len(group_names) == cf.size:
            group_labels = [f"{value}\n" for value in group_names]
        else:
            group_labels = blanks

        if count:
            group_counts = [f"{value:0.0f}\n" for value in cf.flatten()]
        else:
            group_counts = blanks

        if percent:
            group_percentages = [f"{value:.2%}" for value in cf.flatten()/np.sum(cf)]
        else:
            group_percentages = blanks

        box_labels = [
            f"{v1}{v2}{v3}".strip()
            for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
        ]
        box_labels = np.asarray(box_labels).reshape(cf.shape)

        # Summary stats
        if sum_stats:
            accuracy = np.trace(cf) / float(np.sum(cf))
            if cf.shape == (2, 2):
                precision = cf[1, 1] / sum(cf[:, 1]) if sum(cf[:, 1]) > 0 else 0
                recall = cf[1, 1] / sum(cf[1, :]) if sum(cf[1, :]) > 0 else 0
                f1_score = (2 * precision * recall / (precision + recall)
                            if (precision + recall) > 0 else 0)

                final_metrics["true_val_accuracy"] = accuracy
                final_metrics["true_val_precision"] = precision
                final_metrics["true_val_recall"] = recall
                final_metrics["true_val_f1_score"] = f1_score

                stats_text = (
                    f"\n\nAccuracy={accuracy:0.3f}"
                    f"\nPrecision={precision:0.3f}"
                    f"\nRecall={recall:0.3f}"
                    f"\nF1 Score={f1_score:0.3f}"
                )
            else:
                stats_text = f"\n\nAccuracy={accuracy:0.3f}"
        else:
            stats_text = ""

        if figsize is None:
            figsize = plt.rcParams.get('figure.figsize')

        plt.figure(figsize=figsize)

        # Heatmap via matplotlib
        im = plt.imshow(cf, interpolation='nearest', cmap=cmap)

        if cbar:
            plt.colorbar(im)

        # Ticks
        if xyticks:
            if categories == 'auto':
                categories = np.arange(cf.shape[0])
            plt.xticks(np.arange(len(categories)), categories)
            plt.yticks(np.arange(len(categories)), categories)
        else:
            plt.xticks([])
            plt.yticks([])

        # Labels inside cells
        thresh = cf.max() / 2.0
        for i in range(cf.shape[0]):
            for j in range(cf.shape[1]):
                plt.text(j, i, box_labels[i, j],
                         ha="center", va="center",
                         color="white" if cf[i, j] > thresh else "black")

        if xyplotlabels:
            plt.ylabel('True label')
            plt.xlabel('Predicted label' + stats_text)
        else:
            plt.xlabel(stats_text)

        if title:
            plt.title(title)

        plt.tight_layout()
        plt.savefig(save_path,pad_inches=3, bbox_inches='tight')
        print("Val Confusion matrix saved to :", save_path)


    def df_checkup(df):
        print('---- Is null val: ----')
        # print(list(df.isnull().sum()))
        # print('---- Is nan val: ----')
        # print(list(df.isna().sum()))
        # print('---- Is zero: ----')
        # print(list((df == 0).sum(axis=0)))
        # print('---- Duplicates: ----')
        # print(df.duplicated().sum())
        # print('---- Columns: ----')
        # print(df.columns)
        # print('---- Shape: ----')
        # print(df.shape)
        # print('---- Index: ----')
        # print(df.index)



    # Confusion matrix
    try:
        cf_matrix = confusion_matrix(result_df['value_as_number_apnea'].iloc[:-WINDOW], result_df['prediction_class'].iloc[:-WINDOW])
        tn, fp, fn, tp = cf_matrix.ravel()
        final_metrics["tn"] = tn
        final_metrics["fp"] = fp
        final_metrics["fn"] = fn
        final_metrics["tp"] = tp
        final_metrics["sum_tn_tp_minus_sum_fn_fp"] = (tn+tp)-(fn+fp)

    except Exception as e:
        print("Error when making confusion matrix", e)
    else:
        labels = ['True Neg','False Pos','False Neg','True Pos']
        categories = ['0', '1']
        make_confusion_matrix(cf_matrix, title=settings["study_name"], group_names=labels, categories=categories, cmap='Blues', figsize=(7,5), save_path=f'{OUTPUT_PATH}/val_confusion_matrix.png')


    # convert all values in final_metrics to float.
    for k, v in final_metrics.items():
        final_metrics[k] = float(v)

    print("FINAL-METRICS")
    pprint(final_metrics)

    return final_metrics, model