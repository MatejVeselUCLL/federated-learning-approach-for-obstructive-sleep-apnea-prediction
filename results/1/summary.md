In a federated setting, the test consisted of 3 hospitals (cients), 5 rounds and 15 epochs. In a centralized setting, it consisted of 15*5*3 (255) epochs.

Results show no difference in final model performance, as can be seen in the table below.

|                | centralized | federated |
|----------------|-------------|-----------|
| train_accuracy | 0.79        | 0.80      |
| test_accuracy  | 0.83        | 0.83      |
