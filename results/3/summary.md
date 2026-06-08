In a federated setting, the test consisted of 3 hospitals (cients), 5 rounds and 15 epochs. In a centralized setting, it consisted of 15*5*3 (255) epochs.

Results show no difference in final model performance, as can be seen in the table below.

|                           | centralized | federated | difference |
|---------------------------|-------------|-----------|------------|
| train_accuracy            | 0.80        | 0.80      | 0.00       |
| train_f1_1                | 0.49        | 0.52      | 0.03       |
| train_f1_0                | 0.80        | 0.77      | 0.03       |
| train_precision_1         | 0.44        | 0.42      | 0.02       |
| train_precision_0         | 0.84        | 0.87      | 0.03       |
| train_recall_1            | 0.57        | 0.68      | 0.11       |
| train_recall_0            | 0.76        | 0.69      | 0.07       |
| test_accuracy             | 0.79        | 0.83      | 0.04       |
| test_precision            | 0.39        | 0.00      | 0.39       |
| test_recall               | 0.41        | 0.00      | 0.41       |    
| **avg_train_difference**  |             |           | 0.041      |
| **avg_test_difference**   |             |           | 0.280      |
| **avg_difference**        |             |           | 0.113      |

/ ... no equivalent metric