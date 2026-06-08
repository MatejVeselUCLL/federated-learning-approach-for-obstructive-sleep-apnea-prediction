In a federated setting, the test consisted of 3 hospitals (cients), 5 rounds and 15 epochs. In a centralized setting, it consisted of 15*5*3 (255) epochs.

Results show no difference in final model performance, as can be seen in the table below.

|                           | centralized | federated | difference |
|---------------------------|-------------|-----------|------------|
| train_accuracy            | 0.80        | 0.79      | 0.01       |
| train_f1_1                | 0.52        | 0.51      | 0.01       |
| train_f1_0                | 0.78        | 0.74      | 0.02       |
| train_precision_1         | 0.43        | 0.40      | 0.03       |
| train_precision_0         | 0.86        | 0.87      | 0.01       |
| train_recall_1            | 0.64        | 0.70      | 0.06       |
| train_recall_0            | 0.72        | 0.65      | 0.07       |
| test_accuracy             | 0.76        | 0.83      | 0.07       |
| test_precision            | 0.35        | 0.00      | 0.35       |
| test_recall               | 0.48        | 0.00      | 0.48       |    
| **avg_train_difference**  |             |           | 0.030      |
| **avg_test_difference**   |             |           | 0.300      |
| **avg_difference**        |             |           | 0.111      |

/ ... no equivalent metric