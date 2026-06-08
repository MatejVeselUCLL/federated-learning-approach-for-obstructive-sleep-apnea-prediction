In a federated setting, the test consisted of 3 hospitals (cients), 5 rounds and 15 epochs. In a centralized setting, it consisted of 15*5*3 (255) epochs.

Results show no difference in final model performance, as can be seen in the table below.

|                     | centralized | federated | difference |
|---------------------|-------------|-----------|------------|
| train_accuracy      | 0.80        | 0.79      | 0.01       |
| train_f1_1          | 0.51        | 0.52      | 0.01       |
| train_f1_0          | 0.80        | 0.77      | 0.02       |
| train_precision_1   | 0.45        | 0.43      | 0.02       |
| train_precision_0   | 0.85        | 0.86      | 0.01       |
| train_recall_1      | 0.59        | 0.65      | 0.01       |
| train_recall_0      | 0.76        | 0.71      | 0.05       |
| test_accuracy       | /           | 0.83      | /          |
| test_precision      | /           | 0.00      | /          |
| test_recall         | /           | 0.00      | /          |    
| **avg_difference**  |             |           | 0.018      |

/ ... no equivalent metric