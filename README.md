## Project Overview

The **Precepto Hub Hand Signs Recognition** project uses machine learning to recognize hand signs. We compare **three models** for this task:

- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**

## How to Use

### 1. Training the Models

Run the corresponding scripts to train the models:

- **Decision Tree**: `train_model_DT.py`
- **Random Forest**: `train_model_RF.py`
- **SVM**: `train_model_SVM.py`

These scripts will:
- Load the dataset
- Train the model
- Output the accuracy
- Save the trained model

### 2. Model Accuracy Comparison

| **Model**                  | **Accuracy**  |
|----------------------------|---------------|
| Decision Tree              | **95.91%**       |
| Random Forest              | **99.64%**       |
| Support Vector Machine (SVM)| **92.62%**       |

### Conclusion

Based on the **accuracy results**, the **Random Forest** model provides the best performance.
