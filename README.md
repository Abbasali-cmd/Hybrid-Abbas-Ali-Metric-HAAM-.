


# Hybrid Abbas Ali Metric (HAAM)

Hybrid Abbas Ali Metric (HAAM) is a custom distance metric designed to improve KNN classification accuracy, especially for mixed datasets containing both numerical and categorical features.

## ✨ Features
- Uses **log transformation** for numerical features to reduce outlier impact.
- Applies **Hamming distance** for categorical features.
- Supports different transformations like `log`, `sqrt`, and `tanh` for numerical features.
- Works well for **imbalanced datasets**.
- Can be used with **KNN, SVM, and XGBoost** classifiers.

## 📌 Installation

To use **HAAM**, first clone this repository:

```bash
git clone https://github.com/Abbasali-cmd/Hybrid-Abbas-Ali-Metric-HAAM-.git


install.packages(c("class", "e1071", "xgboost", "randomForest", "ggplot2", "caret"))

# Load required libraries
library(class)
library(e1071)
library(xgboost)
library(randomForest)
library(ggplot2)
library(caret)

# Define HAAM Function
Abbas_Ali_Distance <- function(x, y, num_indices, cat_indices, feature_weights, std_devs, transformation = "log") {
  if (transformation == "sqrt") {
    num_distance <- sum((feature_weights / std_devs) * sqrt(1 + abs(x[num_indices] - y[num_indices])))
  } else if (transformation == "tanh") {
    num_distance <- sum((feature_weights / std_devs) * tanh(abs(x[num_indices] - y[num_indices])))
  } else {
    num_distance <- sum((feature_weights / std_devs) * log1p(abs(x[num_indices] - y[num_indices])))
  }
  cat_distance <- sum(x[cat_indices] != y[cat_indices]) / length(cat_indices)
  return(num_distance + cat_distance)
}





