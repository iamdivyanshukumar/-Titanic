# Titanic Survival Prediction

This project predicts the survival of passengers aboard the Titanic using a machine learning model. It leverages Python for data preprocessing, analysis, and training a Random Forest Classifier.

## Project Overview

The goal is to analyze the Titanic dataset, clean and preprocess the data, and train a machine learning model to predict survival outcomes based on passenger features.

## Dataset

- **train.csv**: Contains features and survival outcomes for model training.
- **test.csv**: Contains features without survival outcomes for testing.
- **gender_submission.csv**: A sample submission file with predictions for the test dataset.

## Features and Workflow

### Data Preprocessing
- Removed irrelevant columns: `PassengerId`, `Name`, `Parch`, `SibSp`, `Ticket`, and `Cabin`.
- Created a new feature `Family_size` as the sum of `SibSp`, `Parch`, and 1.
- Handled missing values:
  - Filled missing `Age` and `Fare` with their respective means.
  - Replaced missing `Embarked` values with the most frequent category (`'S'`).

### Exploratory Data Analysis (EDA)
- Visualized data distributions and survival trends using Seaborn and Matplotlib:
  - Count plots for categorical features like `Sex`.
  - KDE plots to analyze survival based on passenger class (`Pclass`).
  - Scatter plots to explore relationships between `Age`, `Fare`, and survival.

### Model Training
- Trained a Random Forest Classifier using `scikit-learn`.
- Performed hyperparameter tuning with `GridSearchCV` to optimize:
  - Number of estimators (`n_estimators`).
  - Maximum tree depth (`max_depth`).
  - Splitting criterion (`criterion`).
- Best parameters: `{ 'n_estimators': 150, 'max_depth': 10, 'criterion': 'gini' }`.

### Evaluation
- Achieved an accuracy of **85.16%** and a precision of **83.58%** on the test set.

### Output
- Created a `submission.csv` file with the following structure:
  | PassengerId | Survived |
  |-------------|----------|
  | 892         | 0        |
  | 893         | 1        |
  | ...         | ...      |

## Technology Stack

- **Programming Language**: Python
- **Libraries**:
  - **Data Processing**: `pandas`, `numpy`
  - **Visualization**: `matplotlib`, `seaborn`
  - **Machine Learning**: `scikit-learn`

## Results

The project successfully predicts survival outcomes on the test dataset with a high level of accuracy, demonstrating effective data preprocessing and model training.
