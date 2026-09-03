# Titanic - Machine Learning from Disaster

A classic **machine learning classification project** for **passenger survival prediction** using the **Titanic** dataset, built with **scikit-learn**. The goal is to predict whether a passenger survived based on demographic and travel information, comparing multiple classical ML algorithms before selecting and tuning a final model.

<p align="center">
  <img src="https://img.shields.io/badge/MachineLearning-Classification-green">
  <img src="https://img.shields.io/badge/Models-7%20Compared-blue">
  <img src="https://img.shields.io/badge/Dataset-Titanic-purple">
  <img src="https://img.shields.io/badge/Language-Python-yellow">
  <img src="https://img.shields.io/badge/Framework-Scikit--learn-red">
</p>

------

# Overview
This project tackles the classic **Titanic survival prediction** problem, a foundational benchmark for tabular classification.

The workflow progresses from data cleaning to a production-style tuned pipeline:
- Clean the raw data and explore survival patterns
- Engineer new features from the raw columns
- Train and tune **7 baseline classifiers** for comparison
- Build a single **preprocessing + model pipeline** for the final estimator
- Evaluate on the official Kaggle test set and compare all approaches

------

# Dataset
- **Name:** Titanic - Machine Learning from Disaster
- **Source:** [Kaggle - Titanic Competition](https://www.kaggle.com/competitions/titanic)
- **Target:** `Survived` (0 = No, 1 = Yes)
- **Files used:** `train.csv`, `test.csv`, `gender_submission.csv`

## Data Characteristics
- Mixed numeric and categorical passenger data (class, age, sex, fare, embarkation port, family relations)
- Missing values in `Age`, `Embarked`, and `Cabin` (`Cabin` mostly missing and dropped entirely)
- Outliers in `SibSp` and `Parch` for a small number of large families
- Class imbalance between survivors and non-survivors

------

# Project Workflow

## 1) Data Cleaning & Exploratory Analysis
`Cleaning.ipynb`
- Dropped high-missing / non-predictive columns: `Cabin`, `Ticket`, `Name`, `PassengerId`
- Imputed missing `Age` with the mean and missing `Embarked` with the mode (`'S'`)
- Removed outlier records with unusually high `SibSp` / `Parch` values
- Renamed `Sex` → `Gender` for clarity
- Visualized survival rate by fare group, gender, passenger class, embarkation port, age group, and family size
- Analyzed the relationship between `SibSp` and `Parch` via a cross-tab heatmap

### Key EDA Findings

**Survival Rate by Age Group and Gender**
- Across every age group, females had a substantially higher survival rate than males — the gap is especially large for the Teen, Young Adult, Adult, and Senior groups.

<img width="584" height="455" alt="survival_by_agegroup" src="https://github.com/user-attachments/assets/d58687ae-3fb4-4bfe-86f8-3b40ac672fd1" />


**Survival Rate by Fare Group and Gender**
- Survival rate increases with fare group (a proxy for passenger wealth) for both genders, reaching its maximum in the Very High fare bracket.

<img width="584" height="455" alt="survival_by_faregroup" src="https://github.com/user-attachments/assets/588f3424-ba94-45cd-a1fa-7e96ec33c3c1" />


**Survival Rate by Passenger Class**
- Ticket class had a direct impact on survival: 1st class passengers (both female and male) survived at clearly higher rates than 2nd and 3rd class, with women in 1st and 2nd class surviving almost universally.

<img width="584" height="455" alt="survival_by_pclass" src="https://github.com/user-attachments/assets/e5651d96-5dd6-40d7-afe0-ac56d76d4b30" />


## 2) Feature Engineering
`Baseline Models/final_data.py`
- **FamilySize** = `SibSp` + `Parch` + 1
- **IsAlone** = 1 if `FamilySize` == 1, else 0
- **FarePerPerson** = `Fare` / `FamilySize`
- **FareGroup**: `Fare` binned into *Low, Medium, High, Very High*
- **AgeGroup**: `Age` binned into *Child, Teen, Young Adult, Adult, Senior*
- Shared `titanic_data()` / `titanic_test_data()` helper functions applied identical cleaning and feature engineering to both the train and test sets

## 3) Baseline Model Comparison
`Baseline Models/*.ipynb`
Seven classical classifiers were trained and hyperparameter-tuned independently to find each algorithm's best configuration:
- Support Vector Machine (SVM)
- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- Decision Tree
- AdaBoost

## 4) Final Model
`Final Model.py`
- Combined preprocessing and modeling into a single **scikit-learn Pipeline**:
  - Numeric features (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `FamilySize`, `FarePerPerson`, `IsAlone`) → median imputation + `StandardScaler`
  - Categorical features (`Embarked`, `Gender`, `FareGroup`, `AgeGroup`) → `OneHotEncoder`
- Final estimator: a tuned **Decision Tree** (`criterion='entropy'`, `max_depth=10`, `ccp_alpha=0.01`)
- Trained on the full training set and evaluated on the official Kaggle `test.csv` against `gender_submission.csv`

------

# Project Structure
```
Titanic - Machine Learning from Disaster/
├── README.md
├── images/
│   ├── survival_by_agegroup.png
│   ├── survival_by_pclass.png
│   └── survival_by_faregroup.png
├── Cleaning.ipynb              # Data cleaning & exploratory analysis
├── Final Model.py              # Final preprocessing + model pipeline
└── Baseline Models/
    ├── final_data.py           # Shared cleaning & feature engineering functions
    ├── SVM.ipynb
    ├── Random Forest.ipynb
    ├── Logistic Regression.ipynb
    ├── KNN.ipynb
    ├── Gradient Boosting.ipynb
    ├── Decision Tree.ipynb
    └── AdaBoost.ipynb
```

------

# How to Run

## 1) Clone the repository
```bash
git clone https://github.com/<your-username>/titanic-machine-learning-from-disaster.git
cd "titanic-machine-learning-from-disaster"
```

## 2) Install dependencies
```bash
pip install -r requirements.txt
```
Or directly:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 3) Get the data
Download `train.csv`, `test.csv`, and `gender_submission.csv` from the [Kaggle Titanic competition](https://www.kaggle.com/competitions/titanic) and update the file paths inside `Cleaning.ipynb`, `Baseline Models/final_data.py`, and `Final Model.py` to point to your local copies.

## 4) Explore & clean the data
```bash
jupyter notebook "Cleaning.ipynb"
```

## 5) Compare baseline models
Run any notebook inside `Baseline Models/` (e.g. `SVM.ipynb`, `Decision Tree.ipynb`) to reproduce the tuning and evaluation for that algorithm.

## 6) Train & evaluate the final model
```bash
python "Final Model.py"
```
This builds the full preprocessing + Decision Tree pipeline, fits it on the training data, and prints the confusion matrix and classification report on the official test set.

------

# Results Summary

| Model | Accuracy | Precision (Survived) | Recall (Survived) | F1 (Survived) |
|---|---|---|---|---|
| KNN | 0.83 | 0.75 | 0.82 | 0.78 |
| Gradient Boosting | 0.86 | 0.85 | 0.76 | 0.80 |
| Random Forest | 0.86 | 0.88 | 0.73 | 0.80 |
| AdaBoost | 0.88 | 0.85 | 0.80 | 0.82 |
| SVM | 0.96 | 0.91 | 0.98 | 0.95 |
| Logistic Regression | 0.96 | 0.94 | 0.95 | 0.95 |
| Decision Tree | 0.96 | 0.94 | 0.96 | 0.95 |
| **Final Model (tuned pipeline)** | **0.97** | **0.95** | **0.97** | **0.96** |

> Random Forest and Gradient Boosting were evaluated on a smaller internal split (216 samples); all other models were evaluated on the full official test set (418 samples).

The final tuned Decision Tree pipeline outperformed every individually-tuned baseline, achieving **97% accuracy** and the best overall precision/recall balance on the official test set.

------

# Key Takeaways
- Engineered features (`FamilySize`, `IsAlone`, `FarePerPerson`, `FareGroup`, `AgeGroup`) meaningfully improved on the raw columns alone
- Tree-based and margin-based models (Decision Tree, SVM, Logistic Regression) clearly outperformed distance-based and weak-learner ensemble approaches (KNN, AdaBoost, Gradient Boosting) on this dataset size
- Wrapping preprocessing and the tuned model into a single pipeline made evaluation on the true Kaggle test set straightforward and reproducible
- Gender and passenger class were the strongest single predictors of survival, consistent with the "women and children first, higher class first" evacuation pattern

------

# Limitations
- Age imputation used a simple mean fill rather than a more informed strategy (e.g. by title or class)
- `Cabin` was dropped entirely instead of extracting a partial-availability or deck feature
- No formal cross-validation grid search is documented for every baseline model; results reflect each notebook's best found configuration
- Random Forest and Gradient Boosting results are not directly comparable to the others due to a different evaluation split

------

# Libraries Used
- `pandas` / `numpy`
- `matplotlib` / `seaborn`
- `scikit-learn` (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, classifiers, `classification_report`)

------

# Author ✍️
**Author:** Ali  
**Field:** Data Science & Machine Learning Student  
**Email:** ali.hz87980@gmail.com  
**GitHub:** [ali-119](https://github.com/ali-119)
