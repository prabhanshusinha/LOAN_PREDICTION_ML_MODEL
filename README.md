# 🏦 Loan Prediction Model using Machine Learning

A Machine Learning classification project that predicts whether a loan application will be approved based on applicant and financial information.

## 📌 Project Overview

This project uses the **Loan Approval Prediction Dataset** and demonstrates a complete basic Machine Learning workflow:

* Data loading and exploration
* Data preprocessing
* Categorical feature encoding
* Train-test split
* Feature selection
* Decision Tree Classification
* Random Forest Classification
* Model evaluation
* Cross-validation

## 📊 Dataset

The dataset contains information about loan applicants, including:

* Number of dependents
* Education
* Self-employment status
* Annual income
* Loan amount
* Loan term
* CIBIL score
* Residential assets value
* Commercial assets value
* Luxury assets value
* Bank assets value
* Loan status

The target variable is:

**`loan_status`** — whether the loan is approved or not.

## 🔧 Technologies & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Kaggle Notebook

## ⚙️ Machine Learning Workflow

### 1. Data Loading

The dataset is loaded using Pandas:

```python
df = pd.read_csv("loan_approval_dataset.csv")
```

### 2. Data Exploration

Initial exploration is performed using:

```python
df.head()
df.shape
df.info()
df.isnull().sum()
df.describe()
```

Categorical values are also inspected using `unique()`.

### 3. Categorical Encoding

`OrdinalEncoder` is used to convert categorical features into numerical values.

The following features are encoded:

* `education`
* `self_employed`

Example:

```python
from sklearn.preprocessing import OrdinalEncoder

encode1 = OrdinalEncoder(
    categories=[[" Graduate", " Not Graduate"]]
)

encode2 = OrdinalEncoder(
    categories=[[" Yes", " No"]]
)
```

The original categorical columns are then removed after encoding.

### 4. Train-Test Split

The dataset is divided into training and testing sets using an **80/20 split**:

```python
x_train, x_test, y_train, y_test = train_test_split(
    df.drop(columns=[" loan_status"]),
    df[" loan_status"],
    test_size=0.2,
    random_state=42
)
```

### 5. Target Encoding

`LabelEncoder` is used to convert the target variable `loan_status` into numerical values.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
```

### 6. Feature Selection

`SelectKBest` with the `f_classif` scoring function is used to select the **10 best features**.

```python
SelectKBest(score_func=f_classif, k=10)
```

### 7. Decision Tree Classifier 🌳

A Decision Tree classifier is used as one of the classification models.

The feature selection and classifier are combined using a Scikit-learn pipeline:

```python
pipe = make_pipeline(
    SelectKBest(score_func=f_classif, k=10),
    DecisionTreeClassifier()
)
```

### 8. Random Forest Classifier 🌲

A Random Forest classifier is also trained:

```python
from sklearn.ensemble import RandomForestClassifier

pipe = make_pipeline(
    SelectKBest(score_func=f_classif, k=10),
    RandomForestClassifier()
)
```

## 📈 Model Evaluation

The models are evaluated using:

### Accuracy

```python
accuracy_score(y_predict, y_test)
```

### Precision

```python
cross_val_score(
    pipe,
    x_train,
    y_train,
    cv=5,
    scoring="precision"
).mean()
```

### Recall

```python
cross_val_score(
    pipe,
    x_train,
    y_train,
    cv=5,
    scoring="recall"
).mean()
```

### F1 Score

```python
cross_val_score(
    pipe,
    x_train,
    y_train,
    cv=5,
    scoring="f1"
).mean()
```

The project uses **5-fold cross-validation** to evaluate model performance more reliably.

## 📁 Repository Structure

```text
loan-prediction-ml/
│
├── loan-prediction-ml.ipynb
└── README.md
```

## 🚀 How to Run

1. Clone the repository.

```bash
git clone <your-repository-url>
```

2. Open the notebook:

```text
loan-prediction-ml.ipynb
```

3. Install the required libraries if necessary:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the notebook cells sequentially.

## 🎯 Learning Objectives

This project was created to practice and understand:

* Pandas data handling
* Exploratory Data Analysis
* Categorical encoding
* Train-test splitting
* Feature selection
* Machine Learning pipelines
* Decision Tree Classification
* Random Forest Classification
* Accuracy, precision, recall and F1-score
* Cross-validation

## 🔮 Future Improvements

Possible improvements include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Comparing additional classification algorithms
* Adding confusion matrix and classification report
* Improving feature engineering
* Handling class imbalance if present
* Deploying the trained model as a web application

## 👨‍💻 Author

**Prabhanshu Kumar**

B.Tech — Data Science

---

⭐ If you found this project useful, consider giving the repository a star!
