# Clothing Recommendation ML Pipeline

A machine learning pipeline that predicts whether a customer recommends a clothing item based on their review. The pipeline handles numerical, categorical, and text data using scikit-learn, with NLP preprocessing via spaCy.

## Getting Started

Instructions for getting a copy of the project running on your local machine.

### Dependencies

```
scikit-learn
pandas
numpy
spacy
seaborn
missingno
notebook
click
```

### Installation

1. Clone the repository:

```
git clone <your-repo-url>
cd <your-repo-folder>
```

2. Create and activate a virtual environment:

```
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Download the spaCy English language model:

```
python -m spacy download en_core_web_sm
```

5. Place the dataset in the data folder:

```
data/reviews.csv
```

6. Launch Jupyter Notebook:

```
jupyter notebook
```

## Project Instructions

The project is contained in `starter.ipynb` and is structured as follows:

### 1. Data Exploration
- Inspect shape, data types, and descriptive statistics
- Check for missing values using `missingno`
- Analyse class balance of the target variable (`Recommended IND`)
- Visualise distributions of numerical features (Age, Positive Feedback Count)
- Inspect unique values and frequency of categorical features
- Check for duplicate rows

### 2. Building the Pipeline
The pipeline uses a `ColumnTransformer` to apply different preprocessing to each feature type:

- **Numerical (Age)**: `StandardScaler`
- **Numerical skewed (Positive Feedback Count)**: `log1p` transformation followed by `StandardScaler`
- **Categorical (Division Name, Department Name, Class Name)**: `OneHotEncoder`
- **Text (Title + Review Text)**: Custom `Merger` → `SpacyLemmatizer` → `TfidfVectorizer`

### 3. Training the Pipeline
- Combines the `ColumnTransformer` with a `LogisticRegression` classifier using `make_pipeline`
- Trained on 90% of the data (`test_size=0.1`)
- Evaluated using ROC AUC score on the held-out test set

### 4. Fine-Tuning the Pipeline
- Uses `GridSearchCV` to optimise hyperparameters
- Optimised parameters: `ngram_range` (TfidfVectorizer) and `penalty` (LogisticRegression)
- Scored using `roc_auc` to account for class imbalance

## Testing

Run all cells in `starter.ipynb` sequentially from top to bottom.

### Key Evaluation Metric

ROC AUC score is used rather than accuracy due to class imbalance in the dataset (~82% recommended, ~18% not recommended).

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
```

A score above 0.8 is considered good. The baseline Logistic Regression achieves ~0.93.

## Built With

* [scikit-learn](https://scikit-learn.org/) - Machine learning pipeline and model
* [spaCy](https://spacy.io/) - NLP lemmatization
* [pandas](https://pandas.pydata.org/) - Data manipulation
* [seaborn](https://seaborn.pydata.org/) - Data visualisation
* [missingno](https://github.com/ResidentMario/missingno) - Missing value analysis

## License

[License](LICENSE.txt)