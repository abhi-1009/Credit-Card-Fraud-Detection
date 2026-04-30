## Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using anonymized transaction data. The project covers end-to-end data exploration, visualization, preprocessing, and classification using three supervised learning models.

## Problem Statement

Credit card fraud causes billions of dollars in losses annually. The goal of this project is to build a reliable classification model that can accurately distinguish fraudulent transactions from genuine ones, using a highly imbalanced real-world dataset.

## Dataset
- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `creditcard.csv`
- **Records:** 284,807 transactions
- **Features:** 30 numerical features (V1–V28 are PCA-transformed for privacy, plus `Time` and `Amount`)
- **Target:** `Class` — `0` (genuine) / `1` (fraudulent)
- **Class imbalance:** Only ~0.17% of transactions are fraudulent

## Main Challenges
| Challenge | Approach |
|---|---|
| Massive data volume | Use fast, scalable models |
| Highly imbalanced classes (~99.8% genuine) | Evaluate using Precision, Recall, F1; consider resampling if needed |
| Data privacy | Features already anonymized via PCA |
| Misclassified/unreported fraud | Use robust evaluation metrics |
| Evolving fraud patterns | Keep models simple and interpretable for easy retraining |

## Project Workflow

### 1. Data Loading & Cleaning
- Loaded dataset using pandas
- Checked for null values → **None found**
- Detected and removed duplicate rows

### 2. Exploratory Data Analysis (EDA)
- Compared fraud vs. genuine transaction amounts using `describe()`
- Plotted feature distributions using histograms
- Visualized transaction time vs. amount by class (scatter plots)
- Generated a **correlation heatmap** — key findings:
  - V2 and V5 are highly negatively correlated with `Amount`
  - V20 shows positive correlation with `Amount`

### 3. Feature Engineering
- Separated features (`X`) from target (`Y`)
- Applied 80/20 and 70/30 train-test splits depending on the model

### 4. Model Training & Evaluation

Three models were trained and compared:
#### Random Forest Classifier
- `n_estimators=50`, `max_depth=10`, `max_features='sqrt'`
- Train/test split: 80/20

#### Logistic Regression
- Hyperparameter tuning via `GridSearchCV` over `C = [0.01, 0.1, 1, 10, 100, 1000]`
- 5-fold cross-validation optimizing ROC-AUC
- Best `C = 0.01`; ROC-AUC on train set: **0.91**

#### Decision Tree Classifier
- Default `DecisionTreeClassifier`
- Train/test split: 70/30

## Results
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | 1.00 | **0.97** | 0.71 | **0.82** |
| Logistic Regression | 1.00 | 0.84 | 0.66 | 0.74 |
| Decision Tree | 1.00 | 0.76 | **0.74** | 0.75 |

> Note: All models report 100% accuracy due to the severe class imbalance. **Precision, Recall, and F1-Score** are more meaningful metrics here.

**Winner: Random Forest** — highest precision (0.97) and F1-Score (0.82), making it the best model for minimizing false fraud alerts while still catching most actual fraud.

## Tech Stack
- **Language:** Python 3
- **Libraries:**
  - Data: `numpy`, `pandas`
  - Visualization: `matplotlib`, `seaborn`
  - ML: `scikit-learn` (Random Forest, Logistic Regression, Decision Tree, GridSearchCV, KFold)

## Getting Started

### Prerequisites
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```
### Run the Notebook
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Update the file path in the notebook:
   ```python
   data = pd.read_csv('path/to/creditcard.csv')
   ```
3. Open and run `creditcard_fraud.ipynb` cell by cell

## Project Structure
```
credit-card-fraud-detection/
│
├── creditcard_fraud.ipynb   # Main notebook (EDA + model training)
├── creditcard.csv           # Dataset (download from Kaggle)
└── README.md
```
## Future Improvements
- Apply **SMOTE** or **undersampling** to handle class imbalance more explicitly
- Test **XGBoost** or **LightGBM** for better recall on fraud cases
- Add **threshold tuning** to optimize the precision-recall tradeoff
- Build a real-time scoring pipeline using a REST API

## License
This project is for educational and research purposes. The dataset is provided by [ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) under its respective terms.

## Author
**Abhijit Sinha**
- GitHub: [@abhi-1009](https://github.com/abhi-1009)
- LinkedIn: [abhijit-sinha-053b159a](https://linkedin.com/in/abhijit-sinha-053b159a)
- Email: sinhaabhijit12@yahoo.com
