# **Credit Card Fraud Detection**

A reproducible machine-learning project for detecting fraudulent credit-card transactions using a public Kaggle dataset. This repository includes data preprocessing, exploratory data analysis (EDA), feature encoding, handling class imbalance, model training, and evaluation — all implemented in a single Jupyter notebook.

---

## **Repository Contents**

* **Credit_card_fraud.ipynb** — Main Jupyter notebook with preprocessing, SMOTE balancing, Random Forest model training, and evaluation.
* **credit_card_fraud.csv** — Dataset used by the notebook (if included).
* **README.md** — Project documentation.

---

## **Dataset**

This project uses the “Fraud Detection” dataset from Kaggle:
[https://www.kaggle.com/kartik2112/fraud-detection](https://www.kaggle.com/kartik2112/fraud-detection)

Dataset fields include:
`trans_date_trans_time, merchant, category, amt, city, state, lat, long, merch_lat, merch_long, city_pop, job, dob, trans_num, is_fraud`

*Note: In this notebook, only selected columns are used. Some columns (job, trans_num, city) are removed during preprocessing.*

---

## **Project Goal**

Build and evaluate a model to classify fraudulent transactions (`is_fraud`). The notebook demonstrates:

* Data loading and feature selection
* Removing unnecessary columns
* One-hot encoding of categorical features (`state`, `category`)
* Handling class imbalance using **SMOTE**
* Splitting the dataset into training/testing sets
* Training a **Random Forest classifier**
* Evaluating model performance using:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix

**Model Performance:**

* **Accuracy:** 0.9988
* **Recall:** 0.9990
* **Precision:** 0.9986
* **F1-Score:** 0.9988

---

## **Requirements**

Install dependencies with:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

You will also need **Jupyter Notebook / Lab**.

---

## **How to Run**

```bash
git clone https://github.com/amankr72/credit_card_fraud_prediction.git
cd credit_card_fraud_prediction
pip install -r requirements.txt   # if you add one
jupyter notebook
```

Open **Credit_card_fraud.ipynb** and run cells in order.

---

## **Notebook Workflow (Actual Steps Performed)**

* Import dataset and inspect structure
* Select relevant columns
* Drop unused fields (`job`, `trans_num`, `city`)
* Apply one-hot encoding to categorical columns
* Define **X** and **y** (`is_fraud`)
* Split data into training/testing sets
* Apply **SMOTE** for class balancing
* Train Random Forest classifier
* Evaluate results and plot confusion matrix

---

## **Possible Future Improvements**

You may add these later:

* Hyperparameter tuning
* Additional models (XGBoost, LightGBM)
* Feature engineering (distance, time-based features, customer history)
* Deployment API for real-time fraud prediction
* Automated pipeline using GitHub Actions

---

## **Contributing**

Contributions and suggestions are welcome. Please open an issue or pull request.

---

## **License**

No license included. Add an MIT License if you want others to reuse this project.

---

## **Contact**

Author: **amankr72**
GitHub: [https://github.com/amankr72](https://github.com/amankr72)

---

If you want, I can also:

✅ generate a **requirements.txt**
✅ create a **clean notebook title/intro**
✅ add **badges** (stars, python version, license, etc.)

Just tell me!
