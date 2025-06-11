# Credit Card Fraud Detection 

Credit card fraud detection is the process of identifying unauthorized or suspicious transactions on a credit card. It uses **data analytics, machine learning (ML), and rule-based systems** to spot patterns that indicate fraudulent behavior.

---

### 🔍 **How It Works: General Process**

1. **Data Collection**:

   * Transaction details: amount, location, merchant, timestamp
   * Cardholder profile: spending habits, typical locations
   * Device/browser information

2. **Data Preprocessing**:

   * Handle missing values
   * Normalize/scale features
   * Encode categorical data
   * Deal with class imbalance (fraud cases are rare)

3. **Feature Engineering**:

   * Time between transactions
   * Distance between transaction locations
   * Frequency of transactions
   * Amount deviations

4. **Modeling Techniques**:

   * **Supervised learning**: Train models on labeled data (fraud or not)

     * Logistic Regression
     * Decision Trees / Random Forest
     * Gradient Boosting (XGBoost, LightGBM)
     * Neural Networks
   * **Unsupervised learning**: For anomaly detection without labeled data

     * Autoencoders
     * Isolation Forest
     * One-Class SVM
   * **Hybrid approaches**: Combine both for better performance

5. **Model Evaluation**:

   * Accuracy is not enough due to class imbalance
   * Use metrics like:

     * Precision, Recall
     * F1-score
     * AUC-ROC
     * Confusion Matrix

6. **Deployment**:

   * Real-time or batch processing
   * Flag transactions for manual review or automatic blocking

---

### ⚠️ **Challenges**

* **Imbalanced data**: Fraud cases are very rare (<0.2%)
* **Concept drift**: Fraud patterns evolve over time
* **False positives**: Risk of rejecting legitimate transactions
* **Latency requirements**: Need real-time detection in many systems

---

### 🧪 Example Dataset

The most commonly used public dataset is the **Kaggle Credit Card Fraud Detection Dataset**, which contains European card transactions:

* 284,807 transactions
* 492 frauds
* Features are anonymized (V1–V28) + Amount, Time, and Class (0 = normal, 1 = fraud)

---

### 🧠 Sample ML Pipeline (Python/Sklearn)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler

# Load your dataset
df = pd.read_csv('creditcard.csv')
X = df.drop('Class', axis=1)
y = df['Class']

# Handle imbalance
rus = RandomUnderSampler()
X_res, y_res = rus.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

Would you like help with a **project**, **code walkthrough**, or **deploying a fraud detection model**?

