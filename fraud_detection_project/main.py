import pandas as pd
import numpy as np
import os
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
import joblib

warnings.filterwarnings('ignore')

# --------- 2. LOAD OR GENERATE DATA ----------
file_path = "creditcard.csv"

if os.path.exists(file_path):
    data = pd.read_csv(file_path)
    print("Dataset Loaded from creditcard.csv\n")
else:
    print("creditcard.csv not found — generating synthetic dataset...\n")
    np.random.seed(42)
    n_legit  = 9000
    n_fraud  = 492  # mirrors real dataset ratio

    # 28 PCA-like features (V1–V28) + Amount + Class
    legit_features = np.random.randn(n_legit, 28) * np.random.uniform(0.5, 3, 28)
    fraud_features = np.random.randn(n_fraud, 28) * np.random.uniform(0.5, 3, 28) + \
                     np.random.uniform(-2, 2, 28)  # slight mean shift

    legit_amount  = np.abs(np.random.exponential(scale=100, size=n_legit))
    fraud_amount  = np.abs(np.random.exponential(scale=300, size=n_fraud))

    cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

    legit_df = pd.DataFrame(
        np.column_stack([legit_features, legit_amount, np.zeros(n_legit)]),
        columns=cols
    )
    fraud_df = pd.DataFrame(
        np.column_stack([fraud_features, fraud_amount, np.ones(n_fraud)]),
        columns=cols
    )

    data = pd.concat([legit_df, fraud_df], ignore_index=True).sample(frac=1, random_state=42)
    data['Class'] = data['Class'].astype(int)
    print(f"Synthetic dataset created: {len(data)} rows ({n_fraud} fraud, {n_legit} legit)\n")

# --------- 3. DATA CLEANING ----------
print("Cleaning Data...")
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)
print("Data Cleaned\n")

# --------- 4. EDA ----------
print("Performing EDA...")

print("\nClass Distribution:")
print(data['Class'].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
data['Class'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue', 'tomato'])
axes[0].set_title("Fraud vs Normal Transactions")
axes[0].set_xlabel("Class (0=Normal, 1=Fraud)")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(['Normal', 'Fraud'], rotation=0)

# Amount distribution by class
data[data['Class'] == 0]['Amount'].hist(bins=50, ax=axes[1], alpha=0.6, label='Normal', color='steelblue')
data[data['Class'] == 1]['Amount'].hist(bins=50, ax=axes[1], alpha=0.6, label='Fraud',  color='tomato')
axes[1].set_title("Transaction Amount by Class")
axes[1].set_xlabel("Amount")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.close()
print("EDA plots saved.\n")

# --------- 5. PREPROCESSING ----------
print("Preprocessing Data...")

scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data[['Amount']])

X = data.drop("Class", axis=1)
y = data["Class"]

# Manual SMOTE-style oversampling (minority class upsampled to match majority)
df_combined = pd.concat([X, y], axis=1)
majority = df_combined[df_combined['Class'] == 0]
minority = df_combined[df_combined['Class'] == 1]

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

balanced = pd.concat([majority, minority_upsampled]).sample(frac=1, random_state=42)
X = balanced.drop("Class", axis=1)
y = balanced["Class"]

print(f"Data Balanced — Class distribution after resampling:\n{y.value_counts()}\n")

# --------- 6. TRAIN-TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}\n")

# --------- 7. MODEL TRAINING ----------
print("Training Models...")

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Models Trained\n")

# --------- 8. PREDICTIONS ----------
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# --------- 9. EVALUATION ----------
print("Evaluating Models...\n")

print("=== Logistic Regression ===")
lr_acc = accuracy_score(y_test, lr_pred)
print(f"Accuracy: {lr_acc:.4f}")
print(confusion_matrix(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

print("\n=== Random Forest ===")
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Accuracy: {rf_acc:.4f}")
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# --------- 10. CONFUSION MATRIX PLOTS ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, pred, title in zip(
    axes,
    [lr_pred, rf_pred],
    ["Logistic Regression", "Random Forest"]
):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    ax.set_title(f"{title}\nAccuracy: {accuracy_score(y_test, pred):.4f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()
print("Confusion matrix plots saved.\n")

# --------- 11. MODEL COMPARISON ----------
print("\nModel Comparison:")
print(f"  Logistic Regression Accuracy : {lr_acc:.4f}")
print(f"  Random Forest Accuracy       : {rf_acc:.4f}")

if rf_acc >= lr_acc:
    best_model = rf_model
    print("  Best Model: Random Forest")
else:
    best_model = lr_model
    print("  Best Model: Logistic Regression")

# --------- 12. SAVE MODEL ----------
joblib.dump(best_model, "fraud_model.pkl")
print("\nModel saved as fraud_model.pkl")

# --------- 13. SAMPLE PREDICTION ----------
print("\nTesting Sample Prediction...")

sample = X_test.iloc[0].values.reshape(1, -1)
prediction = best_model.predict(sample)
actual     = y_test.iloc[0]

print(f"  Actual class  : {'Fraud' if actual == 1 else 'Genuine'}")
print(f"  Predicted class: {'FRAUDULENT TRANSACTION DETECTED!' if prediction[0] == 1 else 'Genuine Transaction'}")

