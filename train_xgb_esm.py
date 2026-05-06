import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

df = pd.read_csv("esm_embeddings.csv")
target_classes = ["Antibacterial", "Antifungal", "Antiviral"]
df = df[df["activity"].isin(target_classes)]

X = df.drop("activity", axis=1).values
y = df["activity"].values

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Training XGBoost...")
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    tree_method='hist',
    early_stopping_rounds=50, 
    random_state=42
)

model.fit(
    X_train, 
    y_train, 
    eval_set=[(X_test, y_test)], 
    verbose=False
)

print("\nSaving model and metadata...")
joblib.dump(model, 'xgb_esm2_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(X_test, 'X_test_data.pkl')
joblib.dump(y_test, 'y_test_data.pkl')
y_pred = model.predict(X_test)
print(f"\nFinal Accuracy: {round(accuracy_score(y_test, y_pred), 4)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Confusion Matrix - XGBoost (ESM2)")
plt.savefig("confusion_matrix.png")
plt.show()
print("\nGenerating SHAP plots (this may take a minute)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
plt.figure()
shap.summary_plot(shap_values, X_test, class_names=le.classes_, show=False)
plt.title("SHAP Feature Importance (ESM2 Embeddings)")
plt.savefig("shap_summary.png")
plt.show()

print("\nSuccess! All models saved and 2 key graphs generated.")