import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

print("Loading models and metadata...")
try:
    meta = joblib.load("model_metadata.pkl")
    X_test = meta['X_test']
    y_test = meta['y_test']
    classes = meta['classes']
    
    if 'feature_names' in meta:
        feat_names = meta['feature_names']
    elif 'feat_names' in meta:
        feat_names = meta['feat_names']
    else:
        feat_names = ["Length", "Hydrophobicity", "Net Charge", "Aromatic %"]
        print("Note: feature_names key missing, using default: ", feat_names)

    model_names = ["RF", "SVM_Linear", "SVM_RBF", "KNN", "XGBoost"]
    models = {}
    
    for name in model_names:
        file_path = f"{name}_model.pkl"
        if os.path.exists(file_path):
            models[name] = joblib.load(file_path)
        else:
            print(f"Warning: {file_path} not found. Skipping {name}.")

    print("Success: Metadata and models loaded.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load data. Error: {e}")
    exit()
for name, model in models.items():
    # --- Confusion Matrix ---
    plt.figure(figsize=(6,5))
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.title(f"Confusion Matrix: {name}")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"1_cm_{name}.png")
    plt.close()

    plt.figure()
    y_score = model.predict_proba(X_test)
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_test == i, y_score[:, i])
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC={auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title(f"ROC Curve: {name}")
    plt.legend(loc='lower right')
    plt.savefig(f"2_roc_{name}.png")
    plt.close()

    plt.figure()
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(y_test == i, y_score[:, i])
        plt.plot(recall, precision, label=f'{classes[i]}')
    plt.title(f"P-R Curve: {name}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f"3_pr_{name}.png")
    plt.close()

for name, model in models.items():
    plt.figure()
    if hasattr(model, 'feature_importances_'):
        vals = model.feature_importances_
        sns.barplot(x=vals, y=feat_names)
    elif hasattr(model, 'coef_'):
        vals = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        sns.barplot(x=vals[:len(feat_names)], y=feat_names)
    else:
        continue # Skip KNN
    plt.title(f"Feature Impact: {name}")
    plt.tight_layout()
    plt.savefig(f"4_feat_{name}.png")
    plt.close()

print("Calculating SHAP values (Class by Class fix)...")
try:
    explainer = shap.TreeExplainer(models["RF"])
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, feature_names=feat_names, class_names=classes, show=False)
    plt.savefig("5_shap_global.png")
    plt.close()

    for i, class_name in enumerate(classes):
        plt.figure()
        shap.summary_plot(shap_values[i], X_test, feature_names=feat_names, show=False)
        plt.title(f"SHAP Analysis: {class_name}")
        plt.tight_layout()
        plt.savefig(f"5_shap_{class_name}.png")
        plt.close()
except Exception as e:
    print(f"SHAP Plotting Error: {e}")

accs = {name: model.score(X_test, y_test) for name, model in models.items()}
plt.figure(figsize=(10,6))
sns.barplot(x=list(accs.keys()), y=list(accs.values()), palette="viridis")
plt.ylim(0, 1)
plt.title("Master Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.savefig("6_final_comparison.png")
plt.close()

print(f"Process Complete! Check your folder for 30+ PNG files.")