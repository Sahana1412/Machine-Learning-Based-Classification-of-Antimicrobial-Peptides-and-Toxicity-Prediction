import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
model = joblib.load('xgb_esm2_model.pkl') 
X_test_esm2 = joblib.load('X_test_data.pkl') 
X_sample = X_test_esm2[:100] 
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, 
    X_sample, 
    plot_type="bar", 
    max_display=20, 
    show=False
)

plt.title("Top 20 ESM-2 Embeddings by SHAP Importance", fontsize=14)
plt.xlabel("Mean SHAP Value (Impact on Model Output)")
plt.tight_layout()
plt.savefig('XGBoost_ESM2_SHAP_Top20.png', dpi=300)
print("Success! Plot saved as 'XGBoost_ESM2_SHAP_Top20.png'")