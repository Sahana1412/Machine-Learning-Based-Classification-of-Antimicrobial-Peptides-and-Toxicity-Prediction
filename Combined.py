import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

def extract_features(seq):
    length = len(seq)
    hydro_scale = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3}
    hydro_val = sum(hydro_scale.get(aa, 0) for aa in seq) / length
    pos_charge = sum(seq.count(aa) for aa in "KRH")
    neg_charge = sum(seq.count(aa) for aa in "DE")
    net_charge = pos_charge - neg_charge
    aromatic = sum(seq.count(aa) for aa in "FYW") / length
    return [length, hydro_val, net_charge, aromatic]

df = pd.read_csv(r"D:\AMRITA\SEMESTER 4\ML for cheminfo and bioinfo\ML_PROJECT\AMP_combined_final.csv")
df = df[df["Activity"].isin(["Antibacterial", "Antifungal", "Antiviral"])].dropna()

print("Extracting features...")
X = np.array([extract_features(s) for s in df["Sequence"]])
le = LabelEncoder()
y = le.fit_transform(df["Activity"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

models = {
    "RF": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "SVM_Linear": SVC(kernel='linear', probability=True, random_state=42),
    "SVM_RBF": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42) # FIXED
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train_res)
    joblib.dump(model, f"{name}_model.pkl")

metadata = {'X_test': X_test_scaled, 'y_test': y_test, 'feat_names': ["Len", "Hydro", "Charge", "Aro"], 'classes': le.classes_}
joblib.dump(metadata, "model_metadata.pkl")
print("All models saved!")