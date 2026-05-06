import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

df = pd.read_csv(r"D:\AMRITA\SEMESTER 4\ML for cheminfo and bioinfo\ML_PROJECT\AMP_combined_final.csv")   # update path if needed
df = df[["Sequence", "Activity"]].dropna()

allowed = ["Antibacterial", "Antiviral", "Antifungal"]
df = df[df["Activity"].isin(allowed)]

print("Class distribution:")
print(df["Activity"].value_counts())

hydro_scale = {
    'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,
    'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,
    'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,
    'V':4.2,'W':-0.9,'Y':-1.3
}
pos = ["K", "R", "H"]
neg = ["D", "E"]
aromatic = ["F", "W", "Y"]

def extract_features(seq):
    seq = str(seq)
    length = len(seq)
    hydrophobicity = np.mean([hydro_scale.get(a, 0) for a in seq])
    net_charge = (sum(seq.count(a) for a in pos) -
                  sum(seq.count(a) for a in neg)) / length
    aromatic_pct = sum(seq.count(a) for a in aromatic) / length
    return [length, hydrophobicity, net_charge, aromatic_pct]

X = np.array(df["Sequence"].apply(extract_features).tolist())

le = LabelEncoder()
y = le.fit_transform(df["Activity"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

svm_rbf = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        probability=True
    ))
])

svm_rbf.fit(X_train, y_train)

y_pred = svm_rbf.predict(X_test)

print("\n==============================")
print("RBF SVM RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_
))

train_sizes, train_scores, test_scores = learning_curve(
    svm_rbf,
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 6),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(7,5))
plt.plot(train_sizes, train_mean, label="Training Accuracy")
plt.plot(train_sizes, test_mean, label="Validation Accuracy")
plt.xlabel("Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve – Nonlinear SVM (RBF)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
