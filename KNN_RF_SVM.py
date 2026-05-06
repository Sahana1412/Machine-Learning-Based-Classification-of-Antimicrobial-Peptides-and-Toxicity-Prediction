import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("AMP_combined_final.csv")

df = df[df["Activity"].isin(["Antibacterial", "Antifungal", "Antiviral"])]
df = df[["Sequence", "Activity"]].dropna()

print("\nClass Distribution:")
print(df["Activity"].value_counts())

def extract_features(seq):

    length = len(seq)

    hydrophobic = "AILMFWYV"
    positive = "KRH"
    negative = "DE"
    cysteine = "C"

    hydro_count = sum(a in hydrophobic for a in seq)
    pos_count = sum(a in positive for a in seq)
    neg_count = sum(a in negative for a in seq)
    cys_count = sum(a in cysteine for a in seq)

    net_charge = pos_count - neg_count

    return [
        length,
        float(hydro_count) / length,
        float(pos_count) / length,
        float(neg_count) / length,
        float(cys_count) / length,
        float(net_charge) / length
    ]


X = np.array([extract_features(seq) for seq in df["Sequence"]])

le = LabelEncoder()
y = le.fit_transform(df["Activity"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

selector = SelectKBest(score_func=f_classif, k='all')
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {

    "Linear SVM": SVC(
        kernel='linear',
        C=1.0,
        class_weight='balanced'
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight='balanced',
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=7,
        p=1
    )
}

for name in models:

    print("\n==============================")
    print("Training {}".format(name))
    print("==============================")

    model = models[name]

    if name == "Random Forest":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        X_for_curve = X_train
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        X_for_curve = X_train_scaled

    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy: {:.2f}%".format(acc * 100))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                xticklabels=le.classes_,
                yticklabels=le.classes_)

    plt.title("{} Confusion Matrix".format(name))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X_for_curve,
        y_train,
        cv=5
    )

    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Training Accuracy")
    plt.plot(train_sizes, val_mean, label="Validation Accuracy")

    plt.title("{} Learning Curve".format(name))
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
