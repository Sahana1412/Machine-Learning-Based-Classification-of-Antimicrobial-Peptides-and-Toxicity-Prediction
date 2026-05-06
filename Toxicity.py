import torch
import numpy as np
import joblib
from transformers import AutoTokenizer, EsmModel

try:
    activity_model = joblib.load('xgb_esm2_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("Models loaded successfully.")
except FileNotFoundError:
    print("Error: .pkl files not found! Ensure they are in the current directory.")

model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
esm_model = EsmModel.from_pretrained(model_name)

def get_esm_embedding(sequence):
    """Generates the 320-dimensional vector the model needs."""
    inputs = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = esm_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def calculate_toxicity(sequence):
    """Rule-based toxicity check: NetCharge > 0.4, Hydro > 0.5, Aromatic > 0.1"""
    L = len(sequence)
    if L == 0: return "Unknown", 0
    pos = sum(sequence.count(a) for a in "KRH")
    neg = sum(sequence.count(a) for a in "DE")
    net_charge = (pos - neg) / L
    kd = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 
          'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 
          'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}
    hydro = sum(kd.get(aa, 0) for aa in sequence) / L
    aromatic_pct = sum(sequence.count(a) for a in "FWY") / L
    is_toxic = (net_charge > 0.4 and hydro > 0.5 and aromatic_pct > 0.1)
    return ("High Toxicity" if is_toxic else "Low Toxicity"), net_charge, hydro, aromatic_pct

def predict_amp(sequence):
    embedding = get_esm_embedding(sequence)
    class_idx = activity_model.predict(embedding)[0]
    class_label = label_encoder.inverse_transform([class_idx])[0]
    probs = activity_model.predict_proba(embedding)[0]
    confidence = np.max(probs) * 100
    tox_label, nc, hy, ar = calculate_toxicity(sequence)
    print("\n" + "="*50)
    print(f"PEPTIDE ANALYSIS: {sequence}")
    print("="*50)
    print(f"PREDICTED CLASS : {class_label.upper()} ({confidence:.2f}% Confidence)")
    print(f"TOXICITY STATUS : {tox_label}")
    print("-" * 50)
    print(f"BIO-PHYSICAL STATS:")
    print(f" > Normalized Net Charge: {nc:.2f}")
    print(f" > Hydrophobicity Score:  {hy:.2f}")
    print(f" > Aromatic Percentage:  {ar*100:.1f}%")
    print("="*50 + "\n")
seq_input = input("Enter Peptide Sequence: ").upper().strip()
predict_amp(seq_input)