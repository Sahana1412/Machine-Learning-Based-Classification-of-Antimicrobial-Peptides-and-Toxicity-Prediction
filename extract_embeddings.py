import torch
import esm
import pandas as pd
import numpy as np
from tqdm import tqdm

df = pd.read_csv("AMP_combined_final.csv")
df = df[["Sequence", "Activity"]].dropna()
df.columns = ["sequence", "activity"]

df["sequence"] = df["sequence"].str.upper()
valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
def clean_sequence(seq):
    return "".join([aa for aa in seq if aa in valid_aas])

df["sequence"] = df["sequence"].apply(clean_sequence)
df = df[df["sequence"].str.len() > 0]

print("Total cleaned sequences:", len(df))

model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

print("Using device:", device)

batch_size = 32
embeddings = []
labels = []

for i in tqdm(range(0, len(df), batch_size)):

    batch_df = df.iloc[i:i+batch_size]
    sequences = batch_df["sequence"].tolist()
    activities = batch_df["activity"].tolist()

    data = [("protein", seq) for seq in sequences]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6])

    token_embeddings = results["representations"][6]

    for j, seq in enumerate(sequences):
        seq_len = len(seq)
        seq_embedding = token_embeddings[j, 1:seq_len+1].mean(0)
        embeddings.append(seq_embedding.cpu().numpy())
        labels.append(activities[j])
        
embeddings = np.array(embeddings)

embedding_df = pd.DataFrame(embeddings)
embedding_df["activity"] = labels

embedding_df.to_csv("esm_embeddings.csv", index=False)

print("✅ Embeddings saved as esm_embeddings.csv")
