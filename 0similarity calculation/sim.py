import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

input_file = "lncRNA2009.csv"
data = pd.read_csv(input_file, header=None)

sequences = list(data[1])

lengths = [len(seq) for seq in sequences]

max_length = max(lengths)

def pad_sequence(seq):
    num_padding = max_length - len(seq)
    padded_seq = seq + "N" * num_padding
    return padded_seq

padded_sequences = []
with ThreadPoolExecutor() as executor:
    with tqdm(total=len(sequences), desc="Padding sequences", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}") as pbar:
        for seq in sequences:
            padded_seq = pad_sequence(seq)
            padded_sequences.append(padded_seq)
            pbar.update(1)

sequences_tensor = torch.tensor([[ord(c) for c in seq] for seq in padded_sequences], device=device, dtype=torch.float32)

matrix = torch.zeros((len(sequences), len(sequences)), device=device)
with tqdm(total=len(sequences) * (len(sequences) + 1) // 2, desc="Calculating cosine similarity") as pbar:
    for i in range(len(sequences)):
        for j in range(i, len(sequences)):
            a = sequences_tensor[i]
            b = sequences_tensor[j]
            cosine_sim = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
            matrix[i][j] = cosine_sim
            matrix[j][i] = cosine_sim
            pbar.update(1)

df = pd.DataFrame(matrix.cpu().numpy())
df.to_csv("cosine_lnc_matrix.csv", index=False, header=False)
