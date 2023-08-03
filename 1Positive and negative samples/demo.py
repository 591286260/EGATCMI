import torch
from transformers import ElectraTokenizer, ElectraModel
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

circRNA_df = pd.read_csv('circRNA.csv', header=None)

miRNA_df = pd.read_csv('miRNA.csv', header=None)

model_name = 'google/electra-base-discriminator'
tokenizer = ElectraTokenizer.from_pretrained(model_name)
model = ElectraModel.from_pretrained(model_name)
model.to(device)

def extract_features(sequence):
    encoded_input = tokenizer(sequence, padding='max_length', truncation=True, max_length=128, return_tensors='pt').to(device)

    return features.tolist()

executor = ThreadPoolExecutor(max_workers=4)

circRNA_features = []
print("Processing circRNA sequences:")
with tqdm(total=len(circRNA_sequences)) as pbar:
    for sequence in circRNA_sequences:

        truncated_sequence = sequence[:20]
        circRNA_features.append(extract_features(truncated_sequence))
        pbar.update(1)

miRNA_features = []
print("Processing miRNA sequences:")
with tqdm(total=len(miRNA_sequences)) as pbar:
    futures = [executor.submit(extract_features, sequence) for sequence in miRNA_sequences]
    for future in futures:
        miRNA_features.append(future.result())
        pbar.update(1)


merged_features = circRNA_features + miRNA_features

pca = PCA(n_components=128)
merged_features = pca.fit_transform(merged_features)

circRNA_ids += miRNA_ids
df = pd.DataFrame(merged_features, columns=[f"Feature_{i+1}" for i in range(128)])
df.insert(0, "ID", circRNA_ids)

df.to_csv('ELECTRA.csv', index=False, header=False)
