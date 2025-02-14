from __future__ import division
from __future__ import print_function
from transformers import ElectraTokenizer, ElectraModel
from concurrent.futures import ThreadPoolExecutor
from torch.nn.functional import pairwise_distance
from gensim.models import Word2Vec
from tqdm import tqdm
from sklearn.decomposition import PCA
import warnings
import os
import glob
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import csv
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models import GAT
warnings.filterwarnings('ignore')
np.random.seed(1337)
def read_csv(file_name):
    with open(file_name, newline='', encoding='utf-8') as csvfile:
        return list(csv.reader(csvfile))
def save_to_csv(data, file_name):
    with open(file_name, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
def generate_negative_samples(positive_samples, all_circRNA, all_miRNA):
    positive_set = set(map(tuple, positive_samples))
    negative_samples = []
    while len(negative_samples) < len(positive_samples):
        circRNA = random.choice(all_circRNA)
        miRNA = random.choice(all_miRNA)
        if (miRNA, circRNA) not in positive_set and [miRNA, circRNA] not in negative_samples:
            negative_samples.append([miRNA, circRNA])
    return negative_samples
original_data = read_csv("./data/interaction9905.csv")
circ_miRNA = [[row[0], row[1]] for row in original_data]
all_circRNA = list(set(row[1] for row in original_data))
all_miRNA = list(set(row[0] for row in original_data))
positive_samples = circ_miRNA
negative_samples = generate_negative_samples(positive_samples, all_circRNA, all_miRNA)
save_to_csv(negative_samples, 'NegativeSamples.csv')
data_final = np.vstack((original_data, negative_samples))
save_to_csv(data_final, 'PositiveAndNegativeSamples.csv')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
circRNA_df = pd.read_csv('./data/circRNA_final_2346.csv', header=None)
miRNA_df = pd.read_csv('./data/miRNA_final_962.csv', header=None)
circRNA_sequences = circRNA_df.iloc[:, 1].tolist()  # CircRNA序列
miRNA_sequences = miRNA_df.iloc[:, 1].tolist()  # miRNA序列
model_name = 'google/electra-base-discriminator'
tokenizer = ElectraTokenizer.from_pretrained(model_name)
model = ElectraModel.from_pretrained(model_name)
model.to(device)
def extract_features(sequence):
    encoded_input = tokenizer(sequence, padding='max_length', truncation=True, max_length=128, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model(**encoded_input)
        return output.last_hidden_state.mean(dim=1).squeeze().tolist()
executor = ThreadPoolExecutor(max_workers=4)
def process_sequences(sequences, feature_list):
    with tqdm(total=len(sequences)) as pbar:
        futures = [executor.submit(extract_features, seq) for seq in sequences]
        for future in futures:
            feature_list.append(future.result())
            pbar.update(1)
circRNA_features = []
process_sequences(circRNA_sequences, circRNA_features)
miRNA_features = []
process_sequences(miRNA_sequences, miRNA_features)
merged_features = circRNA_features + miRNA_features
pca = PCA(n_components=128)
merged_features = pca.fit_transform(merged_features)
merged_ids = circRNA_df.iloc[:, 0].tolist() + miRNA_df.iloc[:, 0].tolist()  # 合并ID列表
df = pd.DataFrame(merged_features, columns=[f"Feature_{i + 1}" for i in range(128)])  # 特征列名
df.insert(0, "ID", merged_ids)  # 插入ID列
df.to_csv('./output/ELECTRA.csv', index=False, header=False)
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    ids = data.iloc[:, 0].tolist()
    sequences = [seq.split() for seq in data.iloc[:, 1].tolist()]
    return ids, sequences
def train_word2vec_model(circRNA_sequences, miRNA_sequences, vector_size=64, window=7):
    model = Word2Vec(sentences=circRNA_sequences + miRNA_sequences, vector_size=vector_size, window=window, min_count=1, workers=4)
    return model
def calculate_similarity(sequences, model, device):
    word2vec_model_vectors = torch.tensor(model.wv.vectors).to(device)
    similarity_matrix = []
    for seq1 in tqdm(sequences, desc="Calculating similarities"):
        seq1_tokens = [model.wv.key_to_index[token] for token in seq1 if token in model.wv]
        if not seq1_tokens:  # Skip empty sequences
            continue
        seq1_tokens_tensor = torch.tensor(seq1_tokens).to(device)
        seq1_embedding = torch.mean(word2vec_model_vectors[seq1_tokens_tensor], dim=0, keepdim=True)
        sim_row = []
        for seq2 in sequences:
            seq2_tokens = [model.wv.key_to_index[token] for token in seq2 if token in model.wv]
            if not seq2_tokens:  # Skip empty sequences
                continue
            seq2_tokens_tensor = torch.tensor(seq2_tokens).to(device)
            seq2_embedding = torch.mean(word2vec_model_vectors[seq2_tokens_tensor], dim=0, keepdim=True)
            distance = pairwise_distance(seq1_embedding, seq2_embedding)
            sim_row.append(distance.item())
        similarity_matrix.append(sim_row)
    return similarity_matrix
def perform_pca(X, n_components=3):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)
def save_to_csv(data, file_path):
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, header=False)
circRNA_ids, circRNA_sequences = load_data('data/circRNA_final_2346.csv')
miRNA_ids, miRNA_sequences = load_data('data/miRNA_final_962.csv')
word2vec_model = train_word2vec_model(circRNA_sequences, miRNA_sequences)
word2vec_model.save('word2vec_model.bin')
word2vec_model = Word2Vec.load('word2vec_model.bin')
circRNA_sim_matrix = calculate_similarity(circRNA_sequences, word2vec_model, device)
miRNA_sim_matrix = calculate_similarity(miRNA_sequences, word2vec_model, device)
circRNA_sim_matrix = pd.DataFrame(circRNA_sim_matrix)
miRNA_sim_matrix = pd.DataFrame(miRNA_sim_matrix)
circRNA_sim_matrix.insert(0, 'ID', circRNA_ids)
miRNA_sim_matrix.insert(0, 'ID', miRNA_ids)
data = circRNA_sim_matrix
ids = data.iloc[:, 0]  # 第一列为 id
X = data.iloc[:, 1:].values  # 其余列为特征数据
X_pca = perform_pca(X, n_components=2)
df_pca_circRNA = pd.DataFrame(X_pca)
df_pca_circRNA.insert(0, 'id', ids)
data = miRNA_sim_matrix
ids = data.iloc[:, 0]  # 第一列为 id
X = data.iloc[:, 1:].values  # 其余列为特征数据
X_pca = perform_pca(X, n_components=2)
df_pca_miRNA = pd.DataFrame(X_pca)
df_pca_miRNA.insert(0, 'id', ids)
merged_df = pd.concat([df_pca_circRNA, df_pca_miRNA], axis=0)
merged_df.to_csv("./output/WMD.csv", index=False, header=False)
from utils import load_data, accuracy
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=250, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
adj, features, labels, idx_train, idx_val, idx_test = load_data()
model = GAT(nfeat=features.shape[1],
        nhid=args.hidden,
        nclass=32,
        dropout=args.dropout,
        nheads=args.nb_heads,
        alpha=args.alpha)
optimizer = optim.Adam(model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)
if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
features, adj, labels = Variable(features), Variable(adj), Variable(labels)
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    if not args.fastmode:
        model.eval()
        output = model(features, adj)
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_val.data.item()
def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("测试集结果:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))
t_total = time.time()
loss_values = []
bad_counter = 0
best = args.epochs + 1
best_epoch = 0
for epoch in range(args.epochs):
    loss_values.append(train(epoch))
    torch.save(model.state_dict(), '{}.pkl'.format(epoch))
    if loss_values[-1] < best:
        best = loss_values[-1]
        best_epoch = epoch
        bad_counter = 0
    else:
        bad_counter += 1
    if bad_counter == args.patience:
        break
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb < best_epoch:
            os.remove(file)
files = glob.glob('*.pkl')
for file in files:
    epoch_nb = int(file.split('.')[0])
    if epoch_nb > best_epoch:
        os.remove(file)
model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
best_features = model(features, adj)
best_features_cpu = best_features.cpu()  # 将特征张量从GPU内存复制到主机内存
df = pd.DataFrame(best_features_cpu.detach().numpy())
df.to_csv('./output/GAT.csv', index=False, header=False)
compute_test()
with open('data/id.csv', 'r', newline='') as id_file:
    id_reader = csv.reader(id_file)
    # 读取每一行的第一列数据，即id列，并存储为id_data列表
    id_data = [row[0] for row in id_reader]
with open('output/GAT.csv', 'r', newline='') as target_file:
    target_reader = csv.reader(target_file)
    target_data = [row for row in target_reader]
for i, row in enumerate(target_data):
    if i < len(id_data):
        row.insert(0, id_data[i])
with open('output/GAT.csv', 'w', newline='') as target_file:
    target_writer = csv.writer(target_file)
    target_writer.writerows(target_data)
df1 = pd.read_csv('output/ELECTRA.csv', header=None, index_col=0, encoding='gbk')
df2 = pd.read_csv('output/WMD.csv', header=None, index_col=0, encoding='gbk')
df3 = pd.read_csv('output/GAT.csv', header=None, index_col=0, encoding='gbk')
merged_df = pd.concat([df1, df2, df3], axis=1)
merged_df.to_csv('./output/F.csv', header=None, encoding='gbk')
def read_feature_file(file_name):
    feature_dict = {}
    with open(file_name, 'r', encoding='gbk') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            id_value = row[0]
            feature_value = row[1:]
            feature_dict[id_value] = feature_value
    return feature_dict
def replace_id_with_feature(input_file_name, feature_dict, output_file_name):
    with open(input_file_name, 'r', encoding='gbk') as csvfile:
        with open(output_file_name, 'w', newline='', encoding='gbk') as output_csvfile:
            csv_writer = csv.writer(output_csvfile)
            for row in csv.reader(csvfile):
                id1, id2 = row
                feature1 = feature_dict.get(id1, [])
                feature2 = feature_dict.get(id2, [])
                combined_feature = feature1 + feature2
                csv_writer.writerow(combined_feature)
if __name__ == "__main__":
    feature_file_name = "output/F.csv"
    feature_dict = read_feature_file(feature_file_name)
    input_file_name = "PositiveAndNegativeSamples.csv"
    output_file_name = "output/SampleFeature(F).csv"
    replace_id_with_feature(input_file_name, feature_dict, output_file_name)