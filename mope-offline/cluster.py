import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import re
from sklearn.preprocessing import StandardScaler
import collections
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
import random
import pickle
import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from finetune import load_expert_model,CHAR_BAG

PASSWORD_END = '\n'
PASSWORD_START = '\t'
min_len=6
max_len=16


def load_passwords(file_path, char_bag, max_len=16, min_len=6, limit=None):
    filterer = Filterer(char_bag, max_len, min_len)
    with open(file_path, "r", encoding="utf-8") as f:
        passwords = [line.strip() for line in f.readlines()]
    passwords = filterer.filter_passwords(passwords)
    return passwords[:limit] if limit else passwords


class Filterer:
    def __init__(self, char_bag, max_len, min_len, uniquify=False, sequence_model=None):
        self.filtered_out = 0
        self.total = 0
        self.total_characters = 0
        self.frequencies = collections.defaultdict(int)
        self.beg_frequencies = collections.defaultdict(int)
        self.end_frequencies = collections.defaultdict(int)
        self.char_bag = char_bag
        self.max_len = max_len
        self.min_len = min_len
        self.uniquify = uniquify
        self.seen = set()
        self.longest_pwd=None

        if sequence_model == "MANY_TO_MANY":
            self.char_bag = self.char_bag.replace("\t", "")

    @staticmethod
    def _update_frequencies(adict, pwd):
        for c in pwd:
            adict[c] += 1

    def is_valid(self, pwd, quick=False):
        if isinstance(pwd, tuple):
            pwd = ''.join(pwd)
        pwd = pwd.strip(PASSWORD_END)

        if not (all(c in self.char_bag for c in pwd) and self.min_len <= len(pwd) <= self.max_len):
            self.filtered_out += 1
            return False

        if self.uniquify:
            if pwd in self.seen:
                return False
            self.seen.add(pwd)
        if quick:
            return True
        self.total_characters += len(pwd)
        self._update_frequencies(self.frequencies, pwd)
        self._update_frequencies(self.beg_frequencies, pwd[0])
        self._update_frequencies(self.end_frequencies, pwd[-1])
        if self.longest_pwd==None:
            self.longest_pwd = len(pwd)
        else:
            self.longest_pwd = max(self.longest_pwd, len(pwd))
        self.total += 1

        return True

    def filter_passwords(self, passwords, quick=False):
        return [pwd for pwd in passwords if self.is_valid(pwd, quick)]

    def summary(self):
        print(f'Filtered {self.filtered_out} of {self.total} passwords')
        print(f'Longest password: {self.longest_pwd}')

def load_passwords(file_path,char_bag="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `\n\t",limit=None,max_len=16,min_len=6):
    passwords = []
    filterer = Filterer(char_bag, max_len, min_len)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            pwd = line.strip()
            if filterer.is_valid(pwd):
                passwords.append(pwd)
        filterer.summary()
    return passwords[:limit] if limit else passwords

def calculate_entropy(pwd):
    if not pwd:
        return 0
    freq = np.array([pwd.count(c) for c in set(pwd)])
    prob = freq / len(pwd)
    return -np.sum(prob * np.log2(prob))


def longest_repeating_sequence(pwd):
    max_len = curr_len = 0
    prev_char = None
    for char in pwd:
        if char == prev_char:
            curr_len += 1
            max_len = max(max_len, curr_len)
        else:
            curr_len = 1
        prev_char = char
    return max_len

def max_consecutive_length(pwd, check_func):
    max_len = curr_len = 0
    for c in pwd:
        if check_func(c):
            curr_len += 1
            max_len = max(max_len, curr_len)
        else:
            curr_len = 0
    return max_len

def count_uppercase(pwd):
    return sum(c.isalpha() and c.isupper() for c in pwd)

def extract_features(passwords):
    features = []

    for pwd in passwords:
        length = len(pwd) if pwd else 1
        num_count = sum(c.isdigit() for c in pwd)
        alpha_count = sum(c.isalpha() for c in pwd)
        special_count = sum(not c.isalnum() for c in pwd)
        entropy = calculate_entropy(pwd)

        switch_count = sum(
            pwd[i].isalpha() != pwd[i + 1].isalpha() for i in range(len(pwd) - 1)
        )
        feat = [
            length,
            num_count / length,
            alpha_count / length,
            special_count / length,
            switch_count,
            max_consecutive_length(pwd, str.isdigit),
            max_consecutive_length(pwd, str.isalpha),
            count_uppercase(pwd) / length,
        ]

        features.append(feat)
    return np.array(features)



def lstm_feature_extractor(passwords=[], model_name="LSTM01M_csdn_256_1"):
    char2idx = {c: i + 1 for i, c in enumerate(CHAR_BAG)}
    char2idx['<PAD>'] = 0
    idx2char = {i: c for c, i in char2idx.items()}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_len = 16
    num_layer = int(model_name.split('_')[-1])
    hidden_size = int(model_name.split('_')[-2])

    lstm_model, _ = load_expert_model(f"31_weights/{model_name}.pth", model_type="LSTM",
                                    vocab_size=len(CHAR_BAG) + 1, embedding_size=64,
                                    hidden_size=hidden_size, num_layers=num_layer, dropout_ratio=0,
                                    sequence_model="MANY_TO_ONE",device=device)

    feature_vectors=[]

    for input_str in tqdm.tqdm(passwords):
        input_tensor = torch.zeros((1, max_len), dtype=torch.long)
        for i, ch in enumerate(input_str[:max_len]):
            input_tensor[0, i] = char2idx[ch]
        input_tensor = input_tensor.to(device)  # Ensure it's on the correct device
        hidden_states=lstm_model.get_hidden(input_tensor)
        feature_vector = hidden_states[0, -1, :]  # Shape: [256]
        feature_vectors.append(feature_vector.cpu().detach().numpy())


    feature_vectors = np.array(feature_vectors)  # Shape: [n_samples, 256]
    return feature_vectors


def perform_clustering(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels


def visualize_clusters_with_tsne(features, labels, title="Clustering Visualization with t-SNE", output_file="cluster_visualization.png"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
    reduced_features = tsne.fit_transform(features)  # 降维

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="viridis", s=10, alpha=0.7)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.savefig(output_file)
    plt.show()


def visualize_clusters_with_tsne_3d(features, labels, title="3D t-SNE Clustering Visualization", output_file="tsne_3d_visualization.png"):
    tsne = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=500)
    reduced_features = tsne.fit_transform(features)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        reduced_features[:, 0], 
        reduced_features[:, 1], 
        reduced_features[:, 2], 
        c=labels, 
        cmap="tab20", 
        s=20, 
        alpha=0.8
    )
    
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    
    fig.colorbar(scatter, ax=ax, label='Cluster Label')
    plt.savefig(output_file)
    plt.show()

def save_clusters_to_files(passwords, labels, output_dir="dbscan"):
    os.makedirs(output_dir, exist_ok=True)
    unique_labels = set(labels)
    for label in unique_labels:
        cluster_passwords = [pwd for pwd, lbl in zip(passwords, labels) if lbl == label]
        file_name = f"cluster_{label}.txt" if label != -1 else "noise.txt"  # -1 为噪声点
        output_path = os.path.join(output_dir, file_name)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(cluster_passwords))
        
        print(f"Saved {len(cluster_passwords)} passwords to {output_path}")

def save_cluster_info(features, labels, output_file="cluster_info.npy"):
    n_clusters = len(np.unique(labels))
    cluster_centers = np.array([features[labels == i].mean(axis=0) for i in range(n_clusters)])
    np.save(output_file, {"cluster_centers": cluster_centers, "labels": labels})
    print(f"Cluster information saved to {output_file}")

def calculate_cluster_probabilities(prefix_vector, cluster_info_file="cluster_info.npy"):
    cluster_info = np.load(cluster_info_file, allow_pickle=True).item()
    cluster_centers = cluster_info["cluster_centers"]

    distances = np.linalg.norm(cluster_centers - prefix_vector, axis=1)
    probabilities = np.exp(-distances)
    probabilities /= probabilities.sum() 

    return probabilities


def getPrefixP(prefix,scaler,cluster_path="data/30_01M/cluster_info.npy"):
    prefix_vec=extract_features([prefix])
    prefix_vec=scaler.transform(prefix_vec)
    return calculate_cluster_probabilities(prefix_vec, cluster_info_file=cluster_path)

def getMax(array):
    return max(array),np.argmax(array)


def make_cluster_data(data_path="../../data",sample="01M", cluster_start=15, n_cluster=20, islstm=False, model_name=""):
    char_bag = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `\n\t"
    dataset_list = [
            "csdn"
        ]
    for data_name in dataset_list:
        file_path = f"{data_path}/{data_name}/1M.txt"

        bits=1
        for i in range(len(sample)):
            if sample[i]=="0":
                bits=bits*10
            else:
                break
        
        sample_num=int(1000000/bits - 5)
        passwords = load_passwords(file_path, char_bag, max_len=max_len, min_len=min_len)
        
        if bits > 1:
            passwords=random.sample(passwords,sample_num)

        if islstm==True:
            data_name=data_name+"_lstm"

        os.makedirs(f'data/{sample}/{data_name}', exist_ok=True)

        save_path=f"data/{sample}/{data_name}/{sample}.txt"
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(passwords))
        scaler = StandardScaler()

        if islstm:
            manual_features=lstm_feature_extractor(passwords)
        else:
            manual_features = extract_features(passwords)
        manual_features = scaler.fit_transform(manual_features)

        with open(f'data/{sample}/{data_name}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        for i in range(n_cluster):
            n_clusters = cluster_start + i*5 

            labels_kmeans = perform_clustering(manual_features, n_clusters=n_clusters)
            output_dir=f"data/{sample}/{data_name}/origin_feature/{n_clusters}_{sample}_scaler"

            save_clusters_to_files(passwords, labels_kmeans, output_dir=output_dir)
            save_cluster_info(manual_features, labels_kmeans, output_file=output_dir+"/cluster_info.npy")

            visualize_clusters_with_tsne(manual_features, labels_kmeans, title="KMeans Clustering Visualization with t-SNE", output_file=output_dir+"/kmeans_clusters.png")
            visualize_clusters_with_tsne_3d(manual_features, labels_kmeans, title="KMeans Clustering Visualization with t-SNE",output_file=output_dir+"/3d.png")

            silhouette_kmeans = silhouette_score(manual_features, labels_kmeans)
            print(f"{n_clusters} KMeans silhouette score: {silhouette_kmeans:.4f}")


if __name__ == "__main__":
    make_cluster_data(sample="01M",islstm=True)