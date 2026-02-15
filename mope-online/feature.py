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

# constant
PASSWORD_END = '\n'
PASSWORD_START = '\t'
min_len=6
max_len=16


def load_passwords(file_path,char_bag="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `\n\t"):
    passwords = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            passwords.append(line)

    return passwords


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

def perform_clustering(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels


def visualize_clusters_with_tsne(features, labels, title="Clustering Visualization with t-SNE", output_file="cluster_visualization.png"):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="viridis", s=10, alpha=0.7)
    plt.colorbar(scatter, label='Cluster Label')
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
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
    # ================
    if len(prefix)>1 and prefix[0]=='\t':
        prefix=prefix[1:]
    if len(prefix)==1 and prefix[0]=='\t':
        prefix=""
    # ================
    prefix_vec=extract_features([prefix])
    prefix_vec=scaler.transform(prefix_vec)
    return calculate_cluster_probabilities(prefix_vec, cluster_info_file=cluster_path)


def getFeatureVec(prefix,scaler,cluster_path="data/30_01M/cluster_info.npy"):
    # ================
    if len(prefix)>1 and prefix[0]=='\t':
        prefix=prefix[1:]
    if len(prefix)==1 and prefix[0]=='\t':
        prefix=""
    # ================
    prefix_vec=extract_features([prefix])
    prefix_vec=scaler.transform(prefix_vec)
    return prefix_vec

def getMax(array):
    return max(array),np.argmax(array)


def make_cluster_data(data_path="../../data",sample="01M", cluster_start=25, n_cluster=10):
    char_bag = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `\n\t"

    sample="100k"
    file_path = f"../../data/2_train/Collection1_cos_{sample}.csv"
    passwords = load_passwords(file_path, char_bag)
    
    first_list=[]
    for line in passwords:
        first_list.append(line.split('\t')[0])
    
    os.makedirs(f'cluster/data/{sample}', exist_ok=True)

    scaler = StandardScaler()
    manual_features = extract_features(first_list)
    manual_features = scaler.fit_transform(manual_features)

    with open(f'cluster/data/{sample}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    for i in range(n_cluster):
        n_clusters = cluster_start + i*5

        labels_kmeans = perform_clustering(manual_features, n_clusters=n_clusters)

        output_dir=f"cluster/data/{sample}/origin_feature/{n_clusters}_{sample}_scaler"
        save_clusters_to_files(passwords, labels_kmeans, output_dir=output_dir)
        save_cluster_info(manual_features, labels_kmeans, output_file=output_dir+"/cluster_info.npy")
        silhouette_kmeans = silhouette_score(manual_features, labels_kmeans)
        print(f"{n_clusters} KMeans silhouette score: {silhouette_kmeans:.4f}")
        print("-----")


if __name__ == "__main__":
    make_cluster_data()


