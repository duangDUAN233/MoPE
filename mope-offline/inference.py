import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import bisect
import json
import math
import tqdm

from MoPE import load_passwords, Guesser
from finetune import load_expert_model, CHAR_BAG, PASSWORD_END, PASSWORD_START
from finetune import getPrefixP
from sklearn.preprocessing import StandardScaler


def moe_next_char_distribution(prefix, expert_guesser, scaler, cluster_info_path):
    cluster_weights = getPrefixP(prefix, scaler, cluster_info_path)

    combined_distribution = {}
    total_weight = 0.0
    for (cluster_idx, guesser) in expert_guesser:
        weight = cluster_weights[cluster_idx]
        total_weight += weight
        expert_dist = guesser.predict_char_probabilities(prefix)
        for char, p in expert_dist.items():
            combined_distribution[char] = combined_distribution.get(char, 0.0) + weight * p

    if total_weight > 0:
        for char in combined_distribution:
            combined_distribution[char] /= total_weight

    total = sum(combined_distribution.values())
    if total > 0:
        for char in combined_distribution:
            combined_distribution[char] /= total
    return combined_distribution


def compute_password_probability_moe(pwd, expert_guesser, scaler, cluster_info_path):
    prefix = PASSWORD_START
    log_prob = 0.0
    for char in pwd:
        distribution = moe_next_char_distribution(prefix, expert_guesser, scaler, cluster_info_path)
        p_char = distribution.get(char, 1e-12)

        try:
            log_prob += math.log(p_char)
        except ValueError as e:
            print(f"Error calculating log probability for character '{char}': {e}")
            print(f"p_char value: {p_char}")
            print(f"Prefix so far: {prefix}")
            log_prob += math.log(1e-12)
            prefix += char
            continue

        prefix += char
    return math.exp(log_prob)



def infer_test(test_name="youku", n_cluster=25, finetune_epoch=1,
         sample_n=100, train_name="csdn", sample="01M", threshold_factor=20,
         device="cuda" if torch.cuda.is_available() else "cpu"):
    
    model_name=f"LSTM01M_{train_name}_256_1"
    parts = model_name.split('_')
    hidden_size = int(parts[-2])
    num_layer = int(parts[-1])
    fla_model, _ = load_expert_model(
        f"weights/{model_name}.pth", model_type="LSTM",
        vocab_size=len(CHAR_BAG) + 1, embedding_size=64,
        hidden_size=hidden_size, num_layers=num_layer, dropout_ratio=0,
        sequence_model="MANY_TO_ONE", device=device
    )
    fla_guesser = Guesser(fla_model, device=device)
    
    scaler = StandardScaler()
    with open(f'data/{sample}/{train_name}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    cluster_dir = f"data/{sample}/{train_name}/origin_feature/{n_cluster}_{sample}_scaler"
    cluster_info_path = f"{cluster_dir}/cluster_info.npy"
    prob = getPrefixP("", scaler, cluster_info_path)
    
    threshold = 1 / n_cluster / threshold_factor
    good_cluster = [(idx, p) for idx, p in enumerate(prob) if p >= threshold]
    total_prob = sum(p for idx, p in good_cluster)
    good_cluster = [(idx, p / total_prob) for idx, p in good_cluster]
    good_cluster = sorted(good_cluster, key=lambda x: x[1], reverse=True)
    
    expert_guesser = []
    basic_dir = f"{cluster_info_path.rsplit('/', 1)[0]}/finetune"
    model_name_full = f"finetuned{finetune_epoch}_{model_name}_cluster{n_cluster}"
    for t in good_cluster:
        expert_model, _ = load_expert_model(
            f"{basic_dir}/{model_name_full}_{t[0]}.pth", model_type="LSTM",
            vocab_size=len(CHAR_BAG) + 1, embedding_size=64,
            hidden_size=hidden_size, num_layers=num_layer, dropout_ratio=0,
            sequence_model="MANY_TO_ONE", device=device
        )
        guesser_expert = Guesser(expert_model, device=device)
        expert_guesser.append((t[0], guesser_expert))
    
    # test
    print(len(expert_guesser))

    test_passwords = load_passwords(f"../../data/{test_name}/test.txt")
    test_passwords = random.sample(test_passwords, sample_n)
    
    count_mope_greater = 0
    for pwd in tqdm.tqdm(test_passwords):
        if isValid(pwd):
            fla_prob = fla_guesser.compute_password_probability(pwd)
            mope_prob = compute_password_probability_moe(pwd+PASSWORD_END, expert_guesser, scaler, cluster_info_path)
            if mope_prob > fla_prob:
                count_mope_greater += 1
    
    total_valid = sum(1 for pwd in test_passwords if isValid(pwd))
    print(f"Total valid passwords: {total_valid}")
    print(f"Number of times MoPE probability > FLA probability: {count_mope_greater}")
    print(f"Percentage: {count_mope_greater / total_valid * 100:.2f}%")


if __name__ == "__main__":
    infer_test()