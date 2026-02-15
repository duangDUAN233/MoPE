import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import bisect
from model import ExpertModel
import math
import tqdm


PASSWORD_START='\t'
PASSWORD_END='\n'
CHAR_BAG = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `\n\t"
)


class Guesser:
    def __init__(self,model, max_len=16, min_len=6, device="cpu",lower_probability_threshold=1e-6, outfile="generated.txt"):
        self.model=model
        self.max_len=max_len
        self.min_len=min_len
        self.device=device
        self.char2idx = {c: i + 1 for i, c in enumerate(CHAR_BAG)}
        self.char2idx['<PAD>']=0
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.lower_probability_threshold=lower_probability_threshold
        self.outfile=outfile

    def predict_char_probabilities(self,input_str):
        input_tensor = torch.zeros((1, self.max_len), dtype=torch.long)
        for i, ch in enumerate(input_str[:self.max_len]):
            input_tensor[0, i] = self.char2idx[ch]

        input_tensor=input_tensor.to(self.device)

        with torch.no_grad():
            # [1, vocab_size]
            logits = self.model(input_tensor)
            # probabilities = torch.softmax(logits, dim=-1)[0].numpy()
            if self.device == "cuda":
                probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            else:
                probabilities = torch.softmax(logits, dim=-1)[0].numpy()

        char_probabilities = {self.idx2char[i]: prob for i, prob in enumerate(probabilities)}
        return char_probabilities


    def compute_probability_range(self, pwd_str, i, j, max_len=16):
        if i >= len(pwd_str) or j >= len(pwd_str):
            raise ValueError(f"Indices i={i} and j={j} are out of range for the password of length {len(pwd_str)}.")
        j = min(j, len(pwd_str) - 1)
        if i>0:
            prefix = pwd_str[:i]
        else:
            prefix = '\t'
        total_log_prob = 0.0
        
        for idx in range(i, j+1):
            next_char_probs = self.predict_char_probabilities(prefix)
            p_ch = next_char_probs.get(pwd_str[idx], 0.0)          
            if p_ch <= 0:
                return 0.0
            
            total_log_prob += np.log(p_ch + 1e-90)
            prefix += pwd_str[idx]

        return np.exp(total_log_prob)


    def compute_password_probability(self, pwd_str, max_len=16):
        prefix = PASSWORD_START
        pwd_str+=PASSWORD_END
        total_log_prob = 0.0
        for ch in pwd_str:
            next_char_probs = self.predict_char_probabilities(prefix)
            p_ch = next_char_probs.get(ch, 0.0)
            if p_ch <= 0:
                return 0.0
            total_log_prob += np.log(p_ch + 1e-45)
            prefix += ch
            if len(prefix) >= max_len:
                break
        return np.exp(total_log_prob)
    

def load_expert_model(file_path, model_type="GRU", vocab_size=100, embedding_size=64,
                      hidden_size=128, num_layers=1, dropout_ratio=0.0,
                      sequence_model="MANY_TO_ONE", device="cpu"):
    loaded_data = torch.load(file_path, map_location=device)

    model = ExpertModel(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        model_type=model_type,
        num_layers=num_layers,
        dropout_ratio=dropout_ratio,
        sequence_model=sequence_model
    )
    model.load_state_dict(loaded_data["model_state_dict"])
    model.to(device)
    print(f"[load_expert_model] loaded from {file_path}")
    return model, loaded_data["char2idx"]


def save_arrays(A, C, file_path="arrays.pkl"):
    data = {"A": A, "C": C}
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Arrays A and C saved to {file_path}")

def load_arrays(file_path="arrays.pkl"):
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None, None
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    print(f"Arrays A and C loaded from {file_path}")
    return data["A"], data["C"]


def sample_one_password(experts_list, experts_weights, max_len=16):
    prefix = PASSWORD_START
    sampled_password = []
    total_log_prob = 0.0

    chosen_expert_index = np.random.choice(len(experts_list), p=experts_weights)
    chosen_expert = experts_list[chosen_expert_index]

    prefix = PASSWORD_START
    sampled_password = []
    total_log_prob = 0.0
    total_log_prob+=math.log(experts_weights[chosen_expert_index] + 1e-45)

    while True:
        next_char_probs = chosen_expert.predict_char_probabilities(prefix)
        chars = list(next_char_probs.keys())
        probs = np.array([next_char_probs[c] for c in chars])
        chosen_char = np.random.choice(chars, p=probs)
        chosen_char_prob = next_char_probs[chosen_char]
        total_log_prob += math.log(chosen_char_prob + 1e-45)
        if chosen_char == PASSWORD_END or len(sampled_password) >= max_len:
            break
        sampled_password.append(chosen_char)
        prefix += chosen_char
    
    pwd="".join(sampled_password)
    other_prob=0
    pwd_prob = math.exp(total_log_prob)

    for i in range(len(experts_list)):
        if i != chosen_expert_index:
            other_prob+=experts_weights[i]*experts_list[i].compute_password_probability(pwd)

    pwd_prob+=other_prob

    return pwd, pwd_prob


def sample_many_passwords(expert_guesser, normalized_weight,n=10_000_000, max_len=16):
    results = []
    for i in tqdm.tqdm(range(n)):
        pwd, p = sample_one_password(expert_guesser, normalized_weight, max_len)
        results.append((pwd, p))
    return results

def generate_and_save_arrays(res, save_path="arrays.pkl"):
    res.sort(reverse=True, key=lambda x: x[1])
    n=len(res)
    A = [prob for _, prob in res]
    C = np.zeros(len(A))
    running_sum = 0.0
    for i, prob in enumerate(A):
        running_sum += 1.0 / (n * prob)
        C[i] = running_sum

    save_arrays(A, C, save_path)
    return A, C

def find_rank(expert_list, normalized_weight, pwd, A, C):
    p=0.0
    for i in range(len(expert_list)):
        p+=normalized_weight[i]*expert_list[i].compute_password_probability(pwd)
    idx = bisect.bisect_left(A[::-1], p)
    if idx < len(C):
        return C[-(idx + 1)]
    else:
        if p >= A[0]:
            return 1
        print(f"{pwd} is stronger than all sampled passwords.")
        return None

def load_passwords(file_path):
    passwds=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            passwds.append(line.strip())
    return passwds

def calculate_cracking_rate(test_passwords, guess_counts):
    x = []
    y = []
    total_passwords = len(test_passwords)

    hits = 0
    for idx, count in enumerate(guess_counts):
        hits += 1
        x.append(count)
        y.append(hits / total_passwords * 100)

    return x, y

def save_cracking_rate(x, y, output_file="cracking_rate_data.txt"):
    with open(output_file, "w") as f:
        for i in range(len(x)):
            f.write(f"{x[i]}\t{y[i]}\n")
    print(f"Cracking rate data saved to '{output_file}'")
