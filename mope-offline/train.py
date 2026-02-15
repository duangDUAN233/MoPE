import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import ExpertModel
import tqdm


PASSWORD_START = "\t"
PASSWORD_END   = "\n"

CHAR_BAG = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `\n\t"
)


class SingleClusterDataset(Dataset):
    def __init__(self, pwd_file, char2idx, idx2char, min_len=4, max_len=16):
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.max_len   = max_len
        self.min_len   = min_len

        self.passwords = self._load_passwords(pwd_file)
        self.samples   = self._build_samples()

    def _load_passwords(self, pwd_file):
        passwords = []
        with open(pwd_file, "r", encoding="utf-8") as f:
            for line in f:
                pwd = line.strip()
                passwords.append(pwd)
        random.shuffle(passwords)
        print(f"[SingleClusterDataset] Loaded {len(passwords)} from {pwd_file}")
        return passwords

    def _build_samples(self):
        samples = []
        for pwd in self.passwords:
            pwd_aug = PASSWORD_START + pwd + PASSWORD_END
            for i in range(1, len(pwd_aug)):
                prefix = pwd_aug[:i]
                target = pwd_aug[i]
                samples.append((prefix, target))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prefix, target_ch = self.samples[idx]
        x_arr = np.zeros(self.max_len, dtype=int)
        for i, c in enumerate(prefix[:self.max_len]):
            x_arr[i] = self.char2idx.get(c, 0)
        y_idx = self.char2idx.get(target_ch, 0)
        return x_arr, y_idx


def collate_fn_m2o(batch):
    xs, ys = [], []
    for (x, y) in batch:
        xs.append(x)
        ys.append(y)
    xs = torch.LongTensor(xs)  
    ys = torch.LongTensor(ys)
    return xs, ys


def save_model(model, file_path, char2idx):
    save_data = {
        "model_state_dict": model.state_dict(),
        "char2idx": char2idx
    }
    torch.save(save_data, file_path)
    print(f"[save_model] saved to {file_path}")

def load_model(file_path, model_type="GRU", vocab_size=100, embedding_size=64,
               hidden_size=128, num_layers=1, dropout_ratio=0.0,
               sequence_model="MANY_TO_ONE", device="cpu"):
    data = torch.load(file_path, map_location=device)
    model = ExpertModel(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        model_type=model_type,
        num_layers=num_layers,
        dropout_ratio=dropout_ratio,
        sequence_model=sequence_model
    )
    model.load_state_dict(data["model_state_dict"])
    model.to(device)
    print(f"[load_model] model loaded from {file_path}, model_type={model_type}")
    return model, data["char2idx"]


def combine_cluster_files(input_dir, output_file):
    try:
        cluster_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith("cluster_") and f.endswith(".txt")])
        if not cluster_files:
            return
        
        combined_content = []
        for file in cluster_files:
            with open(file, "r", encoding="utf-8") as f:
                combined_content.append(f"# Content from {os.path.basename(file)}\n")
                combined_content.extend(f.readlines())
                combined_content.append("\n")

        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(combined_content)

    except Exception as e:
        print(f"error: {e}")


def train_single_expert(
    cluster_file,
    char2idx, 
    idx2char,
    model_type="GRU",
    sequence_model="MANY_TO_ONE",
    max_len=16,
    embedding_size=64,
    hidden_size=128,
    num_layers=1,
    dropout_ratio=0.0,
    epochs=3,
    batch_size=64,
    lr=1e-3
):

    dataset = SingleClusterDataset(cluster_file, char2idx, idx2char, min_len=4, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_m2o)

    model = ExpertModel(
        vocab_size=len(char2idx),
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        model_type=model_type,
        num_layers=num_layers,
        dropout_ratio=dropout_ratio,
        sequence_model=sequence_model
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    base_name = os.path.basename(cluster_file)

    for epoch in tqdm.tqdm(range(epochs)):
        model.train()
        total_loss = 0.0
        for x, y in tqdm.tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            logits = model(x)  # MANY_TO_ONE => [B, vocab_size]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"[TrainSingleExpert] {base_name}, {model_type} epoch={epoch+1}/{epochs}, loss={avg_loss:.4f}")

    return model


def predict_next_char_probabilities(model, prefix, char2idx, idx2char, max_len=16):
    device = next(model.parameters()).device
    x_arr = np.zeros(max_len, dtype=int)
    for i, ch in enumerate(prefix[:max_len]):
        x_arr[i] = char2idx.get(ch, 0)

    x_tensor = torch.tensor([x_arr], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(x_tensor)  # [1, vocab_size]
        probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    prob_dict = {}
    for i, p in enumerate(probs):
        prob_dict[idx2char[i]] = p
    return prob_dict

def test_run():
    char2idx = {c: i + 1 for i, c in enumerate(CHAR_BAG)}
    char2idx['<PAD>'] = 0
    idx2char = {i: c for c, i in char2idx.items()}

    cluster_file = "data/rockyou/combined_clusters.txt"
    model_path   = "weights/lstm01M_csdn_128_1.pth"

    if not os.path.exists(model_path):
        model = train_single_expert(
            cluster_file=cluster_file,
            char2idx=char2idx,
            idx2char=idx2char,
            model_type="LSTM",
            sequence_model="MANY_TO_ONE",
            max_len=16,
            embedding_size=64,
            hidden_size=128,
            epochs=50,
            batch_size=128,
            lr=1e-3,
            num_layers=1,
            dropout_ratio=0.3,
        )
        save_model(model, model_path, char2idx)
    else:
        model, _ = load_model(
            file_path=model_path,
            model_type="LSTM",
            vocab_size=len(char2idx),
            embedding_size=64,
            hidden_size=64,
            num_layers=1,
            dropout_ratio=0,
            sequence_model="MANY_TO_ONE",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    # test
    test_prefix = PASSWORD_START+ "zxcv"
    prob_dict = predict_next_char_probabilities(model, test_prefix, char2idx, idx2char, max_len=16)
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"[test_run] prefix='{test_prefix}', top10 next-char probabilities:")
    for ch, p in sorted_probs:
        ch_show = repr(ch)
        print(f"  {ch_show}: {p:.4f}")


def train(model_type="LSTM",hidden_size=128, num_layer=1,sample="1M", max_len=16, embedding_size=64, epochs=30, batch_size=128, lr=1e-3, dropout_ratio=0.15):
    char2idx = {c: i + 1 for i, c in enumerate(CHAR_BAG)}
    char2idx['<PAD>'] = 0
    idx2char = {i: c for c, i in char2idx.items()}

    train_list=[
        "csdn", 
    ]
    for train_name in train_list:
        cluster_file = f"data/{sample}/{train_name}/{sample}.txt"
        model_path   = f"{epochs}_weights/{model_type}{sample}_{train_name}_{hidden_size}_{num_layer}.pth"

        print(cluster_file+" loaded")
        print("=============================")

        model = train_single_expert(
                cluster_file=cluster_file,
                char2idx=char2idx,
                idx2char=idx2char,
                model_type=model_type,
                sequence_model="MANY_TO_ONE",
                max_len=max_len,
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                num_layers=num_layer,
                dropout_ratio=dropout_ratio,
            )
        save_model(model, model_path, char2idx)


if __name__ == "__main__":
    train(model_type="LSTM",hidden_size=256, num_layer=1,sample="1M", max_len=16, embedding_size=64, epochs=31, batch_size=128, lr=1e-3, dropout_ratio=0.15)