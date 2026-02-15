import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import ExpertModel
from train import SingleClusterDataset,collate_fn_m2o,PASSWORD_END,PASSWORD_START,CHAR_BAG
from cluster import extract_features

def save_expert_model(model, file_path, char2idx):
    data = {
        "model_state_dict": model.state_dict(),
        "char2idx": char2idx
    }
    torch.save(data, file_path)
    print(f"[save_expert_model] saved to {file_path}")

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
    return model, loaded_data["char2idx"]

def fine_tune_expert_on_cluster(base_model_path, cluster_file,
                                char2idx, idx2char,
                                model_type="GRU", sequence_model="MANY_TO_ONE",
                                max_len=16, embedding_size=64, hidden_size=128,
                                num_layers=1, dropout_ratio=0.0,
                                epochs=3, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_data = torch.load(base_model_path, map_location=device)
    model_state = base_data["model_state_dict"]
    model = ExpertModel(
        vocab_size=len(char2idx),
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        model_type=model_type,
        num_layers=num_layers,
        dropout_ratio=dropout_ratio,
        sequence_model=sequence_model
    ).to(device)
    model.load_state_dict(model_state)

    dataset = SingleClusterDataset(cluster_file, char2idx, idx2char, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_m2o)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    base_name = os.path.basename(cluster_file)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x,y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"[fine_tune_expert_on_cluster] {base_name} epoch={epoch+1}/{epochs}, loss={avg_loss:.4f}")

    return model



def calculate_cluster_probabilities(prefix_vector, cluster_info_file="cluster_info.npy"):
    cluster_info = np.load(cluster_info_file, allow_pickle=True).item()
    cluster_centers = cluster_info["cluster_centers"]

    distances = np.linalg.norm(cluster_centers - prefix_vector, axis=1)

    probabilities = np.exp(-distances)
    probabilities /= probabilities.sum()
    return probabilities


def getPrefixP(prefix,scaler,cluster_path=""):
    if len(prefix)>1 and prefix[0]=='\t':
        prefix=prefix[1:]
    if len(prefix)==1 and prefix[0]=='\t':
        prefix=""
    prefix_vec=extract_features([prefix])
    prefix_vec=scaler.transform(prefix_vec)
    return calculate_cluster_probabilities(prefix_vec, cluster_info_file=cluster_path)

def getMax(array):
    return max(array),np.argmax(array)


def finetune(cluster_dir, model_name="", finetune_epoch=1, n_cluster=25):
    char2idx = {'<PAD>':0}
    for i,ch in enumerate(CHAR_BAG, start=1):
        char2idx[ch] = i
    idx2char = {v:k for k,v in char2idx.items()}

    model_type="LSTM"
    if not os.path.exists(f"{cluster_dir}/finetune"):
        os.mkdir(f"{cluster_dir}/finetune")

    base_model_path=f"weights/{model_name}.pth"

    parts = model_name.split('_')
    hidden_size = int(parts[-2])
    layers = int(parts[-1])

    for i in range(n_cluster):
        cluster_file = os.path.join(cluster_dir, f"cluster_{i}.txt")
        if not os.path.exists(cluster_file):
            continue
        expert_model = fine_tune_expert_on_cluster(
            base_model_path=base_model_path,
            cluster_file=cluster_file,
            char2idx=char2idx,
            idx2char=idx2char,
            model_type=model_type,
            sequence_model="MANY_TO_ONE",
            max_len=16,
            embedding_size=64,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout_ratio=0,
            epochs=finetune_epoch,
            batch_size=128,
            lr=1e-3
        )
        save_path = f"{cluster_dir}/finetune/finetuned{finetune_epoch}_{model_name}_cluster{n_cluster}_{i}.pth"
        save_expert_model(expert_model, save_path, char2idx)
        print(f"[main] cluster {i} finetuning done => {save_path}")


def make_finetune(hidden_size=256, num_layer=2, finetune_start=1, finetune_end=3, cluster_start=25, cluster_end=30, model_type="LSTM", sample="01M"):
    train_list=[
        "csdn", 
    ]
    for train_from in train_list:
        data_dir=f"data/{sample}/{train_from}"

        for n_cluster in range(cluster_start,cluster_end + 1,5):
            for finetune_epoch in range(finetune_start,finetune_end):
                cluster_dir = f"{data_dir}/origin_feature/{n_cluster}_{sample}_scaler"
                model_name=f"{model_type}{sample}_{train_from}_{hidden_size}_{num_layer}"

                print(f"now is cluster {n_cluster}")
                print("===============================")
                finetune(cluster_dir, model_name, finetune_epoch=finetune_epoch, n_cluster=n_cluster)


