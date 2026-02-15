import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from MoPE import Guesser, load_passwords
from finetune import load_expert_model, CHAR_BAG, PASSWORD_END, PASSWORD_START,save_expert_model
from finetune import getPrefixP
from sklearn.preprocessing import StandardScaler
import pickle
import tqdm
from model import ExpertModel
import random
import matplotlib.pyplot as plt


char2idx = {c: i + 1 for i, c in enumerate(CHAR_BAG)}
char2idx['<PAD>'] = 0
idx2char = {i: c for c, i in char2idx.items()}
device = "cuda" if torch.cuda.is_available() else "cpu"
max_len = 16


class PasswordDataset(Dataset):
    def __init__(self, passwords):
        self.passwords = passwords

    def __len__(self):
        return len(self.passwords)

    def __getitem__(self, idx):
        password = self.passwords[idx]
        return password

class DistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.7):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels=None):
        hard_loss = self.criterion_ce(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        soft_loss = self.criterion(torch.log(student_logits / self.temperature),
                                   teacher_logits / self.temperature) * (self.temperature ** 2)
        
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        return total_loss
    

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


def predict_char_probabilities(model, input_str):
    input_tensor = torch.zeros((1, max_len), dtype=torch.long)
    for i, ch in enumerate(input_str[:max_len]):
        input_tensor[0, i] = char2idx[ch]
    input_tensor = input_tensor.to(device)  # Ensure it's on the correct device

    with torch.no_grad():
        # [1, vocab_size] -> a batch size of 1
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()  # Convert to numpy array

    char_probabilities = {idx2char[i]: prob for i, prob in enumerate(probabilities)}
    return char_probabilities



def train_student_model_with_distillation(student_model, expert_guesser, scaler, cluster_info_path, model_name, passwords, optimizer, learning_rate, distillation_loss_fn, temperature, alpha):
    student_model.train()
    data = []
    for pwd in passwords:
        pwd = '\t' + pwd
        pwd = pwd + '\n'
        for length in range(1, len(pwd)):
            data.append([pwd[0:length], pwd[length]])
    batch_size = 128
    dataset = PasswordDataset(data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    cnt = 0
    save_cnt = 100

    for epoch in range(5):
        total_loss = 0
        for batch_idx, batch_passwords in enumerate(data_loader):
            optimizer.zero_grad()

            student_logits_batch = []
            teacher_logits_batch = []
            labels_batch = []
            for i in tqdm.tqdm(range(len(batch_passwords[0]))):
                prefix, next_char = batch_passwords[0][i], batch_passwords[1][i]
                teacher_logits = moe_next_char_distribution(prefix, expert_guesser, scaler, cluster_info_path)
                teacher_logits = torch.tensor([teacher_logits[idx2char[i]] for i in range(len(idx2char))]).unsqueeze(0).to(device)
                input_tensor = torch.zeros((1, max_len), dtype=torch.long)
                for i, ch in enumerate(prefix[:max_len]):
                    input_tensor[0, i] = char2idx[ch]
                input_tensor = input_tensor.to(device)
                student_logits = torch.softmax(student_model(input_tensor), dim=-1).unsqueeze(0).to(device)
                threshold=1/len(idx2char)/len(idx2char)
                student_logits = torch.where(student_logits < threshold, torch.tensor(1e-32, device=student_logits.device), student_logits)
                teacher_logits = torch.where(teacher_logits < threshold, torch.tensor(1e-32, device=teacher_logits.device), teacher_logits)
                target_idx = torch.tensor(char2idx[next_char]).unsqueeze(0).to(device)
                labels_batch.append(target_idx)
                student_logits_batch.append(student_logits)
                teacher_logits_batch.append(teacher_logits)

            student_logits_batch = torch.cat(student_logits_batch, dim=0)
            teacher_logits_batch = torch.cat(teacher_logits_batch, dim=0)
            labels_batch = torch.cat(labels_batch, dim=0)

            loss = distillation_loss_fn(student_logits_batch, teacher_logits_batch, labels_batch)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            cnt += 1
            if cnt % save_cnt == 0:
                if int(cnt/save_cnt)>30:
                    return
                save_expert_model(student_model, 
                                  f"distill/{temperature}_{alpha}_distilled{int(cnt/save_cnt)}_{cluster_info_path.split('/')[1]}_{model_name.split('_')[-3]}_{model_name.split('_')[-2]}_{model_name.split('_')[-1]}.pth", 
                                  char2idx=char2idx)
        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss / len(data_loader)}")


def distill_to_student_model(student_model, expert_guesser, scaler, cluster_info_path, model_name, passwords, learning_rate=1e-5, temperature=2, alpha=1):
    distillation_loss_fn = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    train_student_model_with_distillation(student_model, expert_guesser, scaler, cluster_info_path, model_name, passwords, optimizer, learning_rate, distillation_loss_fn,temperature,alpha)


def load_experts_and_train_student_model(expert_model_paths, scaler, cluster_info_path, passwords, student_model, device="cpu", model_name="", n_cluster=25, finetune_epoch=1,sample="01M"):
    parts = model_name.split('_')
    hidden_size = int(parts[-2])
    layers = int(parts[-1])
    print(f"now is cluster {n_cluster}")

    threshold = 1/n_cluster/10
    prob = getPrefixP("", scaler, cluster_info_path)

    good_cluster = []
    for idx in range(len(prob)):
        if prob[idx] >= threshold:
            good_cluster.append((idx, prob[idx]))
    total_prob = sum(p for idx, p in good_cluster)
    good_cluster = [(idx, p / total_prob) for idx, p in good_cluster]
    good_cluster = sorted(good_cluster, key=lambda x: x[1], reverse=True)

    expert_guesser = []
    basic_dir = f"{cluster_info_path.rsplit('/', 1)[0]}/finetune"
    model_name_full = f"finetuned{finetune_epoch}_{model_name}_cluster{n_cluster}"
    print("Loading experts from:", model_name_full)
    for t in good_cluster:
        expert_model, _ = load_expert_model(f"{basic_dir}/{model_name_full}_{t[0]}.pth", model_type="LSTM",
                                            vocab_size=len(CHAR_BAG) + 1, embedding_size=64,
                                            hidden_size=hidden_size, num_layers=layers, dropout_ratio=0,
                                            sequence_model="MANY_TO_ONE", device=device)
        guesser_expert = Guesser(expert_model, device=device)
        expert_guesser.append((t[0], guesser_expert))
    print("Experts loaded.")

    distill_to_student_model(student_model, expert_guesser, scaler, cluster_info_path,model_name, passwords)


def main():
    finetune_epoch = 1
    sample="01M"
    train_list = [
        "csdn"
    ]

    data2cluster={
        "csdn":50
    }

    for train_name in train_list:
        try:
            n_cluster=data2cluster[train_name]
            trainfile_path = f"./data/{sample}/{train_name}/{sample}.txt"
            model_name = f"LSTM01M_{train_name}_256_1"

            passwords = load_passwords(trainfile_path)
            num_layer = int(model_name.split('_')[-1])
            hidden_size = int(model_name.split('_')[-2])
            sample=model_name.split('_')[0][4:]
            student_model, _ = load_expert_model(f"31_weights/{model_name}.pth", model_type="LSTM",
                                            vocab_size=len(CHAR_BAG) + 1, embedding_size=64,
                                            hidden_size=hidden_size, num_layers=num_layer, dropout_ratio=0,
                                            sequence_model="MANY_TO_ONE", device=device)
            student_model = student_model.to(device)
            expert_model_paths = []

            cluster_dir = f"data/{sample}/{train_name}/origin_feature/{n_cluster}_01M_scaler"
            basic_dir = f"{cluster_dir}/finetune"
            model_name_full = f"finetuned{finetune_epoch}_{model_name}_cluster{n_cluster}"
            print("Loading experts from:", model_name_full)

            for idx in range(n_cluster):
                expert_model_paths.append(f"{basic_dir}/{model_name_full}_{idx}.pth")
            cluster_info_path = f"{cluster_dir}/cluster_info.npy"
            scaler = StandardScaler()
            with open(f'data/{sample}/{train_name}/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
            load_experts_and_train_student_model(expert_model_paths, scaler, cluster_info_path, passwords,
                                                student_model, device=device, model_name=model_name, n_cluster=n_cluster, finetune_epoch=finetune_epoch)
            print("Student model distillation completed. Saved to 'distilled_student_model.pth'")
        except:
            continue


if __name__ == "__main__":
    main()
