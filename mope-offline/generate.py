import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from finetune import getPrefixP
import math
import tqdm

PASSWORD_START='\t'
PASSWORD_END='\n'
CHAR_BAG = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `\n\t"
)


class Guesser:
    def __init__(self,model, max_len=16, min_len=6, device="cpu",lower_probability_threshold=1e-6, outfile="generated.txt", save_interval=1000):
        self.model=model
        self.max_len=max_len
        self.min_len=min_len
        self.device=device
        self.char2idx = {c: i + 1 for i, c in enumerate(CHAR_BAG)}
        self.char2idx['<PAD>']=0
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.lower_probability_threshold=lower_probability_threshold
        self.outfile=outfile
        self.save_interval=save_interval + 1

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


    def start_guess(self, outfile):
        self.outfile=outfile
        initial_node=PASSWORD_START
        initial_prob = 1.0
        self.generated_passwords = []
        self.dfscnt=0
        self.dfs(initial_node, initial_prob)
        
        if len(self.generated_passwords)>0:
            self.generated_passwords = sorted(self.generated_passwords, key=lambda x: x[1], reverse=True)
            print(self.generated_passwords)
            with open(self.outfile, 'a') as f:
                f.write(''.join([f"{prefix}" for prefix, prob in self.generated_passwords]))

    def dfs(self, current_prefix, current_prob):
        if len(current_prefix)-1>=self.max_len or current_prefix.endswith(PASSWORD_END):
            if len(self.generated_passwords)>0 and len(self.generated_passwords)%1000==0:
                print(f"{len(self.generated_passwords)} passwords generated")
            if len(current_prefix)-1>= self.min_len:
                self.generated_passwords.append((current_prefix[1:], current_prob))

                if len(self.generated_passwords)%self.save_interval==0:
                    self.generated_passwords = sorted(self.generated_passwords, key=lambda x: x[1], reverse=True)
                    with open(self.outfile,'a') as f:
                        f.write(''.join([f"{prefix}" for prefix, prob in self.generated_passwords]))
                    self.generated_passwords=[]
            return
        char_probabilities = self.predict_char_probabilities(current_prefix)
        for char, prob in sorted(char_probabilities.items(), key=lambda x: x[1], reverse=True):
            if prob * current_prob < self.lower_probability_threshold:
                continue
            next_prefix = current_prefix + char
            next_prob = current_prob * prob 
            self.dfs(next_prefix, next_prob)
    
    def sample_one_password(self, min_len=6, max_len=16):
        while True:
            prefix = PASSWORD_START
            sampled_password = []
            total_log_prob = 0.0

            while True:
                next_char_probs = self.predict_char_probabilities(prefix)
                chars = list(next_char_probs.keys())
                probs = np.array([next_char_probs[c] for c in chars])
                chosen_char = np.random.choice(chars, p=probs)
                chosen_char_prob = next_char_probs[chosen_char]
                total_log_prob += math.log(chosen_char_prob + 1e-12)
                if chosen_char == PASSWORD_END or len(sampled_password) >= max_len:
                    break
                sampled_password.append(chosen_char)
                prefix += chosen_char
            pwd = "".join(sampled_password)
            pwd_prob = math.exp(total_log_prob)

            if len(pwd) < min_len:
                continue
            else:
                break
        return pwd, pwd_prob

    def sample_many_passwords(self, n=10_000_000, max_len=16):
        results = []
        for i in tqdm.tqdm(range(n)):
            pwd, p = self.sample_one_password(max_len=max_len)
            results.append((pwd, p))
        return results



class MoPEGuesser:
    def __init__(self,expert_guesser, max_len=16, min_len=6, device="cpu",lower_probability_threshold=1e-6, outfile="generated.txt", save_interval=1000, scaler=StandardScaler(), cluster_info_path=""):
        self.expert_guesser=expert_guesser
        self.max_len=max_len
        self.min_len=min_len
        self.device=device
        self.char2idx = {c: i + 1 for i, c in enumerate(CHAR_BAG)}
        self.char2idx['<PAD>']=0
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.lower_probability_threshold=lower_probability_threshold
        self.outfile=outfile
        self.save_interval=save_interval + 1
        self.cluster_info_path=cluster_info_path
        self.scaler=scaler
    

    def moe_next_char_distribution(self,prefix):
        cluster_weights = getPrefixP(prefix, self.scaler, self.cluster_info_path)

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
    
    def MoPE_sample_one_password(self, min_len=6, max_len=16):
        while True:
            prefix = PASSWORD_START
            sampled_password = []
            total_log_prob = 0.0
            
            while True:
                distribution = self.moe_next_char_distribution(prefix)
                chars = list(distribution.keys())
                probs = np.array([distribution[c] for c in chars])
                chosen_char = np.random.choice(chars, p=probs)
                chosen_char_prob = distribution[chosen_char]
                total_log_prob += math.log(chosen_char_prob + 1e-12)

                if chosen_char == PASSWORD_END or len(sampled_password) >= max_len:
                    break
                
                sampled_password.append(chosen_char)
                prefix += chosen_char
            
            pwd = "".join(sampled_password)
            pwd_prob = math.exp(total_log_prob)
        
            if len(pwd) < min_len:
                continue
            else:
                break
        return pwd, pwd_prob

    def predict_char_probabilities(self,prefix):
        cluster_weights = getPrefixP(prefix, self.scaler, self.cluster_info_path)

        combined_distribution = {}
        total_weight = 0.0
        for (cluster_idx, guesser) in self.expert_guesser:
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


    def start_guess(self, outfile):
        self.outfile=outfile
        initial_node=PASSWORD_START
        initial_prob = 1.0
        self.generated_passwords = []
        self.dfscnt=0
        self.dfs(initial_node, initial_prob)
        
        if len(self.generated_passwords)>0:
            self.generated_passwords = sorted(self.generated_passwords, key=lambda x: x[1], reverse=True)
            print(self.generated_passwords)
            with open(self.outfile, 'a') as f:
                f.write(''.join([f"{prefix}" for prefix, prob in self.generated_passwords]))

    def dfs(self, current_prefix, current_prob):
        if len(current_prefix)-1>=self.max_len or current_prefix.endswith(PASSWORD_END):
            if len(self.generated_passwords)>0 and len(self.generated_passwords)%1000==0:
                print(f"{len(self.generated_passwords)} passwords generated")
            if len(current_prefix)-1>= self.min_len:
                self.generated_passwords.append((current_prefix[1:], current_prob))

                if len(self.generated_passwords)%self.save_interval==0:
                    self.generated_passwords = sorted(self.generated_passwords, key=lambda x: x[1], reverse=True)
                    with open(self.outfile,'a') as f:
                        f.write(''.join([f"{prefix}" for prefix, prob in self.generated_passwords]))
                    self.generated_passwords=[]
            return
        
        char_probabilities = self.predict_char_probabilities(current_prefix)

        for char, prob in sorted(char_probabilities.items(), key=lambda x: x[1], reverse=True):
            if prob * current_prob < self.lower_probability_threshold:
                continue

            next_prefix = current_prefix + char
            next_prob = current_prob * prob
            self.dfs(next_prefix, next_prob)
