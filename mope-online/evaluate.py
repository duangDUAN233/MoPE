from networkx import jaccard_coefficient
from dataloader import *
from model import *
from tokenizer import *
import numpy as np
import itertools
import tqdm
from feature import *
from collections import defaultdict
from cossim import *
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Evaluator:
    def __init__(self, 
                 model_path, 
                 t1 = KBDPasswordTokenizer(), 
                 t2 = TransTokenizer(), 
                 batch_size=32, 
                 max_len=16, 
                 device=DEFAULT_DEVICE):
        self.t1 = t1 
        self.t2 = t2
        self.max_len = max_len
        self.model = torch.load(model_path).to(device)
        self.model.device = device
        self.device = device
        self.batch_size = batch_size
        self.model.set_mode("predict")
        self.model.eval()
    
    def predict(self, pwds):
        raise NotImplementedError()

class GreedyEvaluator(Evaluator):
    def __init__(self, 
                 model_path, 
                 t1=KBDPasswordTokenizer(), 
                 t2=TransTokenizer(), 
                 batch_size=32, 
                 max_len=16, 
                 device=DEFAULT_DEVICE):
        super().__init__(model_path, t1, t2, batch_size, max_len, device)
        self.model.set_mode("predict")

    def _predict(self, pwds):
        n = len(pwds)
        pwd_ids = self.t1(pwds, padding=False)
        pwd_ids = self.t1.padding(pwd_ids).to(self.device)
        digits = self.model(pwd_ids)

        probs = [sum(x[1] for x in d) for d in digits]
        edits = [[x[0] for x in d] for d in digits]
        return zip(pwds,[self.t2.decode(pwds[i], edits[i]) for i in range(n)], probs)
    

    def predict(self, pwds):
        n = len(pwds)
        combined = iter([])
        for i in range(0, n, self.batch_size):
            batch_x = pwds[i:i + self.batch_size]
            combined = itertools.chain.from_iterable([combined, self._predict(batch_x)])
        return combined

class BeamSearchEvaluator(Evaluator):
    def __init__(self, 
                 model_path, 
                 t1=KBDPasswordTokenizer(), 
                 t2=TransTokenizer(), 
                 batch_size=32, 
                 max_len=16, 
                 device=DEFAULT_DEVICE):
        super().__init__(model_path, t1, t2, batch_size, max_len, device)
        self.model.set_mode("beamsearch")


    def _predict_test(self, pwds,beamwidth=10, topk=2):
        n = len(pwds)
        pwd_ids = self.t1(pwds, padding=False)
        pwd_ids = self.t1.padding(pwd_ids).to(self.device)
        
        digits = self.model(pwd_ids,beamwidth=beamwidth, topk=topk)
        ans = []
        for i in range(n):
            items = []
            outputs = digits[i]
            for output in outputs:
                prob = output[1]
                edits = output[0]
                pwd = edits
                items.append((pwd, prob))
            ans.append(items)
        return zip(pwds,ans)
    

    def _predict(self, pwds,beamwidth=10, topk=2):
        n = len(pwds)
        pwd_ids = self.t1(pwds, padding=False)
        pwd_ids = self.t1.padding(pwd_ids).to(self.device)
        
        digits = self.model(pwd_ids,beamwidth=beamwidth, topk=topk)
        ans = []
        used_pwds = set()
        for i in range(n):
            items = []
            outputs = digits[i]
            for output in outputs:
                if len(items) >topk:
                    break
                prob = output[1]
                edits = output[0]
                pwd = self.t2.decode(pwds[i], edits)
                if pwd not in used_pwds:
                    items.append((pwd, prob))
                    used_pwds.add(pwd)
            ans.append(items)
        return zip(pwds,ans)
    
    def predict(self, pwds, beamwidth=10, topk=2):
        n = len(pwds)
        combined = iter([])
        for i in range(0, n, self.batch_size):
            batch_x = pwds[i:i + self.batch_size]
            combined = itertools.chain.from_iterable([combined, self._predict(batch_x, beamwidth=beamwidth, topk=topk)])
        return combined
    
    def predict_one(self, pwd, beamwidth=10, topk=2):
        return next(self._predict([pwd], beamwidth=beamwidth, topk=topk))

class MoPEEvaluator:
    def __init__(self,model,outputs,scaler,cluster_info_path,n_cluster,threshold=8):
        self.model = model
        self.csv_out = open(outputs, "w")
        self.scaler=scaler
        self.cluster_info_path=cluster_info_path
        self.n_cluster=n_cluster
        self.threshold = 1/n_cluster/threshold
        
    def count_lines(self, file_path):
        line_count = 0
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    line_count += 1
        except FileNotFoundError:
            print(file_path)
        except IOError:
            print(file_path)
        return line_count
    
    def parse_batch(self, inputs):
        src = [x[0] for x in inputs]
        tar = [x[1] for x in inputs]
        return src, tar

    def evaluate(self, path, beamwidth=150, topk=1000, batch_size=64, chosen_num=20):
        hit_count = 0
        pwds = []
        batch_size = batch_size
        with open(path, "r") as f:
            for line in f:
                line = line.strip("\r\n").split("\t")
                src = line[0]
                target = line[1]
                pwds.append((src, target))
                
        for i in tqdm.tqdm(range(0,len(pwds))):
            inputs = pwds[i]
            src, tar = self.parse_batch([inputs])
            prob = getPrefixP(src[0], self.scaler, self.cluster_info_path)
            weights=prob

            for j in range(len(weights)-chosen_num):
                weights[np.argmin(weights)]+=1
            
            for j in range(len(weights)):
                if weights[j]>=1:
                    weights[j]=0
            weights/=np.sum(weights)

            outputs=[]
            for j in range(len(weights)):
                if weights[j]==0:
                    outputs.append([])
                    continue
                else:
                    outputs.append(list(self.model[j].predict(src, beamwidth, topk)))

            prob_dict = defaultdict(float)
            for j in range(len((outputs))):
                if len(outputs[j])==0:
                    continue
                else:
                    for _,model_output in outputs[j]:
                        for key, probability in model_output:
                            probability=np.exp(-probability)*weights[j]
                            prob_dict[key] += probability

            # descending order
            sorted_items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)

            # top 1000
            top_1000 = sorted_items[:1000]

            # print(top_1000)
            for j in range(len(top_1000)):
                hit=False
                pwd,p=top_1000[j][0],top_1000[j][1]
                # print(pwd,tar[0])
                p=-np.log(p)
                if pwd == tar[0]:
                    hit = True
                    hit_count += 1
                    self.csv_out.write(f"{src[0]}\t{tar[0]}\t{j+1}\t{p}\n")
                    break
            if not hit:
                self.csv_out.write(f"{src[0]}\t{tar[0]}\t{-1}\t{0.0}\n")
        print(f">>> Guess Rate: {hit_count / len(pwds)}")

    def finish(self):
        self.csv_out.close()


def main():
    hidden_size=256
    num_layers=3
    n_cluster=65    
    finetune_epoch=10
    chosen_num=10
    sample="100k"
    model_type="LSTM"
    sample_path = "../../data/2_train/Collection1_cos_100.csv"
    dataset = f"../../data/2_train/Collection1_cos_{sample}.csv"
    model_name=f"{model_type}_{dataset.split('/')[-1].split('.')[0].split('_')[0]}_{sample}_{hidden_size}_{num_layers}"
    finetune_folder_path=f"data/{sample}/origin_feature/{n_cluster}_{sample}_scaler/"
    csv_files = [finetune_folder_path + f for f in os.listdir(finetune_folder_path) if f.endswith('.csv')]
    finetune_save=finetune_folder_path+"finetune"
    scaler = StandardScaler()
    with open(f'data/{sample}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    finetune_path_list=[]
    for f in csv_files:
        finetune_path_list.append(f"{finetune_save}/finetune{finetune_epoch}_{model_name}_cluster_{n_cluster}_{f.split('/')[-1].split('.')[0].split('_')[-1]}.pth")

    finetune_path_list=sorted(finetune_path_list,key=lambda x:int(x.split('.')[0].split('_')[-1]))

    beam_model_list=[]
    for finetune_model_path in finetune_path_list:
        model = BeamSearchEvaluator(finetune_model_path, device=DEFAULT_DEVICE)
        beam_model_list.append(model)

    save_guessrate_file=f"{finetune_save}/gate{chosen_num}finetune{finetune_epoch}_{model_name}_cluster_{n_cluster}".split('/')[-1]+f"_guess_{sample_path.split('/')[-1].split('.')[0]}.txt"

    cluster_info_path = f"{finetune_folder_path}/cluster_info.npy"
    
    evaluator = MoPEEvaluator(beam_model_list,save_guessrate_file,scaler,cluster_info_path,n_cluster=n_cluster,threshold=5)
    evaluator.evaluate(sample_path,chosen_num=chosen_num)


if __name__ == '__main__':
    main()