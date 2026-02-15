# train.py

from tokenizer import *
import numpy as np
from tqdm import tqdm
import torch
from model import *
from dataloader import *
from loss import *

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ExpertTrainer:
    def __init__(
            self, 
            t1=KBDPasswordTokenizer(), 
            t2=TransTokenizer(), 
            epoch=50, 
            batch_size=4, 
            hidden_size=128, 
            embed_size=200, 
            num_layers=3,
            max_len=16, 
            dropout=0.4, 
            lr=0.001,
            device=DEFAULT_DEVICE,
            weight=0.2,
            ):
        self.t1 = t1
        self.t2 = t2
        self.pad_token = self.t2.pad_token_id
        self.model = Expert(
            len(self.t1), 
            len(self.t2), 
            pad_token=self.t2.pad_token_id, 
            start_token=self.t2.start_token_id, 
            end_token=self.t2.end_token_id,
            hidden_size=hidden_size, 
            embed_size=embed_size, 
            num_layers=num_layers, 
            maxlen=max_len, 
            dropout=dropout,
            device=device)
        self.max_len = max_len
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.model.set_mode("train")
        self.device = device
        self.model = self.model.to(self.device)
        self.weight = weight
        pass
    
    def train(self, data_path, model_save):
        dataset = mope_dataloader(
            data_path, 
            t1 = self.t1, 
            t2 = self.t2, 
            batch_size=self.batch_size)
        losses = np.full(self.epoch, np.nan)
        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)

        criterion = CustomBCEWithLogitsLoss()
        
        self.model.train()
        for iter in range(self.epoch):
            batch_loss = []
            with tqdm(dataset, desc="Training", leave=False) as dataset_wrapper:
                for pwds, edits in dataset_wrapper:
                    pwds = pwds.to(self.device)
                    
                    # print("edits=",edits)
                    label = self.t2.one_hot_encode(edits,weight=self.weight)
                    # print("labels=",label)
                    # print("pwds=",pwds)
                    edits = edits.to(self.device)
                    label = label.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(pwds, edits)
                    batch_size = edits.size(0)
                    loss = criterion(outputs, label)
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item() / batch_size)
                    avg_loss = sum(batch_loss) / ((len(batch_loss)+1))
                    dataset_wrapper.set_postfix(loss=avg_loss)
            avg_loss = sum(batch_loss) / ((len(batch_loss)+1))
            print(f">>> Epoch: {iter}, Loss: {avg_loss}")
            losses[iter] = avg_loss
        self.save(model_path=model_save)
        return losses
 
    def save(self, model_path):
        torch.save(self.model, model_path)
        print(f">>> Model saved in {model_path}")
        pass


class ExpertFinetuner:
    def __init__(
            self, 
            model,
            t1=KBDPasswordTokenizer(), 
            t2=TransTokenizer(), 
            finetune_epoch=5, 
            batch_size=4, 
            hidden_size=128, 
            embed_size=200, 
            num_layers=3,
            max_len=16, 
            dropout=0.4, 
            lr=0.001,
            device=DEFAULT_DEVICE,
            weight=0,            
            
            ):
        self.t1 = t1
        self.t2 = t2
        self.pad_token = self.t2.pad_token_id
        self.model=model
        self.model.to(device)
        self.max_len = max_len
        self.epoch = finetune_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.model.set_mode("train")
        self.device = device
        self.model = self.model.to(self.device)
        self.weight = weight
        pass
    
    def train(self, data_path, model_save):
        dataset = mope_dataloader(
            data_path, 
            t1 = self.t1, 
            t2 = self.t2, 
            batch_size=self.batch_size)
        losses = np.full(self.epoch, np.nan)
        optimizer = optim.Adam(self.model.parameters(), lr = self.lr)

        criterion = CustomBCEWithLogitsLoss()
        
        self.model.train()
        for iter in range(self.epoch):
            batch_loss = []
            with tqdm(dataset, desc="Training", leave=False) as dataset_wrapper:
                for pwds, edits in dataset_wrapper:
                    pwds = pwds.to(self.device)
                    
                    # print("edits=",edits)
                    label = self.t2.one_hot_encode(edits,weight=self.weight)
                    # print("labels=",label)
                    # print("pwds=",pwds)
                    edits = edits.to(self.device)
                    label = label.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(pwds, edits)

                    # print("outputs=",outputs)
                    batch_size = edits.size(0)
                    loss = criterion(outputs, label)
                    
                    # print("loss=",loss)
                    loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item() / batch_size)
                    avg_loss = sum(batch_loss) / ((len(batch_loss)+1))
                    dataset_wrapper.set_postfix(loss=avg_loss)
            avg_loss = sum(batch_loss) / ((len(batch_loss)+1))
            print(f">>> Epoch: {iter}, Loss: {avg_loss}")
            losses[iter] = avg_loss
        self.save(model_path=model_save)
        return losses
 
    def save(self, model_path):
        torch.save(self.model, model_path)
        print(f">>> Model saved in {model_path}")
        pass
       

def train(epoch=10,batch_size=128,hidden_size=256,embed_size=256,
          num_layers=3,max_len=16,dropout=0.4,lr=1e-2,device=DEFAULT_DEVICE,
          weight=0.1,model_type="LSTM",sample="100k",dataset="../../data/2_train/Collection1_cos_100k.csv"):
    trainer = ExpertTrainer(
            epoch=epoch, 
            batch_size=batch_size, 
            hidden_size=hidden_size, 
            embed_size=embed_size, 
            num_layers=num_layers,
            max_len=max_len, 
            dropout=dropout, 
            lr=lr, 
            device=device,
            weight=weight,
        )
    model_name=f"{model_type}_{dataset.split('/')[-1].split('.')[0].split('_')[0]}_{sample}_{hidden_size}_{num_layers}"
    model_save = f"weights/{model_name}.pth"
    trainer.train(dataset, model_save)

def finetune(finetune_epoch=5,batch_size=128,hidden_size=256,n_cluster=25,
             embed_size=256,num_layers=3,max_len=16,dropout=0.4,lr=1e-2,
             device=DEFAULT_DEVICE,weight=0.1,model_type="LSTM",
             sample="100k",dataset="../../data/2_train/Collection1_cos_100k.csv"):

    model_name=f"{model_type}_{dataset.split('/')[-1].split('.')[0].split('_')[0]}_{sample}_{hidden_size}_{num_layers}"
    basic_model_path = f"weights/{model_name}.pth"  
    finetune_folder_path=f"data/{sample}/origin_feature/{n_cluster}_{sample}_scaler/"

    model = torch.load(basic_model_path)

    finetuner=ExpertFinetuner(
        finetune_epoch=finetune_epoch,
        model=model,
        batch_size=batch_size, 
        hidden_size=hidden_size, 
        embed_size=embed_size, 
        num_layers=num_layers,
        max_len=max_len, 
        dropout=dropout, 
        lr=lr, 
        device=device,
        weight=weight,
    )

    csv_files = [finetune_folder_path + f for f in os.listdir(finetune_folder_path) if f.endswith('.csv')]

    finetune_save=finetune_folder_path+"finetune"
    os.makedirs(finetune_save, exist_ok=True)

    for f in csv_files:
        save_path=f"{finetune_save}/finetune{finetune_epoch}_{model_name}_cluster_{n_cluster}_{f.split('/')[-1].split('.')[0].split('_')[-1]}.pth"
        finetuner.train(f,save_path)


def main():
    epoch=10
    batch_size=128
    hidden_size=256
    embed_size=256
    num_layers=3
    max_len=16
    dropout=0.4
    lr=1e-2
    device=DEFAULT_DEVICE
    weight=0
    model_type="LSTM"
    sample="100k"
    finetune_epoch=4
    dataset = f"../../data/2_train/Collection1_cos_{sample}.csv"
    n_cluster=65

    train(epoch=epoch,batch_size=batch_size,hidden_size=hidden_size,
          embed_size=embed_size,num_layers=num_layers,max_len=max_len,
          dropout=dropout,lr=lr,device=device,weight=weight,
          model_type=model_type,sample=sample,dataset=dataset)

    finetune(finetune_epoch=finetune_epoch,batch_size=batch_size,hidden_size=hidden_size,n_cluster=n_cluster,
            embed_size=embed_size,num_layers=num_layers,max_len=max_len,dropout=dropout,lr=lr,
            device=device,weight=weight,model_type=model_type,
            sample=sample,dataset=dataset)


if __name__ == '__main__':
    main()