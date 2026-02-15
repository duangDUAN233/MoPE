from cluster import make_cluster_data
from train import train
from finetune import make_finetune


if __name__=="__main__":
    hidden_size=256
    num_layer=1
    sample="01M"
    make_cluster_data(data_path="../../data", sample=sample, cluster_start=30, n_cluster=9)
    train(model_type="LSTM", hidden_size=hidden_size, num_layer=num_layer, sample=sample, max_len=16, embedding_size=64,epochs=30, batch_size=128, lr=0.001, dropout_ratio=0.15)
    make_finetune(hidden_size=hidden_size, num_layer=num_layer, finetune_start=1, finetune_end=2, cluster_start=30, cluster_end=71, model_type="LSTM",sample=sample)


