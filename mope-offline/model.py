import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, model_type="LSTM",
                 num_layers=1, dropout_ratio=0.0, sequence_model="MANY_TO_ONE"):
        super().__init__()
        self.model_type = model_type.upper()
        self.sequence_model = sequence_model.upper()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        if self.model_type == "LSTM":
            self.rnn = nn.LSTM(
                embedding_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_ratio if num_layers > 1 else 0.0
            )
        elif self.model_type == "GRU":
            self.rnn = nn.GRU(
                embedding_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout_ratio if num_layers > 1 else 0.0
            )
        elif self.model_type == "CONV1D":
            self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=hidden_size, kernel_size=3, padding=1)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # final output layer
        self.dropout = nn.Dropout(dropout_ratio)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # many to many -> return logits seq
        if self.sequence_model == "MANY_TO_MANY":
            self.fc = nn.Linear(hidden_size, vocab_size)
        else:
            self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)  # [B, T, embedding_size]
        if self.model_type in ["LSTM", "GRU"]:
            if self.sequence_model == "MANY_TO_ONE":
                # final hidden state
                rnn_out, _ = self.rnn(emb)  # [B, T, hidden_size]
                last_step = rnn_out[:, -1, :]  # [B, hidden_size]
                out = self.dropout(last_step)
                logits = self.fc(out)         # [B, vocab_size]
                return logits

            elif self.sequence_model == "MANY_TO_MANY":
                rnn_out, _ = self.rnn(emb)   # [B, T, hidden_size]
                rnn_out = self.dropout(rnn_out)
                logits = self.fc(rnn_out)    # [B, T, vocab_size]
                return logits

        elif self.model_type == "CONV1D":
            # expected: [B, in_channels, T]
            # [B, T, emb]->[B, emb, T]
            emb_t = emb.permute(0, 2, 1)    # [B, emb, T]
            conv_out = self.conv(emb_t)     # [B, hidden_size, T]
            conv_out = conv_out.permute(0, 2, 1)  # [B, T, hidden_size]

            if self.sequence_model == "MANY_TO_ONE":
                last_step = conv_out[:, -1, :]
                out = self.dropout(last_step)
                logits = self.fc(out)
                return logits
            else:
                # many_to_many
                out = self.dropout(conv_out)
                logits = self.fc(out)  # [B, T, vocab_size]
                return logits
        else:
            raise ValueError(f"Unsupported forward for {self.model_type}")
        
    
    def get_hidden(self,x):
        emb = self.embedding(x)  # [B, T, embedding_size]
        # final hidden state
        rnn_out, _ = self.rnn(emb)  # [B, T, hidden_size]
        return rnn_out
