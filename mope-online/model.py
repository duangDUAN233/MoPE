# model.py

import operator
import numpy as np
import random
import os, errno
import sys
from tqdm import trange
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import heapq

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class BeamSearchNode(object):
    def __init__(self, hc, hx, previousNode, wordId, logProb):
        self.hc = hc 
        self.hx = hx
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb

    def eval(self):
        return self.logp
    
    def __lt__(self, node):
        return self.logp < node.logp

class MaxHeap:
    def __init__(self, max_len=10):
        self.max_len = max_len
        self.heap = []
    
    def can_push(self, value):
        if len(self.heap) < self.max_len:
            return True
        if self.heap[0][0] >= -value:
            return False
        return True
    
    def push(self, value, node):
        if len(self.heap) < self.max_len:
            heapq.heappush(self.heap, (-value, node))
        else:
            heapq.heappushpop(self.heap,(-value, node))
    
    def get_queue(self):
        return [x[1] for x in self.heap]
    
    def __len__(self):
        return len(self.heap)

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 3, embed_size=200, dropout=0.3):
        
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.embedding = nn.Embedding(num_embeddings = input_size, embedding_dim = self.embed_size)
        self.input_fc = nn.Linear(
            self.embed_size, 
            self.hidden_size
        )
        self.lstm = nn.LSTM(
            input_size = self.hidden_size, 
            hidden_size = self.hidden_size, 
            num_layers = num_layers, 
            batch_first=True, 
            dropout=dropout)
    
    def forward(self, x_input, encoder_hidden=None):
        '''
        : param x_input:               input of shape (seq_len, # in batch, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence 
        '''
        out = self.embedding(x_input)
        out = self.input_fc(out)
        _, self.hidden = self.lstm(out, encoder_hidden)
        return self.hidden
    
    def init_hidden(self, batch_size, device=torch.device('cpu')):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state 
        '''
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))


class LSTMDecoder(nn.Module):
    ''' Decodes hidden state output by encoder '''
    def __init__(self, decode_size, hidden_size, num_layers = 3, embed_size=128, dropout=0.3):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''
        super(LSTMDecoder, self).__init__()
        self.vocab_size = decode_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.embedding = nn.Embedding(num_embeddings = decode_size, embedding_dim = self.embed_size)
        self.input_fc = nn.Linear(
            self.embed_size, 
            self.hidden_size
        )
        self.lstm = nn.LSTM(
            input_size = self.hidden_size, 
            hidden_size = self.hidden_size,
            num_layers = num_layers, 
            batch_first=True, 
            dropout=dropout)
        self.linear = nn.Linear(hidden_size, decode_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.train_mode = "train"

    def forward(self, x_input, encoder_hidden_states):
        
        '''        
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence 
        '''
        output = self.embedding(x_input)
        output = self.input_fc(output)
        lstm_out, hidden = self.lstm(output, encoder_hidden_states)
        output = self.linear(lstm_out)
        output = self.softmax(output)
        return output, hidden



    def init_start_token(self, batch_size, start_token, device=torch.device('cpu')):
        return torch.tensor(
            [[start_token] for _ in range(batch_size)], dtype=torch.long).to(device)


class Expert(nn.Module):
    def __init__(self, vocab_size, trans_size, pad_token,start_token,end_token,hidden_size=128, 
                 embed_size=200, maxlen=32, num_layers=1, dropout=0.4,device=torch.device('cpu')):
        super(Expert, self).__init__()
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.input_size = vocab_size
        self.output_size = trans_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.mode = "train"
        self.maxlen = maxlen
        self.device = device

        self.encoder = LSTMEncoder(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            embed_size=self.embed_size,
            dropout=dropout
        )

        self.decoder = LSTMDecoder(
            decode_size=self.output_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            embed_size=self.embed_size,
            dropout=dropout
        )

    def set_mode(self,mode):
        self.mode = mode
        self.decoder.train_mode = mode

    def forward_train(self, pwds, edits):
        batch_size = pwds.size(0)
        encoder_hidden = self.encoder(pwds)
        decoder_input = self.decoder.init_start_token(batch_size, self.start_token, self.device)
        decoder_hidden = encoder_hidden
        maxlen = edits.size(1)
        outputs = torch.zeros(batch_size, maxlen, self.output_size).to(self.device)
        for t in range(maxlen): 
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, t] = decoder_output.squeeze(1)
            decoder_input = edits[:, t].unsqueeze(1)
        return outputs
    
    def forward_beamsearch(self, pwds, beamwidth=10, topk=10, max_predict=6):
        '''
        Warning: beamwidth * max_predict > topk
        '''
        batch_size = pwds.size(0)
        encoder_hidden = self.encoder(pwds)
        encoder_hc = encoder_hidden[0]
        encoder_hx = encoder_hidden[1]
        maxlen = self.maxlen
        outputs = [None for _ in range(batch_size)]
        for t in range(batch_size):
            decoder_hc = encoder_hc[:, t:t+1]
            decoder_hx = encoder_hx[:, t:t+1]
            node_id = 0
            root = (node_id, decoder_hc, decoder_hx, None, self.start_token, 0)
            node_id += 1

            queue = MaxHeap(beamwidth)
            queue.push(0, root)
            end_nodes = []
            predict_cnt = 0
            while max_predict > predict_cnt and len(queue) > 0 and len(end_nodes) < topk:
                Q = queue.get_queue()
                decoder_hc = torch.cat([node[1] for node in Q], dim=1)
                decoder_hx = torch.cat([node[2] for node in Q], dim=1)
                decoder_hidden = (decoder_hc, decoder_hx)
                decoder_input = torch.tensor([[node[4]] for node in Q]).to(self.device)
                decoder_output, decoder_hidden = self.decoder(decoder_input, (decoder_hc, decoder_hx))
                decoder_hc = decoder_hidden[0]
                decoder_hx = decoder_hidden[1]
                topv, topi = decoder_output.squeeze(1).topk(beamwidth, dim=1)
                topv = -topv
                topv = topv.tolist()
                topi = topi.tolist()
                queue = MaxHeap(beamwidth)
                for i in range(len(Q)):
                    node = Q[i]
                    hc = decoder_hc[:, i:i+1]
                    hx = decoder_hx[:, i:i+1]
                    for j in range(beamwidth):
                        topi_v = topi[i][j]
                        logp = node[-1]+topv[i][j]
                        if topi_v == self.end_token:
                            end_nodes.append((node_id, hc, hx, node, topi_v, logp))
                            node_id += 1
                        else:
                            if queue.can_push(logp):
                                queue.push(logp, (node_id, hc, hx, node, topi_v, logp))
                                node_id += 1
                predict_cnt += 1
            guesses = []
            for node in end_nodes:
                edits = []
                cur = node 
                while cur[3] != None:
                    edits.append((cur[4]))
                    cur = cur[3]
                edits = edits[::-1]
                guesses.append((edits, node[-1]))
            guesses = sorted(guesses, key=lambda x:x[1])
            outputs[t] = guesses
        return outputs


    def forward(self, pwds, edits=None, beamwidth=10, topk=10):
        ans = None
        if self.mode == "train":
            ans = self.forward_train(pwds, edits)
        elif self.mode == "predict":
            with torch.no_grad():
                pass
        elif self.mode == "beamsearch":
            with torch.no_grad():
                ans = self.forward_beamsearch(pwds,beamwidth=beamwidth, topk=topk)
        return ans
