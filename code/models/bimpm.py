import torch
import torch.nn as nn
from pdb import set_trace as bp

class BiMPM(nn.Module):

    def __init__(self, args, vocab, class_size):
        super(BiMPM, self).__init__()

        self.c_embed_size = int(args['--char-embed-size'])
        self.w_embed_size = int(args['--embed-size'])
        self.l = int(args['--perspective'])
        self.dropout = float(args['--dropout'])
        self.bi_hidden = int(args['--bi-hidden-size'])
        self.char_hidden = int(args['--char-hidden-size'])
        self.rnn_type = args['--rnn-type']
        self.char_layer_size = int(args['--char-lstm-layers'])
        self.context_layer_size = int(args['--bilstm-layers'])
        self.char_inp = vocab
        self.classes = class_size

        self.char_embedding = nn.Embedding(num_embeddings = self.char_inp,\
                                           embedding_dim = self.c_embed_size)

        self.char_lstm = nn.LSTM(input_size = self.c_embed_size,\
                                 hidden_size = self.char_hidden,\
                                 num_layers = self.char_layer_size)

        self.context_lstm = nn.LSTM(input_size = self.w_embed_size,\
                                    hidden_size = self.bi_hidden,\
                                    num_layers = self.context_layer_size,\
                                    bidirectional = True)

        self.aggregation_lstm = nn.LSTM(input_size = self.l,
                                         hidden_size = self.bi_hidden)


        for i in range(1, 9):
            setattr(self, f'w{i}', nn.Parameter(torch.rand(self.l, self.bi_hidden)))

        self.ff1 = nn.Linear(self.bi_hidden * 4, self.bi_hidden * 2)
        self.ff2 = nn.Linear(self.bi_hidden * 2, self.classes)

        bp()        
