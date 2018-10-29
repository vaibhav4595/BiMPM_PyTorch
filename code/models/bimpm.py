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
        self.char_inp = vocab + 100
        self.classes = class_size
        self.char_use = args['--char']

        if self.char_use:
            self.char_embedding = nn.Embedding(num_embeddings=self.char_inp,\
                                               embedding_dim=self.c_embed_size,\
                                               padding_idx=0)

            self.char_lstm = nn.LSTM(input_size=self.c_embed_size,\
                                     hidden_size=self.char_hidden,\
                                     num_layers=self.char_layer_size)

            self.context_lstm = nn.LSTM(input_size=self.w_embed_size + self.char_hidden,\
                                        hidden_size=self.bi_hidden,\
                                        num_layers=self.context_layer_size,\
                                        bidirectional=True)

        else:
            self.context_lstm = nn.LSTM(input_size=self.w_embed_size,\
                                        hidden_size=self.bi_hidden,\
                                        num_layers=self.context_layer_size,\
                                        bidirectional=True)



        self.aggregation_lstm = nn.LSTM(input_size = self.l,
                                         hidden_size = self.bi_hidden)


        for i in range(1, 9):
            setattr(self, f'w{i}', nn.Parameter(torch.rand(self.l, self.bi_hidden)))

        self.ff1 = nn.Linear(self.bi_hidden * 4, self.bi_hidden * 2)
        self.ff2 = nn.Linear(self.bi_hidden * 2, self.classes)


    def init_char_embed(self, c1, c2):
        c1_embed = self.char_embedding(c1)
        char_p1 = self.char_lstm(c1_embed)
        c2_embed = self.char_embedding(c2)
        char_p2 = self.char_lstm(c2_embed)
        return char_p1[0][-1], char_p2[0][-1]

    def forward(self, p1, p2, c1, c2):

        if self.char_use:
            char_p1, char_p2 = self.init_char_embed(c1, c2)
            dim1, dim2, _ = p1.size()
            char_p1 = char_p1.view(dim1, dim2, -1)
            dim1, dim2, _ = p2.size()
            char_p2 = char_p2.view(dim1, dim2, -1)
            p1_input = torch.cat((p1, char_p1), 2)
            p2_input = torch.cat((p2, char_p2), 2)

            context1 = self.context_lstm(p1_input)
            context2 = self.context_lstm(p2_input)

        else:
            context1 = self.context_lstm(p1)
            context2 = self.context_lstm(p2)

        bp()
        return None
