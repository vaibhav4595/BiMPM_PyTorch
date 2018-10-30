import torch
import torch.nn as nn
import torch.nn.functional as F
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
            setattr(self, f'w{i}', nn.Parameter(torch.rand(1, 1, self.bi_hidden, self.l)))

        self.ff1 = nn.Linear(self.bi_hidden * 4, self.bi_hidden * 2)
        self.ff2 = nn.Linear(self.bi_hidden * 2, self.classes)


    def init_char_embed(self, c1, c2):
        c1_embed = self.char_embedding(c1)
        char_p1 = self.char_lstm(c1_embed)
        c2_embed = self.char_embedding(c2)
        char_p2 = self.char_lstm(c2_embed)
        return char_p1[0][-1], char_p2[0][-1]


    def cosine_similarity(self, prod, norm):
        # As set in PyTorch documentation
        eps = 1e-8
        norm = norm * (norm > eps).float() + eps * (norm <= eps).float()

        return prod / norm

    def full_matching(self, p1, p2, w_matrix):
        p1 = torch.stack([p1] * self.l, dim = 3)
        p1 = w_matrix * p1

        p1_seq_len = p1.size(0)
        p2 = torch.stack([p2] * p1_seq_len, dim = 0)
        p2 = torch.stack([p2] * self.l, dim = 3)
        p2 = w_matrix * p2
        result = F.cosine_similarity(p1, p2)
        return result

    def maxpool_matching(self, p1, p2, w_matrix):
        p1 = torch.stack([p1] * self.l, dim = 3)
        p1 = w_matrix * p1
        
        p2 = torch.stack([p2] * self.l, dim = 3)
        p2 = w_matrix * p2

        p1_norm = p1.norm(p = 2, dim = 2, keepdim=True)
        p2_norm = p2.norm(p = 2, dim = 2, keepdim=True)

        full_mat = torch.matmul(p1.permute(1, 3, 0, 2), p2.permute(1, 3, 2, 0))
        deno_mat = torch.matmul(p1_norm.permute(1, 3, 0, 2), p2_norm.permute(1, 3, 2, 0))

        result, _ = self.cosine_similarity(full_mat, deno_mat).max(dim = 3)
        result = result.permute(2, 0, 1)
        return result

    def attentive_matching(self):
        return None

    def max_attentive_matching(self):
        return None

    def forward(self, p1, p2, c1, c2, p1_len, p2_len):

        if self.char_use:
            char_p1, char_p2 = self.init_char_embed(c1, c2)
            dim1, dim2, _ = p1.size()
            char_p1 = char_p1.view(dim1, dim2, -1)
            dim1, dim2, _ = p2.size()
            char_p2 = char_p2.view(dim1, dim2, -1)
            p1_input = torch.cat((p1, char_p1), 2)
            p2_input = torch.cat((p2, char_p2), 2)

            context1_full, (context1_lh, _) = self.context_lstm(p1_input)
            context2_full, (context2_lh, _) = self.context_lstm(p2_input)

        else:
            context1_full, (context1_lh, _) = self.context_lstm(p1)
            context2_full, (context2_lh, _) = self.context_lstm(p2)

        context1_forw, context1_back = torch.split(context1_full, self.bi_hidden, 2)
        context1_lh_forw, context1_lh_back = context1_lh[0], context1_lh[1]

        context2_forw, context2_back = torch.split(context2_full, self.bi_hidden, 2)
        context2_lh_forw, context2_lh_back = context2_lh[0], context2_lh[1]

        # 4 tensors from forward and backward matching (full matching)
        match_p1_forw = self.full_matching(context1_forw, context2_lh_forw, self.w1)
        match_p1_back = self.full_matching(context1_back, context2_lh_back, self.w2)
        match_p2_forw = self.full_matching(context2_forw, context1_lh_forw, self.w1)
        match_p2_back = self.full_matching(context2_back, context1_lh_back, self.w2)

        # 4 tensors from forward and backward matching (max-pooling matching)
        maxm_p1_forw = self.maxpool_matching(context1_forw, context2_forw, self.w3)
        maxm_p1_back = self.maxpool_matching(context1_back, context2_back, self.w4)
        maxm_p2_forw = self.maxpool_matching(context2_forw, context1_forw, self.w3)
        maxm_p2_back = self.maxpool_matching(context2_back, context1_back, self.w4)
        return None
