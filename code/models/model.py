import pymagnitude
import torch
from torch.nn.utils.rnn import pad_sequence
from pdb import set_trace as bp
from models.bimpm import BiMPM


class Model(object):
    def __init__(self, args, vocab_data, class_size):
        self.data = args['--data']
        self.char_use = True
        self.cembed_size = int(args['--char-embed-size'])
        self.wembed_size = int(args['--embed-size'])
        self.perspec = int(args['--perspective'])
        self.cuda = args['--cuda']
        self.vocab = vocab_data
        self.vocab_len = len(vocab_data.char2index)
        self.wembeddings = pymagnitude.Magnitude(args['--glove-path'])
        self.model = BiMPM(args, self.vocab_len, class_size)
        self.criterion = torch.nn.CrossEntropyLoss()

        if self.cuda == True:
            self.criterion.cuda()

    def forward(self, label, p1, p2, method='train'):
        word_tensor1, word_tensor2, char_tensor1, char_tensor2, label_tensor,\
                p1_orig_len, p2_orig_len = self.format_data(label, p1, p2)
        label_tensor = label_tensor
        if self.cuda == True:
            word_tensor1.cuda()
            word_tensor2.cuda()
            char_tensor1.cuda()
            char_tensor2.cuda()
            label_tensor.cuda()
            p1_orig_len.cuda()
            p2_orig_len.cuda()

        predicted = self.model(word_tensor1, word_tensor2, char_tensor1, char_tensor2,\
                          p1_orig_len, p2_orig_len)
        loss = self.criterion(predicted, label_tensor)
        return predicted, loss

    def evaluate(self, label, p1, p2):
        return None

    def format_data(self, label, p1, p2):
        word_p1_inp = []
        maxp1 = 0
        p1_orig_length = []        
        # Construct GloVe for each word, and find max word length (for one sentence)
        for each in p1:
            p1_orig_length.append(len(each))
            word_p1_inp.append(torch.Tensor(self.wembeddings.query(each)))
            for word in each:
                maxp1 = max(maxp1, len(word))

        word_p2_inp = []
        maxp2 = 0
        p2_orig_length = []
        # Construct GloVe for each word, and find max word length (for other sentence)
        for each in p2:
            p2_orig_length.append(len(each))
            word_p2_inp.append(torch.Tensor(self.wembeddings.query(each)))
            for word in each:
                maxp2 = max(maxp2, len(word))

        word_p1_inp = pad_sequence(word_p1_inp)
        word_p2_inp = pad_sequence(word_p2_inp)
        char_p1_inp = []
        char_p2_inp = []

        # Initiliase character indices for each word of the sentence1
        for sent in p1:
            sent_arr = []
            for word in sent:
                word_arr = [self.vocab.char2id(char) for char in word]
                word_arr = word_arr + [0] * (maxp1 - len(word))
                sent_arr.append(word_arr)
            char_p1_inp.append(torch.LongTensor(sent_arr))

        # Intiliase character indices for each word of the sentence2
        for sent in p2:
            sent_arr = []
            for word in sent:
                word_arr = [self.vocab.char2id(char) for char in word]
                word_arr = word_arr + [0] * (maxp2 - len(word))
                sent_arr.append(word_arr)
            char_p2_inp.append(torch.LongTensor(sent_arr))

        char_p1_inp = pad_sequence(char_p1_inp)
        char_p1_inp = char_p1_inp.view(-1, maxp1).t()

        char_p2_inp = pad_sequence(char_p2_inp)
        char_p2_inp = char_p2_inp.view(-1, maxp2).t()

        # Initiliase label tensor
        label = torch.LongTensor([int(each) for each in label])

        return (word_p1_inp, word_p2_inp, char_p1_inp, char_p2_inp, label,\
                p1_orig_length, p2_orig_length)

    def set_labels(self, value):
        self.classes = value

    def get_label(self, label):
        label = [int(each) for each in label]
        if self.cuda:
            return torch.LongTensor(label).cuda()
        else:
            return torch.LongTensor(label)
