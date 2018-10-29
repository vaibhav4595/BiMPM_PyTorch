import pymagnitude
import torch
from torch.nn.utils.rnn import pad_sequence
from pdb import set_trace as bp


class Model(object):
    def __init__(self, args, vocab_data):
        self.data = args['--data']
        self.char_use = True
        self.cembed_size = int(args['--char-embed-size'])
        self.wembed_size = int(args['--embed-size'])
        self.perspec = int(args['--perspective'])
        self.cuda = args['--cuda']
        self.vocab = vocab_data
        self.wembeddings = pymagnitude.Magnitude(args['--glove-path'])

    def forward(self, label, p1, p2):
        word_tensor1, word_tensor2, char_tensor1, char_tensor2, label_tensor =\
                self.format_data(label, p1, p2)
        label_tensor = label_tensor.unsqueeze(1)

        bp()

    def format_data(self, label, p1, p2):
        word_p1_inp = []
        maxp1 = 0
        
        # Construct GloVe for each word, and find max word length (for one sentence)
        for each in p1:
            word_p1_inp.append(torch.FloatTensor(self.wembeddings.query(each)))
            for word in each:
                maxp1 = max(maxp1, len(word))

        word_p2_inp = []
        maxp2 = 0

        # Construct GloVe for each word, and find max word length (for other sentence)
        for each in p2:
            word_p2_inp.append(torch.FloatTensor(self.wembeddings.query(each)))
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
            char_p1_inp.append(torch.FloatTensor(sent_arr))

        # Intiliase character indices for each word of the sentence2
        for sent in p2:
            sent_arr = []
            for word in sent:
                word_arr = [self.vocab.char2id(char) for char in word]
                word_arr = word_arr + [0] * (maxp2 - len(word))
                sent_arr.append(word_arr)
            char_p2_inp.append(torch.FloatTensor(sent_arr))

        char_p1_inp = pad_sequence(char_p1_inp)
        char_p2_inp = pad_sequence(char_p2_inp)

        # Initiliase label tensor
        label = torch.FloatTensor([int(each) for each in label])

        return (word_p1_inp, word_p2_inp, char_p1_inp, char_p2_inp, label)

    def set_labels(self, value):
        self.classes = value
