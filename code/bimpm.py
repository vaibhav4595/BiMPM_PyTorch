import pymagnitude
from pdb import set_trace as bp


class BiMPM(object):
    def __init__(self, args, vocab_data):
        self.data = args['--data']
        self.char_use = True
        self.cembed_size = int(args['--char-embed-size'])
        self.wembed_size = int(args['--embed-size'])
        self.cuda = args['--cuda']
        self.vocab = vocab_data
        self.wembeddings = pymagnitude.Magnitude(args['--glove-path'])

    def forward(self, label, p1, p2):
        word_p1_inp = [self.wembeddings.query(each) for each in p1]
        word_p2_inp = [self.wembeddings.query(each) for each in p2]

        word_p1_inp = []
        char_p1_inp = []
        maxp1 = 0
        for each in p1:
            word_p1_inp.append(self.wembeddings.query(each))
            for word in each:
                maxp1 = max(maxp1, len(word))
                char_p1_inp.append([self.vocab.char2id(char) for char in word])

        word_p2_inp = []
        char_p2_inp = []
        maxp2 = 0
        for each in p2:
            word_p2_inp.append(self.wembeddings.query(each))
            for word in each:
                maxp2 = max(maxp2, len(word))
                char_p2_inp.append([self.vocab.char2id(char) for char in word])

        if self.char_use:
            for i in range(len(char_p1_inp)):
                for j in range(0, maxp1 - len(char_p1_inp[i])):
                    char_p1_inp[i].append(0)
            for i in range(len(char_p2_inp)):
                for j in range(0, maxp2 - len(char_p2_inp[i])):
                    char_p2_inp[i].append(0)

        bp()
    def set_labels(self, value):
        self.classes = value
