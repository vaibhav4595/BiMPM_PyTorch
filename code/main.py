# coding=utf-8

"""
Implementation of Bilateral Multi Perspective Matching in PyTorch

Usage:
    main.py vocab
    main.py train
    main.py test
    main.py train --train-src=<file> --dev-src=<file> --vocab-src=<file> [options]
    main.py test --test-src=<file> --vocab-src=<file> MODEL_PATH [options] 

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file [default: ../data/quora/train_small.tsv]
    --dev-src=<file>                        dev source file [default: ../data/quora/dev_small.tsv]
    --test-src=<file>                       test source file [default: ../data/quora/test.tsv]
    --vocab-src=<file>                      vocab source file [default: ../data/quora/vocab.pkl]
    --glove-path=<file>                     pretrained glove embedding file [default: ../data/glove/glove.840B.300d.magnitude]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --rnn-type=<str>                        type of rnn (lstm, gru, rnn) [default: lstm]
    --embed-size=<int>                      embedding size [default: 300]
    --char-embed-size=<int>                 char embedding size [default: 20]
    --bi-hidden-size=<int>                  bidirectional lstm hidden size [default: 100]
    --char-hidden-size=<int>                character lstm hidden size [default: 50]
    --char-lstm-layers=<int>                number of layers in character lstm [default: 1]
    --bilstm-layers=<int>                   number of layers in bidi lstm [default: 1]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 50]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --lr=<float>                            learning rate [default: 0.001]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.1]
    --data=<str>                            type of dataset [default: quora]
    --perspective=<int>                     number of perspectives for the model [default: 20]
    --char                                  whether to use character embeddings or not, default is true [default: True]
"""

from docopt import docopt
from pdb import set_trace as bp
from vocab import Vocab
from models.model import Model
import utils
import pymagnitude

def train(args):
    train_path = args['--train-src']
    dev_path = args['--dev-src']
    vocab_path = args['--vocab-src']

    if args['--data'] == 'quora':
        train_data = utils.read_data(train_path, 'quora')
        dev_data = utils.read_data(dev_path, 'quora')
        vocab_data = utils.load_vocab(vocab_path)
        network = Model(args, vocab_data, 2)

    epoch = 0
    batch_size = int(args['--batch-size'])
    while True:
        epoch += 1
        
        for labels, p1, p2, idx in utils.batch_iter(train_data, batch_size):
            network.forward(labels, p1, p2)

        break
        
def main(args):
    if args['train']:
        train(args)
    elif args['test']:
        test(args)
    elif args['vocab']:
        print("Need to use another file for this")

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
