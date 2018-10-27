# coding=utf-8

"""
Implementation of Bilateral Multi Perspective Matching in PyTorch

Usage:
    main.py vocab
    main.py train
    main.py test
    main.py train --train-src=<file> --dev-src=<file> [options]
    main.py test --test-src=<file> MODEL_PATH [options] 

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file [default: ../data/quora/train.tsv]
    --dev-src=<file>                        dev source file [default: ../data/quora/dev.tsv]
    --test-src=<file>                       test source file [default: ../data/quora/test.tsv]
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 300]
    --char-embed=<int>                      char embedding size [default: 20]
    --bi-hidden-size=<int>                  bidirectional lstm hidden size [default: 100]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --lr=<float>                            learning rate [default: 0.001]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.1]
    --data=<str>                            type of dataset [default: quora]
"""

from docopt import docopt
from parser import dataparser
from pdb import set_trace as bp

def main(args):
    print("hello")

if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
