# simple lstm RNN experiment

import numpy as np
import torch
from argparse import ArgumentParser


args = ArgumentParser(description="test lstm-rnn")
args.add_argument("--vocab_size", type = int, default = 100, help = "vocabulary size")
args.add_argument("--edim", type = int, default = 8, help = "embedding dimension")
args.add_argument("--maxlen", type = str, deafult = 20, help = "maximum length of sequences")
args.add_argument("--minlen", type = str, deafult = 20, help = "minimum length of sequences")
args.add_argument("--num", type = int, default = 1000, help = "number of samples of dummy language")
args.add_argument("--batch_size", type = int, default = 32, help = "batch size for training")
args.add_argument("--lr", type = float, default = 0.001, help = "batch_size in training")
args.add_argument("--epochs", type = int, default = 3, help = "number of training epochs")
args = args.parse_args()

dummy_language = np.random.