# simple RNN experiment
# quick tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html

import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser


args = ArgumentParser(description="test rnn")
args.add_argument("--vocab_size", type = int, default = 100, help = "vocabulary size")
args.add_argument("--edim", type = int, default = 8, help = "embedding dimension")
args.add_argument("--maxlen", type = str, default = 20, help = "maximum length of sequences")
args.add_argument("--minlen", type = str, default = 4, help = "minimum length of sequences")
args.add_argument("--num", type = int, default = 1000, help = "number of samples of dummy language")
args.add_argument("--batch_size", type = int, default = 32, help = "batch size for training")
args.add_argument("--lr", type = float, default = 0.001, help = "batch_size in training")
args.add_argument("--epochs", type = int, default = 3, help = "number of training epochs")
args = args.parse_args()

special_tokens = {"[pad]": args.vocab_size, "[bos]": args.vocab_size + 1, "[eos]": args.vocab_size + 2}
dummy_language = []
for seq in [
    np.random.randint(args.vocab_size, size = (l)).tolist()
    for l in np.random.randint(args.minlen, args.maxlen - 2, size = (args.num))
]:
    # sequence maxlen is -2 for special tokens
    seq = [special_tokens["[bos]"]] + seq + [special_tokens["[eos]"]]
    if len(seq) < args.maxlen:
        seq += [special_tokens["[pad]"],]*(args.maxlen - len(seq))
    dummy_language.append(seq)
args.vocab_size = args.vocab_size + 2

dummy_language = np.array(dummy_language)
for x in dummy_language[:20]:
    print(x)


# --- Making the model --- #
class RNN(nn.Module):
    def __init__(self, config):
        super(RNN, self).__init__()
        self.hidden_size = config.hidden_size
        self.i2h = nn.Linear(n_categories + config.input_size +
                             config.hidden_size, config.hidden_size)
        self.i2o = nn.Linear(n_categories + config.input_size +
                             config.hidden_size, config.output_size)
        self.o2o = nn.Linear(config.hidden_size + config.output_size, config.output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
