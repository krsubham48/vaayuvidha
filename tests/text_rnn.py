# simple RNN experiment
# quick tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
# difference between LSTM and LSTMCell: https://stackoverflow.com/questions/57048120/pytorch-lstm-vs-lstmcell

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
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
class SimpleLSTM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.edim)
        self.lstm = nn.LSTM(config.edim, config.edim)
        self.lm_head = nn.Linear(config.edim, config.vocab_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        head_space = self.lm_head(lstm_out.view(len(sentence), -1))
        lm_scores = F.log_softmax(head_space, dim=1)
        return lm_scores

# make model and loss fn
model = SimpleLSTM(args)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print(model)
print(dummy_language[0][0])
