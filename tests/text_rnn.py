# simple one-to-one RNN experiment that takes in a word and learns to predict the next word but
# without a continuing state
# quick tutorial: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# difference between LSTM and LSTMCell: https://stackoverflow.com/questions/57048120/pytorch-lstm-vs-lstmcell
# this sample uses the LSTM

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import ArgumentParser
from tqdm import trange

# set seeds for reproducibility
np.random.seed(4)
torch.manual_seed(4)

args = ArgumentParser(description="test rnn")
args.add_argument("--vocab_size", type = int, default = 100, help = "vocabulary size")
args.add_argument("--edim", type = int, default = 8, help = "embedding dimension")
args.add_argument("--maxlen", type = str, default = 20, help = "maximum length of sequences")
args.add_argument("--minlen", type = str, default = 4, help = "minimum length of sequences")
args.add_argument("--num", type = int, default = 1000, help = "number of samples of dummy language")
args.add_argument("--batch_size", type = int, default = 8, help = "batch size for training")
args.add_argument("--lr", type = float, default = 0.001, help = "batch_size in training")
args.add_argument("--epochs", type = int, default = 50, help = "number of training epochs")
args = args.parse_args()

special_tokens = {"[pad]": args.vocab_size, "[bos]": args.vocab_size+1, "[eos]": args.vocab_size+2}
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
args.vocab_size = args.vocab_size + 3

dummy_language = np.array(dummy_language)
# for x in dummy_language[:20]:
#     print(x)

# --- Making the model --- #
class SimpleLSTM(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        print(config.vocab_size)
        self.wte = nn.Embedding(config.vocab_size, config.edim)
        self.lstm = nn.LSTM(config.edim, config.edim)
        self.lm_head = nn.Linear(config.edim, config.vocab_size)

    def forward(self, sentence, _flat = True):
        # for shapes assume the input has size torch.Size([3, 20])
        input_shape = sentence.shape
        embeds = self.wte(sentence) # torch.Size([3, 20, 8])
        embeds = embeds.view(input_shape[0]*input_shape[1], 1, -1)  # torch.Size([20*3, 1, 8])
        lstm_out, _ = self.lstm(embeds)
        out = self.lm_head(lstm_out.view(input_shape[0], input_shape[1], -1)) # [3,20,vocab_size]
        lm_scores = F.log_softmax(out, dim=1)
        return lm_scores

# make model and loss fn
model = SimpleLSTM(args)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# run one dummy epoch
print("----- BEFORE -----")
with torch.no_grad():
    b = torch.tensor(dummy_language[:3])
    inputs = b[:,:-1]
    # print(inputs.shape) # torch.Size([3, 19])
    tag_scores = model(inputs)
    # print(tag_scores.shape) # torch.Size([3, 19, 103])
    print("target:", inputs[0, :-1])
    print("output:", torch.argmax(tag_scores[0], dim = -1))

# to train the network
print("----- LEARNING -----")
for e in trange(args.epochs):
    for i in range(len(dummy_language)):
        model.zero_grad()
        b = torch.from_numpy(dummy_language[i*args.batch_size:(i+1)*args.batch_size])
        if b.shape[0] == 0:
            continue
        to_shape = b.shape[0]*(b.shape[1]-1)
        outputs = model(b[:,:-1])
        outputs = outputs.view(to_shape, -1)
        # print("OP:", outputs.shape)
        # print(b[:, 1:].shape)
        target = b[:, 1:].reshape(to_shape,)
        # print("TR:", target.shape)
        loss = loss_function(outputs, target)
        # print("loss:", loss)
        loss.backward()
        optimizer.step()

# see what it has learned
print("----- AFTER -----")
with torch.no_grad():
    # inputs = prepare_sequence(training_data[0][0], word_to_ix)
    inputs = torch.tensor(dummy_language[:3])
    # print(inputs.shape)  # torch.Size([3, 20])
    tag_scores = model(inputs[:, :-1])
    # print(tag_scores.shape)  # torch.Size([3, 20, 103])
    print("target:", inputs[0, :-1])
    print("output:", torch.argmax(tag_scores[0], dim=-1))
