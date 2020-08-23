# implementing sequence to sequence RNN
# tutorial: https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
# difference between LSTM and LSTMCell: https://stackoverflow.com/questions/57048120/pytorch-lstm-vs-lstmcell
# this sample uses the LSTMCell

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from argparse import ArgumentParser
from torch.nn.modules.rnn import LSTMCell
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
args.add_argument("--batch_size", type = int, default = 32, help = "batch size for training")
args.add_argument("--lr", type = float, default = 0.001, help = "batch_size in training")
args.add_argument("--epochs", type = int, default = 300, help = "number of training epochs")
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
class SimpleLSTMCell(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.edim)
        self.cell = LSTMCell(config.edim, config.edim)
        self.lm_head = nn.Linear(config.edim, config.vocab_size)

    def forward(self, sentence, hidden_state = None):
        # for shapes assume the input has size torch.Size([3, 1])
        input_shape = sentence.shape
        seqlen = input_shape[1]
        embeds = self.wte(sentence)  # torch.Size([3, 1, 8])
        # print(embeds.shape) # torch.Size([3, 1, 8])
        embeds = embeds.view(input_shape[0]*input_shape[1], -1)
        # print(embeds.shape)  # torch.Size([3, 1, 8])
        lstm_out, lstm_hidden = self.cell(embeds, hidden_state)
        out = self.lm_head(lstm_out.view(input_shape[0], -1))  # [3,20,vocab_size]
        lm_scores = F.log_softmax(out, dim=1)
        return lm_scores, (lstm_out, lstm_hidden)


# make model and loss fn
model = SimpleLSTMCell(args)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# run one dummy epoch
print("----- BEFORE -----")
with torch.no_grad():
    b = dummy_language[:3]
    inputs = b[:, :-1]
    # print(inputs.shape) # torch.Size([3, 19])
    output = []
    # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero
    # https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html#torch.nn.LSTMCell
    hidden_state = None
    for i in range(inputs.shape[1]):
        input_this = np.asarray([seq[i] for seq in b]).reshape(3, 1)
        tag_scores, hidden_state = model(torch.from_numpy(input_this), hidden_state)
        output.append(tag_scores.numpy())
    output = np.asarray(output)  # (19, 3, 103)
    output = np.transpose(output, (1, 0, 2))  # (3, 19, 103)
    print("target:", inputs[0])
    print("output:", np.argmax(output[0], axis=-1))

# let's train this bitch
pbar = trange(args.epochs, ncols = 100)
prev_loss = -1
for e in pbar:
    this_ep_loss = []
    pbar.set_description(f"Epoch: {e+1}/{args.epochs}; loss: {round(prev_loss, 3)}")
    for i in range(len(dummy_language)):
        b = torch.from_numpy(dummy_language[i*args.batch_size:(i+1)*args.batch_size])
        if b.shape[0] == 0:
            continue
        model.zero_grad()
        hidden_state = None
        loss = 0
        for i in range(inputs.shape[1]-1):
            input_this = np.asarray([seq[i] for seq in b]).reshape(len(b), 1)
            target = torch.from_numpy(np.asarray([seq[i + 1] for seq in b]))
            tag_scores, hidden_state = model(torch.from_numpy(input_this), hidden_state)
            loss += loss_function(tag_scores, target)
        this_ep_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    prev_loss = sum(this_ep_loss)/len(this_ep_loss)
    

print("----- AFTER -----")
with torch.no_grad():
    b = dummy_language[:3]
    inputs = b[:, :-1]
    # print(inputs.shape) # torch.Size([3, 19])
    output = []
    hidden_state = None
    for i in range(inputs.shape[1]):
        input_this = np.asarray([seq[i] for seq in b]).reshape(3, 1)
        tag_scores, hidden_state = model(torch.from_numpy(input_this), hidden_state)
        output.append(tag_scores.numpy())
    # print(tag_scores.shape) # torch.Size([3, 19, 103])
    output = np.asarray(output)
    print(output.shape)  # (19, 3, 103)
    output = np.transpose(output, (1, 0, 2))
    print(output.shape)  # (3, 19, 103)
    print("target:", inputs[0])
    print("output:", np.argmax(output[0], axis=-1))
