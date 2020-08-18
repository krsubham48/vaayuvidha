# Tests

This repo has quick scripts used to make architecture tests on all levels.

1. [`text_rnn.py`](text_rnn.py): Simple one-one LSTM network, see how the predictions have changed after 50 epochs
```
➜  vaayuvidha git:(master) ✗ python3 tests/text_rnn.py
103
----- BEFORE -----
target: tensor([101,  77,  46,  32,  90,  61,  69,   6,  65,  73,  56,  48,  93,  41, 18, 102, 100, 100])
output: tensor([  3,  35,  42,  71,  76,   2,  47,   9,  50,  34,  40,  25,  85,  28, 24, 100,  99,  99,  78])
----- LEARNING -----
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [02:53<00:00,  3.48s/it]
----- AFTER -----
target: tensor([101,  77,  46,  32,  90,  61,  69,   6,  65,  73,  56,  48,  93,  41, 18, 102, 100, 100, 100])
output: tensor([ 12,  16,  36,  36,  34,  34,  51,  35,  74,  23,  23, 102, 102, 102, 102, 100, 100, 100, 100])
```

2. [`rnn_gen.py`](rnn_gen.py): LSTM used with hidden state to generate dummy sequences
```
➜  vaayuvidha git:(master) ✗ python3 tests/rnn_gen.py
----- BEFORE -----
target: [101  77  46  32  90  61  69   6  65  73  56  48  93  41  18 102 100 100 100]
output: [ 9 24 59 50 50 59 50  9 50 59 59 59 59 59 59 59 59 59 59]
Epoch: 300/300; loss: 49.135: 100%|███████████████████████████████| 300/300 [02:41<00:00,  1.86it/s]
----- AFTER -----
target: [101  77  46  32  90  61  69   6  65  73  56  48  93  41  18 102 100 100 100]
output: [ 44  13  18  15  17  45 102 102 102  38 102 102 102 102 102 100 100 100 100]
```

3. [`test_gnn.py`](test_gnn.py): Simple 1-layer autoencoder like GNN network
```
(vaayuvidha) ➜  vaayuvidha git:(master) ✗ python3 tests/test_gnn.py
GCNConv(
  (lin): Linear(in_features=4, out_features=32, bias=True)
  (lin2): Linear(in_features=32, out_features=4, bias=True)
)
----- BEFORE -----
Batch: Batch(batch=[90], edge_index=[2, 788], i0=[90, 1], i1=[90, 1], i2=[90, 1], i3=[90, 1], t0=[90, 1], t1=[90, 1], t2=[90, 1], t3=[90, 1])
out_shape: torch.Size([90, 4])
targ_shape: torch.Size([90, 4])
tensor([0.6977, 0.2161, 0.9763, 0.0062]) tensor([ 0.0017,  0.0949, -0.0846,  0.2044])
----- LEARNING -----
Epoch: 50/50; loss: 0.251: 100%|████████████████████████████████████| 50/50 [00:27<00:00,  1.85it/s]
----- AFTER -----
Batch: Batch(batch=[90], edge_index=[2, 788], i0=[90, 1], i1=[90, 1], i2=[90, 1], i3=[90, 1], t0=[90, 1], t1=[90, 1], t2=[90, 1], t3=[90, 1])
out_shape: torch.Size([90, 4])
targ_shape: torch.Size([90, 4])
tensor([0.6977, 0.2161, 0.9763, 0.0062]) tensor([0.4981, 0.4922, 0.4844, 0.4842])
```

4. [`test_temporal_gnn.py`](test_temporal_gnn.py): Simple temporal GNN on fixed graph
