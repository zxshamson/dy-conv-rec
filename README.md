# Introduction:
This is the implementation in PyTorch for my ACL2020 paper:

"Dynamic Online Conversation Recommendation"

# Requirement:

* Python: 2.7+

* Pytorch: 1.1.0+

# Before running:
You need to download Glove pre-training embeddings from: 
https://nlp.stanford.edu/projects/glove/

"glove.840B.300d.txt" for default.

You can also specify your own pre-trained file by using parameter "pretrained_file".

# Usage:

`python main.py [filename] [modelname]`

```
[filename]: "technology" or "todayilearned" or "funny".
[modelname]: "DCR".

optional arguments:
  --cuda_dev          choose to use which GPU (default: "0")
  --factor_dim        dimension for user factor modeling (default: 20)
  --neg_sample_num    sampling numbers for negative instances each positive instance when training (default: 5)
  --kernal_num        number of kernals for CNN encoder (default: 100)
  --kernal_kind       kind of kernals for CNN encoder (default: 1)
  --embedding_dim     dimension for word embedding (default: 300)
  --hidden_dim        dimension for hidden states (default: 200)
  --batch_size        batch size (default: 256)
  --max_epoch         maximum iteration times (default: 100)
  --lr                initial learning rate (default: 0.001)
  --dropout           dropout rate (default: 0.2)
  --mlp_layers_num    number of layers for MLP (default: 2)
  --gcn_layers_num    number of layers for GCN (default: 1)
  --runtime           record the current running time (default: 0)
  --pos_weight        weights for positive instances in loss function (default: 100)
  --optim             training optimizer (default: "adam", choices: "adam", "sgd")
  --att               attention mechanism (default: "u", choices: "u", "n")
  --month_num         number of months using for training data (default: 4)
  --pretrained_file   specify your word embedding pretrained file (default: "glove.840B.300d.txt", "NULL" for no pretrained)
  --no_lstm           without LSTM module (action="store_true")
  --no_gcn            without GCN module (action="store_true")
```

# Datasets:

Including "technology", "todayilearned" and "funny".

Each dataset contains 6 files. "201501" - "201504" are train files, "201505\_test" is test file, "201505\_valid" is validation file.


Format for each line in train files:

[Conv ID] \t [Msg ID] \t [Parent ID] \t [Original sentence] \t [words after preprocessing] \t [User ID] \t [posting time] \t [ups num] \t [downs num]

(In our model, we don't utilize [ups num] and [downs num], but we think it can benefit future work.)

Format for each line in test/validation file:

There are two kinds of lines:

1. Conversation messages whose format just as in train files; 
2. Prediction lines in the form of "[Conv ID] \t [User IDs]", following the last message in that converation. Each User ID is the user that would join the conversation later.
