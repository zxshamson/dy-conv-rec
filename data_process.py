import random
import collections
import torch
import numpy as np
import torch.utils.data as data
from itertools import chain
from corpus_process import Corpus


class MyDataset(data.Dataset):
    def __init__(self, convs, convs_per_user, train=True, num_sampling=5):
        self.uc_pairs = []
        self.data_label = []
        if train:
            for u in convs_per_user.keys():
                neg_convs = list(convs - convs_per_user[u])
                num = len(neg_convs)
                for c in convs_per_user[u]:
                    self.uc_pairs.append((u, c))
                    self.data_label.append(1)
                    for t in xrange(num_sampling):
                        j = np.random.randint(num)
                        self.uc_pairs.append((u, neg_convs[j]))
                        self.data_label.append(0)
        else:
            for u in convs_per_user.keys():
                for c in convs_per_user[u]:
                    self.uc_pairs.append((u, c))
                    self.data_label.append(1)
                neg_convs = list(convs - convs_per_user[u])
                num = len(neg_convs)
                for t in xrange(100):
                    j = np.random.randint(num)
                    self.uc_pairs.append((u, neg_convs[j]))
                    self.data_label.append(0)

        self.uc_pairs = torch.LongTensor(self.uc_pairs)
        self.data_label = torch.Tensor(self.data_label)

    def __getitem__(self, idx):
        return self.uc_pairs[idx], self.data_label[idx]

    def __len__(self):
        return len(self.data_label)


def form_dataset(corp, batch_size):
    train_convs = corp.convs_in_train
    test_convs = corp.convs_in_test
    valid_convs = corp.convs_in_valid

    train_convs_per_user = collections.defaultdict(set)
    for u in corp.user_history.keys():
        convs = [msg[0] for msg in corp.user_history[u]]
        train_convs_per_user[u] = set(convs)

    test_data = MyDataset(test_convs, corp.test_pred, train=False)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, num_workers=0, shuffle=True)
    dev_data = MyDataset(valid_convs, corp.valid_pred, train=False)
    dev_loader = data.DataLoader(dev_data, batch_size=batch_size, num_workers=0, shuffle=True)
    return train_convs, train_convs_per_user, test_loader, dev_loader


def create_tensor(convs):
    conv_turn_size = max([len(convs[c]) for c in convs.keys()])
    conv_sent_len = max([len(sent) for sent in chain.from_iterable([convs[c] for c in convs.keys()])])
    text_vec = []
    for key in sorted(convs.keys()):
        t = []
        for sent in convs[key]:
            pad_len = max(0, conv_sent_len - len(sent))
            t.append(sent + [0] * pad_len)
        pad_size = max(0, conv_turn_size - len(t))
        t.extend([[-1] * 4 + [0] * (conv_sent_len - 4)] * pad_size)
        text_vec.append(t)
    return torch.LongTensor(text_vec)


def create_history_tensor(user_num, user_history):
    msg_num = max([len(user_history[u]) for u in user_history.keys()])
    msg_num = 10000 if msg_num > 10000 else msg_num
    msg_vec = []
    for u in range(user_num):
        m = user_history[u][-10000:] if len(user_history[u]) != 0 else []
        pad_size = max(0, msg_num - len(m))
        m.extend([[-1, 2e18, -1]] * pad_size)
        msg_vec.append(m)
    return torch.LongTensor(msg_vec)


def create_arc_info(convs, no_time=True):  # "no_time=True" indicates that we only care about replying relations
    conv_num = convs.size(0)
    turn_num = convs.size(1)
    arcs = torch.zeros((2, conv_num, turn_num, turn_num))  # arcs[0]: in arcs, arcs[1]: out arcs
    for i in range(conv_num):
        for turn in convs[i]:
            if turn[0] == -1:
                break
            if turn[2] == 0:
                continue
            if turn[3] >= turn_num:  # For some presenting sequence mistakes
                turn[3] = 0
            # replying relations
            arcs[0, i, turn[2], turn[3]] += 1
            arcs[1, i, turn[3], turn[2]] += 1
            # time series relations
            if not no_time:
                arcs[0, i, turn[2], turn[2]-1] += 1
                arcs[1, i, turn[2]-1, turn[2]] += 1

    return arcs


def create_pretrain_embeddings(corp, embedding_dim=300, filename=None):  # we defaultly use 'glove.840B.300d.txt' as pretrained file, you can specify other pretrained files
    if embedding_dim == 300 and filename is None:
        pretrain_file = 'glove.840B.300d.txt'
    elif filename:
        pretrain_file = filename
    else:
        print "creating pretrained embeddings fail!"
        exit(0)
    pretrain_words = {}
    with open(pretrain_file, 'r') as f:
        for line in f:
            infos = line.split()
            wd = infos[0]
            vec = np.array(infos[1:]).astype(np.float)
            pretrain_words[wd] = vec
    word_idx = corp.r_wordIDs
    vocab_num = corp.wordNum
    weights_matrix = np.zeros((vocab_num, embedding_dim))
    for idx in word_idx.keys():
        try:
            weights_matrix[idx] = pretrain_words[word_idx[idx]]
        except KeyError:
            weights_matrix[idx] = np.random.normal(size=(embedding_dim,))
    if torch.cuda.is_available():  # run in GPU
        return torch.Tensor(weights_matrix).cuda()
    else:
        return torch.Tensor(weights_matrix)


if __name__ == '__main__':
    train_files = ['Datafiles/technology_201501.data', 'Datafiles/technology_201502.data',
                   'Datafiles/technology_201503.data', 'Datafiles/technology_201504.data']
    test_file = 'Datafiles/technology_201505_test.data'
    valid_file = 'Datafiles/technology_201505_valid.data'
    corp = Corpus(train_files, test_file, valid_file)
    print corp.userNum, len(corp.user_history.keys())
    his = create_history_tensor(corp.userNum, corp.user_history)
    print his.size()
    print his[-1]
