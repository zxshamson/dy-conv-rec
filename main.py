import os
import sys
import random
import time
import collections
import torch
import argparse
import numpy as np
from torch import nn, optim
import torch.utils.data as data
import torch.nn.functional as F
from corpus_process import Corpus
from data_process import MyDataset, form_dataset
from data_process import create_pretrain_embeddings, create_tensor, create_history_tensor, create_arc_info
from config import parse_config
from rank_eval import cal_map, cal_ndcg_all, cal_precision_N, cal_mrr
from Modules import DCR


def weighted_binary_cross_entropy(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * target * torch.log(torch.clamp(output, min=1e-15, max=1)) + \
            weights[0] * (1 - target) * torch.log(torch.clamp(1 - output, min=1e-15, max=1))
    else:
        loss = target * torch.log(torch.clamp(output, min=1e-15, max=1)) + \
            (1 - target) * torch.log(torch.clamp(1 - output, min=1e-15, max=1))

    return torch.neg(torch.mean(loss))


def evaluate(model, eval_data, dev=False):
    model.eval()
    labels_list = collections.defaultdict(list)
    for step, one_data in enumerate(eval_data):
        label = one_data[-1].data.numpy()
        predictions = model(one_data[0])
        if torch.cuda.is_available():  # run in GPU
            pred_label = predictions.cpu().data.numpy()
            uc_pairs = one_data[0].cpu().data.numpy()
        else:
            pred_label = predictions.data.numpy()
            uc_pairs = one_data[0].data.numpy()
        for n in xrange(len(label)):
            labels_list[uc_pairs[n, 0]].append((pred_label[n], label[n]))
    res_map = cal_map(labels_list)
    if dev:
        return res_map
    else:
        res_p1 = cal_precision_N(labels_list, 1)
        res_p5 = cal_precision_N(labels_list, 5)
        res_ndcg5 = cal_ndcg_all(labels_list, 5)
        res_ndcg10 = cal_ndcg_all(labels_list, 10)
        res_mrr = cal_mrr(labels_list)
        return res_map, res_p1, res_p5, res_ndcg5, res_ndcg10, res_mrr


def train_epoch(model, train_data, loss_weights, optimizer, epoch):
    model.train()
    start = time.time()
    print('Epoch: %d start!' % epoch)
    avg_loss = 0.0
    count = 0
    for step, one_data in enumerate(train_data):
        label = one_data[-1]
        if torch.cuda.is_available():  # run in GPU
            label = label.cuda()
        predictions = model(one_data[0])
        loss = weighted_binary_cross_entropy(predictions, label, loss_weights)
        avg_loss += loss.item()
        count += 1
        if count % 100000 == 0:
            print('Epoch: %d, iterations: %d, loss: %g' % (epoch, count, loss.item()))
        optimizer.zero_grad()
        loss.backward()
        # if clip != -1:
        #    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    avg_loss /= len(train_data)
    end = time.time()
    print('Epoch: %d done! Train avg_loss: %g! Using time: %.2f minutes!' % (epoch, avg_loss, (end-start)/60))
    return avg_loss


def find_files(filename, month_num=4):
    file_prefix = 'Datafiles/' + filename + '_20150'
    train_files = []
    for m in range(month_num):
        train_files.append(file_prefix + str(m+1) + '.data')
    test_file = file_prefix + '5_test.data'
    valid_file = file_prefix + '5_valid.data'
    return train_files, test_file, valid_file


def train(config):
    print 'Start training: ' + config.filename + ' ' + config.modelname
    trainfile, testfile, validfile = find_files(config.filename, config.month_num)
    modelname = config.modelname
    # Process the corpus and prepare pretrained parameters (if any)
    corp = Corpus(trainfile, testfile, validfile)
    config.user_num, config.conv_num, config.vocab_num = corp.userNum, corp.convNum, corp.wordNum
    train_convs, train_convs_per_user, test_data, dev_data = form_dataset(corp, config.batch_size)
    conv_data = create_tensor(corp.convs)
    if config.pretrained_file == 'NULL':
        embedding_matrix = None
    else:
        embedding_matrix = create_pretrain_embeddings(corp, config.embedding_dim, config.pretrained_file)
    # Set the model and saving path
    if modelname == 'DCR':
        user_history = create_history_tensor(corp.userNum, corp.user_history)
        arcs = create_arc_info(conv_data, no_time=True)
        model = DCR(config, conv_data, user_history, arcs, embedding_matrix)
        path_name = str(config.batch_size) + "_" + str(config.factor_dim) + "_" + str(config.embedding_dim) + "_" + \
            str(config.kernal_num) + "_" + str(config.hidden_dim) + "_" + str(config.gcn_layers_num) + "_" + \
            str(config.neg_sample_num) + "_" + str(config.lr) + "_" + str(int(config.pos_weight)) + "_" + config.att
        if config.month_num != 4:
            path_name += "_" + str(config.month_num) + "m_" + str(config.runtime)
        else:
            path_name += "_" + str(config.runtime)
        if config.pretrained_file == 'NULL':
            path_name += "_npembed"
        if config.no_lstm:
            path_name += "_nlstm"
        if config.no_gcn:
            path_name += "_ngcn"
        if config.mlp_layers_num == 0:
            path_name += "_nmlp"
    else:
        print 'Modelname Wrong!'
        exit()
    res_path = "BestResults/" + modelname + "/" + config.filename + "/"
    mod_path = "BestModels/" + modelname + "/" + config.filename + "/"
    if not os.path.isdir(res_path):
        os.makedirs(res_path)
    if not os.path.isdir(mod_path):
        os.makedirs(mod_path)
    mod_path += path_name + '.model'
    res_path += path_name + '.data'

    # Set the optimizing parameters
    loss_weights = torch.Tensor([1, config.pos_weight])
    if torch.cuda.is_available():  # run in GPU
        model = model.cuda()
        loss_weights = loss_weights.cuda()
    if config.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 / ((epoch + 1) ** 0.5))
    best_dev_map = 0.0
    best_epoch = -1
    best_epoch_loss = 999999
    no_improve = 0
    # Begin training
    for epoch in range(config.max_epoch):
        train_data = MyDataset(train_convs, train_convs_per_user, train=True, num_sampling=config.neg_sample_num)
        train_loader = data.DataLoader(train_data, batch_size=config.batch_size, num_workers=0, shuffle=True)
        loss = train_epoch(model, train_loader, loss_weights, optimizer, epoch)
        dev_map = evaluate(model, dev_data, dev=True)
        if dev_map > best_dev_map:
            no_improve = 0
            best_dev_map = dev_map
            os.system('rm ' + mod_path)
            best_epoch = epoch
            best_epoch_loss = loss
            print('New Best Dev!!! MAP: %g' % best_dev_map)
            torch.save(model.state_dict(), mod_path)
        else:
            no_improve += 1
            print('Current Best Dev MAP: %g, Dev MAP: %g' % (best_dev_map, dev_map))
        if no_improve > 8:
            break
        scheduler.step()
    model.load_state_dict(torch.load(mod_path))
    # Evaluate and save results
    res = evaluate(model, test_data)
    print('Result in test set: MAP: %g, Precision@1: %g, Precision@5: %g, nDCG@5: %g, nDCG@10: %g, MRR: %g' %
          (res[0], res[1], res[2], res[3], res[4], res[5]))
    with open(res_path, 'w') as f:
        f.write('MAP\tPre@1\tPre@5\tnDCG@5\tnDCG@10\tMRR\n')
        f.write('%g\t%g\t%g\t%g\t%g\t%g\n' % (res[0], res[1], res[2], res[3], res[4], res[5]))
        if modelname != "Pop" and modelname != "Random":
            f.write('Dev MAP: %g\n' % best_dev_map)
            f.write('Best epoch: %d\n' % best_epoch)
            f.write('Best epoch loss: %g\n' % best_epoch_loss)


if __name__ == '__main__':
    config = parse_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_dev
    train(config)



