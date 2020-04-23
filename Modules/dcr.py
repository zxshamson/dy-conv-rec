import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from cnnencoder import CNNEncoder
from gcn import GCN
import torch.nn.utils.rnn as rnn_utils


class DCR(nn.Module):
    def __init__(self, config, conv_data, user_history, arcs, pretrained_embed=None, his_num=50):
        super(DCR, self).__init__()
        self.user_num = config.user_num
        self.conv_num = config.conv_num
        self.vocab_num = config.vocab_num
        self.user_factor_dim = config.factor_dim
        self.kernal_num = config.kernal_num
        self.kernal_kind = config.kernal_kind
        self.embed_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.conv_data = conv_data
        self.user_history = user_history
        self.arc_in = arcs[0]
        self.arc_out = arcs[1]
        self.att = config.att
        self.mlp_layers_num = config.mlp_layers_num
        self.gcn_layers_num = config.gcn_layers_num
        self.use_gates = True
        self.use_lstm = not config.no_lstm
        self.use_gcn = not config.no_gcn
        self.his_num = his_num
        # user embedding layer
        if self.user_factor_dim != 0:
            self.user_embedding = nn.Embedding(self.user_num, self.user_factor_dim)
            self.factor_to_hidden = nn.Linear(self.user_factor_dim, self.hidden_dim)
        # word embedding layer
        self.word_embedding = nn.Embedding(self.vocab_num, self.embed_dim, padding_idx=0)
        if pretrained_embed is not None:
            self.word_embedding.load_state_dict({'weight': pretrained_embed})
        # turn modeling layer
        self.turn_modeling = CNNEncoder(self.embed_dim, self.kernal_num, self.kernal_kind, config.dropout)
        turn_hidden_dim = self.kernal_num * 3
        # conv modeling layer
        if self.use_lstm:
            self.seq_modeling = nn.GRU(turn_hidden_dim + self.user_factor_dim, self.hidden_dim // 2, bidirectional=True)
            if self.use_gcn:
                self.conv_modeling = GCN(self.hidden_dim, self.hidden_dim, self.hidden_dim, self.gcn_layers_num,
                                         self.use_gates, config.dropout)
        else:
            self.conv_modeling = GCN(turn_hidden_dim + self.user_factor_dim, self.hidden_dim, self.hidden_dim,
                                     self.gcn_layers_num, self.use_gates, config.dropout)
        # user modeling layer
        self.user_dynamic_modeling = nn.GRU(turn_hidden_dim, self.hidden_dim, bidirectional=False)
        # output layer
        if self.mlp_layers_num != 0:
            self.mlp1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self.mlp2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
            self.out_layer = nn.Linear(self.hidden_dim // 2, 1)
        else:
            self.out_layer = nn.Linear(self.hidden_dim, 1)
        # activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, uc_pairs):
        userids = uc_pairs[:, 0]
        convids = uc_pairs[:, 1]
        history = self.user_history[userids]
        convs = self.conv_data[convids]
        arc_in = self.arc_in[convids]
        arc_out = self.arc_out[convids]
        if torch.cuda.is_available():  # run in GPU
            userids = userids.cuda()
            history = history.cuda()
            convs = convs.cuda()
            arc_in = arc_in.cuda()
            arc_out = arc_out.cuda()
        batch_size = convs.size(0)

        # user history modeling preprocessing: find the latest history messages before target conv for each user
        his_msgs = []
        msg_num_per_user = []
        for i in range(batch_size):
            his_inds = (history[i, :, 1] < convs[i, 0, 1]).nonzero().view(-1)  # each history message is in the form of [convID, posttime, turnID]
            his_inds = his_inds[-self.his_num:]  # only encode the latest his_num history messages
            if len(his_inds) == 0:
                cur_his_msgs = torch.LongTensor([])
            else:
                his_before_conv = history[i][his_inds]
                cur_his_msgs = torch.cat([self.conv_data[his_before_conv[j, 0], his_before_conv[j, 2], 4:].view(1, -1)
                                          for j in range(len(his_before_conv))], dim=0)  # each turn in conv_data is in the form of [userID, posttime, turnID, parentID, words]
            if torch.cuda.is_available():  # run in GPU
                cur_his_msgs = cur_his_msgs.cuda()
            his_msgs.append(cur_his_msgs)
            msg_num_per_user.append(len(cur_his_msgs))
        msg_num_per_user = torch.LongTensor(msg_num_per_user)
        if torch.cuda.is_available():  # run in GPU
            msg_num_per_user = msg_num_per_user.cuda()

        if len(msg_num_per_user.nonzero()) == 0:  # all users with zero history
            if self.user_factor_dim != 0:
                user_out = self.tanh(self.factor_to_hidden(self.user_embedding(userids)))
            else:
                user_out = self.tanh(torch.randn((batch_size, self.hidden_dim)))
                if torch.cuda.is_available():  # run in GPU
                    user_out = user_out.cuda()
        else:
            sorted_msg_num_per_user, sorted_indices = torch.sort(msg_num_per_user, descending=True)
            _, desorted_indices = torch.sort(sorted_indices, descending=False)
            sorted_userids = userids[sorted_indices]
            ind = sorted_msg_num_per_user.nonzero()[-1]  # find users with zero history
            # user history modeling 1: encode each message into distinct representations
            all_msg = torch.cat([his_msgs[sorted_indices[i]] for i in range(ind+1)], dim=0)
            all_msg_reps = self.turn_modeling(self.word_embedding(all_msg))
            user_msg_reps = torch.split(all_msg_reps, list(sorted_msg_num_per_user[:ind+1]))
            # user history modeling 2: one GRU layer to model the dynamic of users
            paded_user_reps = rnn_utils.pad_sequence(user_msg_reps)
            packed_user_reps = rnn_utils.pack_padded_sequence(paded_user_reps, sorted_msg_num_per_user[:ind+1])
            if self.user_factor_dim != 0:
                user_init_hidden = self.tanh(self.factor_to_hidden(self.user_embedding(sorted_userids)))
            else:
                user_init_hidden = self.tanh(torch.randn((batch_size, self.hidden_dim)))
                if torch.cuda.is_available():  # run in GPU
                    user_init_hidden = user_init_hidden.cuda()
            user_states, user_out = self.user_dynamic_modeling(packed_user_reps, user_init_hidden[:ind+1].unsqueeze(0))
            user_out = user_out.squeeze(0)
            user_out = torch.cat([user_out, user_init_hidden[ind + 1:]], dim=0)
            user_out = user_out[desorted_indices]

        # conv modeling 1: encode each turn into distinct representations
        turn_num_per_conv = (convs[:, :, 0] >= 0).sum(dim=1)
        all_turn = torch.cat([convs[i, :turn_num_per_conv[i], 4:] for i in range(batch_size)], dim=0)
        all_turn_uids = torch.cat([convs[i, :turn_num_per_conv[i], 0] for i in range(batch_size)], dim=0)
        all_turn_reps = self.turn_modeling(self.word_embedding(all_turn))
        if self.user_factor_dim != 0:
            all_turn_reps = torch.cat([self.user_embedding(all_turn_uids), all_turn_reps], dim=1)
        conv_turn_reps = torch.split(all_turn_reps, list(turn_num_per_conv))

        # conv modeling 2: BiGRU+GCN to model the structure of convs
        if self.use_lstm:
            paded_conv_reps = rnn_utils.pad_sequence(conv_turn_reps)
            packed_conv_reps = rnn_utils.pack_padded_sequence(paded_conv_reps, turn_num_per_conv, enforce_sorted=False)
            packed_conv_out, _ = self.seq_modeling(packed_conv_reps)
            conv_out = rnn_utils.pad_packed_sequence(packed_conv_out, batch_first=True)[0]
            if self.use_gcn:
                t = conv_out.size(1)
                conv_out = self.conv_modeling(conv_out, arc_in[:, :t, :t], arc_out[:, :t, :t])
        else:
            paded_conv_reps = rnn_utils.pad_sequence(conv_turn_reps, batch_first=True)
            t = paded_conv_reps.size(1)
            conv_out = self.conv_modeling(paded_conv_reps, arc_in[:, :t, :t], arc_out[:, :t, :t])

        # conv modeling 3: attention mechanism
        if self.att == 'u':  # user-aware attention
            if torch.cuda.is_available():  # run in GPU
                masks = torch.where(convs[:, :, 0] >= 0, torch.Tensor([0.]).cuda(), torch.Tensor([-np.inf]).cuda())
            else:
                masks = torch.where(convs[:, :, 0] >= 0, torch.Tensor([0.]), torch.Tensor([-np.inf]))
            masks = masks[:, :conv_out.size(1)]
            query = user_out.unsqueeze(1)
            att_weights = (conv_out * query).sum(-1) + masks
            att_weights = F.softmax(att_weights, 1)
            final_conv_out = torch.bmm(conv_out.transpose(1, 2), att_weights.unsqueeze(2)).squeeze(2)
        else:  # no attention but only average pooling
            if torch.cuda.is_available():  # run in GPU
                masks = torch.where(convs[:, :, 0] >= 0, torch.Tensor([1.]).cuda(), torch.Tensor([0]).cuda())
            else:
                masks = torch.where(convs[:, :, 0] >= 0, torch.Tensor([1.]), torch.Tensor([0]))
            masks = masks[:, :conv_out.size(1)].unsqueeze(-1)
            weights = 1.0 / turn_num_per_conv.unsqueeze(-1).float()
            final_conv_out = torch.mul(torch.mul(conv_out, masks).sum(dim=1), weights)

        # output layer
        if self.mlp_layers_num != 0:
            out_reps = torch.cat([user_out, final_conv_out], dim=1)
            out_reps = self.relu(self.mlp2(self.relu(self.mlp1(out_reps))))
        else:
            out_reps = torch.mul(user_out, final_conv_out)
        final_out = torch.sigmoid(self.out_layer(out_reps).view(-1))

        return final_out





