import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, use_gates=True):
        super(GCNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.sigmoid = nn.Sigmoid()
        self.use_gates = use_gates

        self.W_in = nn.Linear(in_features, out_features)
        self.W_out = nn.Linear(in_features, out_features)
        self.W_self = nn.Linear(in_features, out_features)

        if self.use_gates:
            self.V_in_gate = nn.Parameter(torch.randn(out_features))
            self.b_in_gate = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.b_in_gate, 1)
            self.V_out_gate = nn.Parameter(torch.randn(out_features))
            self.b_out_gate = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.b_out_gate, 1)
            self.V_self_gate = nn.Parameter(torch.randn(out_features))
            self.b_self_gate = nn.Parameter(torch.Tensor(1))
            nn.init.constant_(self.b_self_gate, 1)

    def forward(self, reps, arc_in, arc_out):

        batch_size = reps.size(0)
        turn_num = reps.size(1)
        arc_self = torch.eye(turn_num).repeat(batch_size, 1, 1)
        if torch.cuda.is_available():  # run in GPU
            arc_self = arc_self.cuda()

        hidden_in = self.W_in(reps)
        hidden_out = self.W_out(reps)
        hidden_self = self.W_self(reps)

        if self.use_gates:
            in_gate = torch.mul(hidden_in, self.V_in_gate) + self.b_in_gate
            hidden_in = hidden_in * self.sigmoid(in_gate)
            out_gate = torch.mul(hidden_out, self.V_out_gate) + self.b_out_gate
            hidden_out = hidden_out * self.sigmoid(out_gate)
            self_gate = torch.mul(hidden_self, self.V_self_gate) + self.b_self_gate
            hidden_self = hidden_self * self.sigmoid(self_gate)

        hidden_in = torch.bmm(arc_in, hidden_in)
        hidden_out = torch.bmm(arc_out, hidden_out)
        hidden_self = torch.bmm(arc_self, hidden_self)
        return hidden_in + hidden_out + hidden_self


class GCN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=0, num_layers=1, use_gates=True, dropout=0.2):
        super(GCN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.use_gates = use_gates
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.gcn_layers = []

        if self.num_layers == 1:
            gcn = GCNLayer(in_dim, out_dim, use_gates)
            self.gcn_layers.append(gcn)
        else:
            gcn = GCNLayer(in_dim, hidden_dim, use_gates)
            self.gcn_layers.append(gcn)
            for i in range(1, self.num_layers - 1):
                gcn = GCNLayer(hidden_dim, hidden_dim, use_gates)
                self.gcn_layers.append(gcn)
            gcn = GCNLayer(hidden_dim, out_dim, use_gates)
            self.gcn_layers.append(gcn)
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

    def forward(self, reps, arc_in, arc_out):
        out_reps = reps
        for i in range(self.num_layers - 1):
            out_reps = self.gcn_layers[i](out_reps, arc_in, arc_out)
            out_reps = self.dropout(self.relu(out_reps))
        out_reps = self.gcn_layers[self.num_layers - 1](out_reps, arc_in, arc_out)
        return out_reps



